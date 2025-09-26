# sagemaker_clone.py
from math import sqrt
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss, average_precision_score

from pathlib import Path
from urllib.parse import urlparse
import time, uuid, tarfile, json
import numpy as np, pandas as pd, xgboost as xgb
import shap, matplotlib.pyplot as plt


def _p(u):  # accepts "file://..." or plain path
    u = u.config["DataSource"]["S3DataSource"]["S3Uri"] if getattr(u, "config", None) else u
    p = urlparse(u); return Path((p.netloc + p.path + "?format=libsvm") if p.scheme == "file" else u)

# --- Estimator (SageMaker-ish) ---
class Estimator:
    """
    Minimal XGBoost 'SageMaker-like' estimator.
    - image_uri: pretend container image string
    - Input: LibSVM files (label + index:value pairs)
    - fit(...) writes: <output_path>/<base_job_name>-<instance_type>-TS-XXXXXX/output/model.tar.gz
    - explain(...): writes SHAP CSVs + PNGs + JSON report under .../explain/
    - create_model(), transformer(): mimic SageMaker APIs
    - evaluate_from_batch(...): compute metrics from Batch Transform .out predictions
    """
    def __init__(
        self,
        image_uri,
        role=None,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path="./model",
        base_job_name="xgb-job",
        hyperparameters=None,
    ):
        self.image_uri = image_uri
        self.role = role
        self.instance_count = instance_count
        self.instance_type = instance_type
        self.output_path = output_path
        self.base_job_name = base_job_name
        self.hyperparameters = dict(hyperparameters or {})
        self.model_data = None
        self._job_dir = None
        self._booster = None

    def set_hyperparameters(self, **kwargs):
        self.hyperparameters.update(kwargs)

    def _job(self):
        if self._job_dir is None:
            root = _p(self.output_path)
            jid = f"{self.base_job_name}-{self.instance_type}-{time.strftime('%Y-%m-%d-%H-%M-%S')}-{uuid.uuid4().hex[:6]}"
            self._job_dir = (root / jid / "output"); self._job_dir.mkdir(parents=True, exist_ok=True)
        return self._job_dir

    def fit(self, inputs: dict):
        # XGBoost 2.x: must specify format
        train_uri = str(_p(inputs["train"]))
        dtr = xgb.DMatrix(train_uri)
        evals = [(dtr, "train")]
        if "validation" in inputs:
            val_uri = str(_p(inputs["validation"]))
            dva = xgb.DMatrix(val_uri)
            evals.append((dva, "validation"))

        params = {**self.hyperparameters}
        num_round = int(params.pop("num_round", params.pop("num_boost_round", 200)))
        early = int(params.pop("early_stopping_rounds", 0)) or None
        verbose = params.pop("verbose_eval", True)
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "auc")
        params.setdefault("verbosity", 0)

        evals_result = {}
        self._booster = xgb.train(
            params, dtr, num_boost_round=num_round, evals=evals,
            early_stopping_rounds=early, verbose_eval=verbose, evals_result=evals_result
        )

        out = self._job()
        model_file = out / "xgboost-model"
        self._booster.save_model(model_file)

        tar_path = out / "model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_file, arcname="xgboost-model")
        self.model_data = str(tar_path)

        # (optional) save evals_result for convenience
        (out.parent / "metrics").mkdir(parents=True, exist_ok=True)
        with open(out.parent / "metrics" / "training_metrics.json", "w") as f:
            json.dump(evals_result, f, indent=2)

        return self.model_data

    def explain(self, data_path: str, top_k: int = 20, out_subdir: str = "explain"):
        from sklearn.datasets import load_svmlight_file
        X_sparse, y = load_svmlight_file(str(_p(data_path)).split('?')[0])  # ignore "?format=libsvm"
        X = X_sparse.toarray()

        feat_names = self._booster.feature_names or [f"f{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feat_names)

        explainer = shap.TreeExplainer(self._booster)
        sv = explainer.shap_values(X_df)

        exp_dir = self._job().parent / out_subdir
        exp_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(sv, columns=feat_names).to_csv(exp_dir / "shap_values.csv", index=False)

        mean_abs = np.abs(sv).mean(axis=0)
        summ = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
        summ.to_csv(exp_dir / "shap_summary.csv", index=False)

        plt.figure(); shap.summary_plot(sv, X_df, show=False); plt.tight_layout()
        plt.savefig(exp_dir / "shap_summary_beeswarm.png", dpi=160); plt.close()
        plt.figure(); shap.summary_plot(sv, X_df, plot_type="bar", show=False); plt.tight_layout()
        plt.savefig(exp_dir / "shap_summary_bar.png", dpi=160); plt.close()

        report = {
            "job_dir": str(self._job()),
            "explain_dir": str(exp_dir),
            "data_path": str(_p(data_path)),
            "n_rows": int(X_df.shape[0]),
            "n_features": int(X_df.shape[1]),
            "top_features": summ.head(top_k).to_dict("records"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_artifact": self.model_data,
            "params": self.hyperparameters,
            "image_uri": self.image_uri,
            "instance_type": self.instance_type,
            "instance_count": self.instance_count,
        }
        with open(exp_dir / "explain_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return str(exp_dir)

    # --- SageMaker-like APIs ---

    def create_model(self, role=None, image_uri=None, instance_type=None):
        """Return a Model bound to this estimator's artifact (mimics Estimator.create_model)."""
        return Model(
            model_data=self.model_data,
            image_uri=image_uri or self.image_uri,
            role=role or self.role,
            base_job_name=self.base_job_name,
            instance_type=instance_type or self.instance_type,
        )

    def transformer(self, instance_count=1, instance_type=None, accept="text/csv",
                    assemble_with="Line", output_path=None):
        """Return a Transformer for batch predictions (mimics Estimator.transformer)."""
        if not self.model_data: _ = self._job()  # ensure a job dir exists
        model = self.create_model()
        return model.transformer(
            instance_count=instance_count,
            instance_type=instance_type or self.instance_type,
            accept=accept,
            assemble_with=assemble_with,
            output_path=output_path or str((self._job().parent / "batch-preds").resolve()),
        )

# --- Minimal Model / Transformer to mirror SageMaker flow ---
class Model:
    def __init__(self, model_data, image_uri, role=None, base_job_name="xgb-job", instance_type="ml.m5.large"):
        self.model_data = str(_p(model_data)) if isinstance(model_data, str) else str(model_data)
        self.image_uri = image_uri
        self.role = role
        self.base_job_name = base_job_name
        self.instance_type = instance_type
        self._booster = None

    @classmethod
    def from_model_data(cls, model_data, image_uri, role=None, **kwargs):
        return cls(model_data=model_data, image_uri=image_uri, role=role, **kwargs)

    def _load_booster(self):
        mdl_path = _p(self.model_data)
        if mdl_path.is_dir() and (mdl_path / "xgboost-model").exists():
            model_file = mdl_path / "xgboost-model"
        else:
            tmp_dir = mdl_path.parent / ("_mdl_" + uuid.uuid4().hex[:6])
            tmp_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(mdl_path, "r:gz") as tar:
                tar.extractall(tmp_dir)
            model_file = tmp_dir / "xgboost-model"
        bst = xgb.Booster(); bst.load_model(model_file)
        self._booster = bst
        return bst

    def transformer(self, instance_count=1, instance_type=None, accept="text/csv",
                    assemble_with="Line", output_path="./batch-preds"):
        return Transformer(
            model=self,
            instance_count=instance_count,
            instance_type=instance_type or self.instance_type,
            accept=accept,
            assemble_with=assemble_with,
            output_path=output_path,
        )

class Transformer:
    def __init__(self, model: Model, instance_count=1, instance_type="ml.m5.large",
                 accept="text/csv", assemble_with="Line", output_path="./batch-preds"):
        self.model = model
        self.instance_count = instance_count
        self.instance_type = instance_type
        self.accept = accept
        self.assemble_with = assemble_with
        self._output_dir = Path(str(_p(output_path)).split('?')[0]); self._output_dir.mkdir(parents=True, exist_ok=True)
        self._output_uri = str(self._output_dir)

    @property
    def output_path(self):
        return self._output_uri
    
    def transform(self, data, content_type="text/libsvm", split_type="Line"):
        inp = _p(data)
        # If it's a directory → expand; otherwise treat it as a single file (even if relative)
        files = sorted([p for p in inp.iterdir() if p.is_file()]) if inp.is_dir() else [inp]

        bst = self.model._booster or self.model._load_booster()
        for f in files:
            
            d = xgb.DMatrix(f)

            if getattr(bst, "best_iteration", None) is not None:
                preds = bst.predict(d, iteration_range=(0, bst.best_iteration + 1))
            else:
                preds = bst.predict(d)

            out_file = self._output_dir / (str(f.name).split('?')[0] + ".out")
            np.savetxt(out_file, preds, fmt="%.10f", delimiter="," if self.accept == "text/csv" else " ")

    def wait(self):
        return

def _sample_from_spec(name, spec, rng):
    # Real SageMaker tuner objects (duck-typed)
    # - ContinuousParameter: has min_value/max_value
    # - IntegerParameter:    has min_value/max_value, name contains "Integer"
    # - CategoricalParameter: has values
    if hasattr(spec, "values"):
        return rng.choice(list(spec.values))
    if hasattr(spec, "min_value") and hasattr(spec, "max_value"):
        is_int = ("Integer" in spec.__class__.__name__) or isinstance(getattr(spec, "min_value"), int)
        lo, hi = float(spec.min_value), float(spec.max_value)
        return int(rng.integers(int(lo), int(hi) + 1)) if is_int else float(rng.uniform(lo, hi))

    # Simple Python specs:
    # - tuple/list of (low, high) → continuous
    # - list of choices → categorical
    # - scalar → fixed value
    if isinstance(spec, (tuple, list)):
        if len(spec) == 2 and all(isinstance(x, (int, float)) for x in spec):
            lo, hi = spec
            return float(rng.uniform(float(lo), float(hi)))
        return rng.choice(list(spec))
    return spec  # fixed

class HyperparameterTuner:
    """
    Minimal tuner that:
      - samples real SageMaker Parameter objects or simple specs
      - runs trials sequentially with your Estimator
      - scores on 'validation' using objective_metric_name (e.g. 'validation:auc')
    """
    def __init__(self, estimator, objective_metric_name, objective_type,
                 hyperparameter_ranges, max_jobs=10, max_parallel_jobs=1, seed=42):
        self.estimator = estimator
        self.objective_metric_name = objective_metric_name  # e.g. 'validation:auc'
        self.objective_type = objective_type                # 'Maximize' or 'Minimize'
        self.hyperparameter_ranges = hyperparameter_ranges  # dict param -> spec
        self.max_jobs = max_jobs
        self.max_parallel_jobs = max_parallel_jobs  # ignored (sequential)
        self.seed = seed
        self.best_estimator_ = None
        self.best_hyperparameters_ = None
        self.best_metric_ = None
        self.trials_ = []

    def _metric_fn(self):
        m = self.objective_metric_name.split(":", 1)[-1].lower()
        if m == "auc":     return lambda y, p: roc_auc_score(y, p)
        if m == "rmse":    return lambda y, p: sqrt(mean_squared_error(y, p))
        if m == "logloss": return lambda y, p: log_loss(y, p)
        if m == "aucpr":   return lambda y, p: average_precision_score(y, p)
        return lambda y, p: roc_auc_score(y, p)

    def _is_better(self, a, b):
        if b is None: return True
        return (a > b) if self.objective_type.lower().startswith("max") else (a < b)

    def fit(self, inputs):
        import xgboost as xgb
        # load validation once (LibSVM path)
        dval = xgb.DMatrix(str(_p(inputs["validation"])))
        y_val = dval.get_label()

        rng = np.random.default_rng(self.seed)
        metric_fn = self._metric_fn()

        for i in range(self.max_jobs):
            sampled = {k: _sample_from_spec(k, v, rng) for k, v in self.hyperparameter_ranges.items()}

            from copy import deepcopy
            est = deepcopy(self.estimator)
            est.base_job_name = f"{self.estimator.base_job_name}-t{i+1}"
            est.set_hyperparameters(**sampled)

            est.fit(inputs)

            preds = est._booster.predict(dval)
            score = metric_fn(y_val, preds)

            self.trials_.append({"trial": i+1, "params": sampled, "metric": score})

            if self._is_better(score, self.best_metric_):
                self.best_metric_ = score
                self.best_hyperparameters_ = sampled
                self.best_estimator_ = est

        return self.best_estimator_
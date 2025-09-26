FROM public.ecr.aws/glue/aws-glue-libs:5

USER root
RUN python3 -m pip install notebook jupyterlab pandas matplotlib scikit-learn boto3 sagemaker

USER hadoop
CMD ["python3", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=''", "--ServerApp.root_dir=/home/hadoop/workspace"]

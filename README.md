# AWS End-to-End AI/ ML Pipeline on Healthcare Dataset

## Overview
This project demonstrates how to build an **end-to-end machine learning pipeline on AWS** to classify healthcare data and predict the likelihood of stroke. It covers the entire workflow, from raw data ingestion to model deployment, using scalable cloud-native tools. The dataset reflects common healthcare attributes such as demographics, glucose levels, hypertension, and BMI, and the task is framed as a binary classification problem: predicting whether a patient is at risk of stroke.  

The project is organized into **three Jupyter notebooks**, each covering a major stage of the pipeline:

1. [**Data Ingestion and Processing**](https://github.com/GiggleSamurai/aws-end-to-end-ml-pipeline/blob/main/data_engineering.ipynb)  
   Ingest data into S3, clean and transform it using PySpark and distributed computing.  

2. [**Exploration and Feature Engineering**](https://github.com/GiggleSamurai/aws-end-to-end-ml-pipeline/blob/main/exploratory_data_analysis.ipynb)  
   Explore the dataset, visualize key features, and apply transformations such as encoding, scaling, and feature binning.  

3. [**Model Training and Evaluation**](https://github.com/GiggleSamurai/aws-end-to-end-ml-pipeline/blob/main/modeling.ipynb)  
   Preprocess with Scikit-learn pipelines, train models in SageMaker, perform hyperparameter tuning, and evaluate results using AUC-PR, F1-score, and SHAP values.   

## Architecture and Workflow
1. **Data Ingestion**  
   - Load raw healthcare data into Amazon S3.  
   - Use PySpark and distributed computing for scalable ETL processing.  

2. **Data Processing**  
   - Clean and transform the dataset.  
   - Handle missing values and outliers.  
   - Apply categorical encoding and standardization to normalize features.  

3. **Exploration and Feature Engineering**  
   - Visualize feature distributions and correlations.  
   - Apply binning techniques (e.g., age groups).  
   - Engineer additional features relevant to stroke prediction.  

4. **Model Training and Tuning**  
   - Split data into training, validation, and test sets (60/20/20).  
   - Use Scikit-learn pipelines for preprocessing.  
   - Train classification models in AWS SageMaker.  
   - Perform hyperparameter tuning with Bayesian optimization.  

5. **Evaluation and Interpretation**  
   - Evaluate using metrics suitable for imbalanced data (AUC-PR, F1-score).  
   - Optimize thresholds to balance precision and recall.  
   - Apply SHAP values to interpret feature importance.  

6. **Deployment and Monitoring**  
   - Deploy the trained model using SageMaker endpoints.  
   - Set up monitoring for data drift and model performance.  
   - Enable continuous learning pipelines for retraining with new data.  

## Technology Stack
- **AWS Services:** S3, SageMaker, Glue, Studio  
- **Data Processing:** PySpark, Delta Lake  
- **Modeling:** Scikit-learn, XGBoost, SageMaker built-in algorithms  
- **Visualization:** Matplotlib, Seaborn  
- **Explainability:** SHAP 

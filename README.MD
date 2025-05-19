# Customer Churn Prediction System

This project was developed as the final assignment for the Machine Learning course under Dr. Nermeen at the Faculty of Graduate Studies for Statistical Research (FGSSR) – Cairo University.

## Objective

To build a predictive system that helps businesses identify customers likely to churn (leave the service) and provide intelligent retention strategies using machine learning.

## Team Members

- Nourhan Khaled  
- Fatma Ehab  
- Hend Labib  
- Mona Elgoba  
- Mayar Mustafa

## Project Pipeline

- *Data Preprocessing*: Handling missing values, feature encoding, and scaling (RobustScaler, MinMaxScaler)
- *Feature Selection*: Based on correlation and model relevance
- *Imbalance Handling*: SMOTE oversampling used for training
- *Models Trained*:
  - Logistic Regression
  - XGBoost
  - LightGBM
  - Stacking Classifier
- *Evaluation*: Compared models using Precision, Recall, F1-Score
- *Deployment*:
  - Deployed via *Gradio*
  - Integrated with *LLM-powered module* for personalized retention offers

## Deployment

You can try the system live via Gradio (link provided in notebook or deployment instructions).

# CCFraudDetection

09/2024 <br>
Personal Project

# Project Overview

This project focuses on detecting credit card fraud using a machine learning model. The dataset was obtained from [Kaggle](https://www.kaggle.com/code/nathanxiang/credit-card-fraud-analysis-and-modeling), and the objective is to identify fraudulent transactions from a dataset of credit card transactions. This project allowed me to further use and experiment with different machine learning models.

## Dataset

The dataset consists of transactions labeled as either fraudulent or non-fraudulent. Each transaction includes a number of features representing various attributes of the transaction.

- **Fraudulent transactions**: Labeled as `1`
- **Non-fraudulent transactions**: Labeled as `0`

The dataset is highly imbalanced, with the majority of transactions being non-fraudulent. As such, handling this imbalance was a critical aspect of model training.

## Project Goals

- **Explore different machine learning models** to determine which ones perform best for imbalanced datasets.
- **Analyze model performance** using metrics such as precision, recall, and F1-score due to the imbalanced nature of the dataset.


## Steps Involved

1. **Data Preprocessing**: Clean and prepare the dataset for model training, addressing the issue of class imbalance through undersampling/oversampling techniques.
2. **Model Training**: Train multiple models including Logistic Regression, Support Vector Machine (SVM), and Random Forest Classifier (RFC), with a focus on evaluating their performance on the imbalanced dataset.
3. **Model Evaluation**: Evaluate the models using confusion matrix, precision, recall, and F1-score to determine which one performs best for fraud detection.



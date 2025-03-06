# CCFraudDetection

09/2024 <br>
Personal Project <br>
*Last Updated: 03/2025*  


## Project Overview

This project is an exploratory project focused on detecting credit card fraud by uncovering patterns in the data through extensive EDA and comparing different machine learning models. The main objective is to explore the dataset, identify trends and gain insights via EDA, and determine which model best distinguishes fraudulent transactions from legitimate ones.

## Dataset

- **Fraudulent transactions:** Labeled as `1`
- **Non-fraudulent transactions:** Labeled as `0`

The dataset was obtained from [Kaggle](https://www.kaggle.com/code/nathanxiang/credit-card-fraud-analysis-and-modeling).

## Project Goals

- **Exploratory Data Analysis (EDA):**  
  Dive into the dataset to uncover hidden patterns, trends, and anomalies that may indicate fraud. This includes analyzing transaction amounts, time patterns, demographics, and spatial factors.
  
- **Model Comparison:**  
  Develop and compare multiple machine learning models to determine which approach works best for fraud detection on an imbalanced dataset.
  
- **Performance Evaluation:**  
  Evaluate models using metrics like precision, recall, F1-score, and ROC AUC to assess their effectiveness in distinguishing between fraudulent and non fraudulent transactions.

## Methodology

### Data Preprocessing & EDA

- **Data Cleaning & Feature Engineering:**  
  Prepare the dataset by handling missing values, outliers, and generating new features.
  
- **Exploratory Data Analysis (EDA):**  
  Use visualization techniques to analyse patterns in transaction amounts, time of transaction, user demographics, and spatial relationships. This helps in understanding the underlying structure and potential indicators of fraud.

### Modeling Pipelines

To streamline the modeling process, three pipelines are constructed with the following common steps:
  
1. **StandardScaler:** Standardises features to ensure consistency.
2. **PCA (n_components=5):** Reduces dimensionality to speed up training and reduce noise.
3. **SMOTE:** Synthesizes samples for the minority (fraud) class to address imbalance.
4. **Classifier:** Model-specific classifiers are used:
   - **Random Forest**
   - **Gradient Boosting**
   - **XGBoost**
5. **Caching with joblib.Memory:** Caches intermediate computations to improve runtime.

### Model Tuning

Initial experiments using grid search CV for hyperparameter tuning proved computationally expensive. Instead, fixed parameters were used to balance performance with practical runtime, allowing rapid comparisons between models.

- **Model Performance:**  
  The Random Forest, Gradient Boosting, and XGBoost pipelines were compared. Each model exhibited unique strengths, with Random Forest and XGBoost showing particularly strong performance in distinguishing fraudulent from non fraudulent transactions.

## Conclusion

This project combines EDA and machine learning to tackle credit card fraud detection. By exploring the dataset and comparing multiple models, the project lays a solid foundation for understanding the fraud patterns and building effective detection systems.

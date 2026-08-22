# Credit Card Fraud Detection

09/2024 <br>
Personal Project <br>
*Last Updated: 08/2026*

## Project Overview

This project investigates credit card fraud using exploratory data analysis, feature engineering and machine learning. Fraud represents fewer than 1% of the transactions, so the project focuses on detecting the fraud class rather than producing a misleadingly high accuracy.

The work is divided into three notebooks covering EDA, feature engineering and model evaluation. Alongside transaction level features, I extend the original analysis by comparing each transaction with the cardholder's earlier activity. All history features use only information which would have been available when the transaction occurred.

## Dataset

The project uses the [Credit Card Transactions Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection). It contains synthetic transactions produced using the Sparkov simulator.

- `fraudTrain.csv`: 1,296,675 transactions from January 2019 to June 2020
- `fraudTest.csv`: 555,719 later transactions from June 2020 to December 2020
- Target variable: `is_fraud`, where `1` represents fraud and `0` represents a non-fraudulent transaction

The CSV files are not stored in the repository because they are approximately 480 MB in total. Instructions for adding them are included in [data/README.md](data/README.md).

## Notebook Structure

### 1. [Exploratory Data Analysis](01_EDA.ipynb)

The first notebook checks the data and investigates:

- the class imbalance and the limitations of accuracy
- transaction-amount distributions and fraud rates within fixed amount bands
- cardholder age and recorded gender
- hourly, weekly and monthly fraud patterns
- transaction volume and fraud rate by merchant category
- customer-to-merchant distance, state and city population
- the interaction between merchant category and transaction amount
- transaction amount and timing compared with the cardholder's earlier activity

The EDA compares fraud rates with the number of transactions behind them. This avoids treating a high-volume group as high risk simply because it contains more transactions. The category-and-amount heatmap also shows why nonlinear models are useful: the meaning of transaction amount changes considerably across merchant categories.

### 2. [Feature Engineering](02_Feature_Engineering.ipynb)

The second notebook creates 28 model features from the EDA. The main features are:

- log-transformed transaction amount and city population
- customer-to-merchant distance calculated using the Haversine formula
- raw and cyclical hour, weekday and month features
- merchant category converted into dummy variables
- time since the cardholder's previous transaction
- current amount compared with the cardholder's earlier mean

The history variables are calculated after sorting the transactions chronologically. Transactions recorded for the same cardholder at exactly the same time receive the same earlier history, so they cannot be used as past information for one another.

Age, gender, date of birth, names, addresses, job title, merchant name, card number, transaction ID and state are not used as model inputs. Card number is used only to construct the earlier-history features and is then removed.

The same transformations are stored in [feature_engineering.py](feature_engineering.py), which the model notebook imports so both notebooks use the same feature definitions.

### 3. [Model Comparison](03_Model_Comparison.ipynb)

The final notebook compares five models:

- Dummy Classifier
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

The data is split chronologically rather than randomly:

| Purpose | Period | Transactions | Fraud cases |
|---|---|---:|---:|
| Model training | January–December 2019 | 924,850 | 5,220 |
| Model validation | January–March 2020 | 172,843 | 1,123 |
| Threshold selection | 1 April–21 June 2020 | 198,982 | 1,163 |
| Final test | 21 June–31 December 2020 | 555,719 | 2,145 |

The final test begins immediately after the threshold period, so the two sets of transactions do not overlap.

Average precision is used to select the model because it measures fraud ranking across different thresholds. After selecting the model, its classification threshold is chosen on a separate period by maximising F2 score. F2 places more weight on recall, which reflects the aim of detecting fraud without allowing the model to flag almost everything.

SMOTE is not used in the final version. There are already thousands of fraud examples, and creating artificial rows between dummy-encoded categories is harder to justify than learning from the original observations and choosing a fraud-focused threshold.

## Validation Results

| Model | Average Precision | ROC AUC |
|---|---:|---:|
| XGBoost | **0.9010** | **0.9978** |
| Random Forest | 0.8724 | 0.9915 |
| Gradient Boosting | 0.8086 | 0.9804 |
| Logistic Regression | 0.4521 | 0.9164 |
| Dummy Classifier | 0.0065 | 0.5000 |

XGBoost gives the strongest validation average precision among the five configurations tested and is therefore refitted using the training and validation periods.

## Final XGBoost Results

The probability threshold of `0.1089` is selected before the supplied final test labels are evaluated.

| Measure | Final Test Result |
|---|---:|
| Average Precision | 0.8542 |
| ROC AUC | 0.9971 |
| Precision | 58.67% |
| Recall | **86.15%** |
| F1 Score | 0.6980 |
| F2 Score | **0.7877** |
| Fraud cases caught | **1,848 of 2,145** |
| Fraud cases missed | 297 |
| False alerts | 1,302 |
| False alerts per 10,000 transactions | 23.43 |
| Fraudulent transaction value caught | **94.34%** |

The class-normalised confusion matrix is used instead of allowing the very large number of correct non-fraud predictions to dominate the visual comparison.

## Checking the History Features

To check whether the cardholder-history extension genuinely helps, I fit the same XGBoost model with and without the two history variables.

| Feature Set | Validation Average Precision |
|---|---:|
| Without cardholder history | 0.8524 |
| With cardholder history | **0.9010** |

Removing the two history features reduces validation average precision from 0.9010 to 0.8524. This suggests that comparing each transaction with the cardholder's earlier activity adds useful information.

## Main Findings

- Transaction amount, hour and merchant category contain the clearest fraud patterns.
- Merchant category and amount interact strongly, supporting the use of nonlinear models.
- Comparing a transaction with the cardholder's own earlier activity improves later-period performance.
- XGBoost produces the strongest validation fraud ranking among the five configurations tested.
- Final-test average precision is 0.8542, lower than the validation result of 0.9010.
- Recall varies across merchant categories and recorded gender, even though gender is not used by the model.
- Accuracy is not used to select the model because correct non-fraud predictions dominate this dataset.

## Limitations

- The dataset is synthetic, so the patterns and model scores should not be treated as expected performance at a real bank.
- One manually selected configuration of each model is compared on one chronological validation period, so a wider search or different period could change the ranking.
- The history features assume transactions are processed in time order. The same cardholders also appear repeatedly, so this is not a full test on new customers.
- All 163 transactions from the 16 unseen cards are labelled as fraud, which appears to be a synthetic data artefact. This group cannot be treated as a realistic new customer test.
- The recorded gender result is a diagnostic within synthetic data and should not be treated as evidence of real world group differences.
- Performance is lower on the later test period, so a real system would need continued monitoring.
- Feature importance describes how the fitted model uses the variables but does not establish causation.
- F2 treats every fraud case equally. A real system would choose its threshold using investigation capacity and the financial costs of missed fraud and false alerts.

## Libraries

- pandas and NumPy
- Matplotlib and seaborn
- scikit-learn
- XGBoost
- Jupyter

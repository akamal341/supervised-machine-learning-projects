# Spam Email Detection using Various Classification Models

## Project Overview

This project explores the development and evaluation of classification models for detecting spam emails. By employing multiple supervised learning techniques, the project aims to understand the precision/recall tradeoff critical in spam detection. The models used include dummy classifiers, Support Vector Classification (SVC), and Logistic Regression.

## Table of Contents

1. [Project Description](#project-description)
2. [Data Import and Preparation](#data-import-and-preparation)
3. [Dummy Classifiers](#dummy-classifiers)
4. [SVC Classifier](#svc-classifier)
5. [Decision Function with SVC](#decision-function-with-svc)
6. [Logistic Regression Classifier](#logistic-regression-classifier)
7. [Precision-Recall Curves](#precision-recall-curves)
8. [Confusion Matrix](#confusion-matrix)
9. [Grid Search on Logistic Regression](#grid-search-on-logistic-regression)
10. [Normalizing Features](#normalizing-features)
11. [Setup Instructions](#setup-instructions)
12. [Key Results](#key-results)
13. [Contact Information](#contact-information)

---

## Project Description

The primary focus of this project is to build and evaluate classification models for detecting spam emails. The problem is addressed as a binary classification task, where emails are categorized as "spam" (Class 1) or "ham" (Class 0). This involves understanding the precision/recall tradeoff, which is crucial for a highly precise spam filter.

## Data Import and Preparation

- **Data Source:** The dataset is imported from `assets/spam.csv`.
- **Data Splitting:** The dataset is split into training and testing sets.
- **Feature Normalization:** Features are normalized using `StandardScaler` from `sklearn.preprocessing` to standardize the scale, avoiding data leakage.

## Dummy Classifiers

Two dummy classifiers are used as baseline models:
1. A classifier that respects the training set's label distribution.
2. A classifier that classifies everything as the majority class.

We compare their precision, recall, and accuracy.

## SVC Classifier

An SVC classifier with default hyperparameters is trained and its performance evaluated based on accuracy, recall, and precision.

## Decision Function with SVC

An SVC classifier with specified hyperparameters (`C`: 1e9, `gamma`: 1e-8) is used to evaluate the confusion matrix on the testing set using a specific threshold for the decision function.

![image](https://github.com/user-attachments/assets/21552501-3acb-4e8d-9d22-e1804b489c9e)

## Logistic Regression Classifier

A Logistic Regression spam classifier is trained and its performance is evaluated. Precision-Recall and ROC curves are plotted to understand the precision/recall tradeoff.

![image](https://github.com/user-attachments/assets/0d7b38d2-c884-4c50-9817-78d5ebd6bb04)

## Precision-Recall Curves

The project investigates the recall when precision is 0.90 and the true positive rate when the false positive rate is 0.10.

## Confusion Matrix

A confusion matrix is generated for the Logistic Regression model with specified hyperparameters. This helps in understanding the relationship between correct and incorrect predictions.

## Grid Search on Logistic Regression

A `GridSearchCV` is performed over specified hyperparameters for a Logistic Regression classifier, aiming to optimize for precision.

## Normalizing Features

The effect of feature normalization on model precision is compared by running `GridSearchCV` with and without feature normalization.

## Setup Instructions

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/akamal341/spam-email-detection.git
    cd spam-email-detection
    ```

2. **Install Dependencies**:
    ```sh
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3. **Run the Jupyter Notebook**:
    Open and execute the provided Jupyter Notebook to see the data manipulation and analysis.

## Key Results

1. **Dummy Classifiers:** Provided a baseline to compare real classifier performance.
2. **SVC Classifier:** Achieved high accuracy, recall, and precision.
3. **Decision Function:** Showcased how decision thresholds impact classification performance.
4. **Logistic Regression:** Plotted Precision-Recall and ROC curves to understand model performance.
5. **Grid Search:** Optimized hyperparameters for Logistic Regression, demonstrating the importance of parameter tuning.
6. **Normalization Impact:** Highlighted how feature normalization affects model precision.

## Contact Information

For any questions or further information, please contact:
- **Name:** Asad Kamal
- **Email:** aakamal {/@/} umich {/dot/} edu
- **LinkedIn:** [LinkedIn Profile](https://linkedin.com/in/asadakamal)

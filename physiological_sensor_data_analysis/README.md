# Physiological Sensor Data Analysis with Tree-Based Classification

## Project Overview

This project explores the use of tree-based classification methods to analyze physiological sensor data collected from smartphone-based sensors. The primary goal is to predict the activity of test subjects based on physiological measurements using supervised learning techniques. The activities include neutral, emotional, mental, and physical states.

## Table of Contents

1. [Project Description](#project-description)
2. [Data Import and Preparation](#data-import-and-preparation)
3. [Standard Train-Test Split](#standard-train-test-split)
4. [Baseline Tree Model](#baseline-tree-model)
5. [Custom Train-Test Split](#custom-train-test-split)
6. [Multiclass Confusion Matrix](#multiclass-confusion-matrix)
7. [Feature Importance](#feature-importance)
8. [Final Project: Optimized Model](#final-project-optimized-model)
9. [Setup Instructions](#setup-instructions)
10. [Key Results](#key-results)
11. [Contact Information](#contact-information)

---

## Project Description

This project aims to build and evaluate several classification models to predict the activity of test subjects based on physiological sensor data. The analysis leverages tree-based methods such as Decision Trees, Random Forests, and Gradient Boosting. The dataset contains 4480 rows with 533 measurement features collected from 40 test volunteers during various activities.

## Data Import and Preparation

- **Data Source:** The dataset is imported from `assets/sensor_data.csv`.
- **Feature Normalization:** Features are standardized using `StandardScaler` from `sklearn.preprocessing`.
- **Feature Selection:** Features containing the substring "_mad_" are selected for model training and evaluation.

## Standard Train-Test Split

A standard train-test split is performed using `train_test_split` to provide a baseline for model evaluation.

## Baseline Tree Model

A baseline Decision Tree model is trained using default hyperparameters. The performance is evaluated to provide a benchmark for further model improvements.

## Custom Train-Test Split

Due to the nature of the data collection, a custom train-test split function is implemented to avoid data leakage and ensure a robust evaluation setup.

## Multiclass Confusion Matrix

A Logistic Regression model is trained, and a multiclass confusion matrix is generated to understand the relationship between correct and incorrect predictions.

![image](https://github.com/user-attachments/assets/bc5fd162-8e2b-409e-b573-2a9924cc5e92)

## Feature Importance

The importance of each feature is evaluated using a RandomForestClassifier. The top features are identified and used to improve model performance.

## Final Project: Optimized Model

The final task involves creating an optimized model using a Gradient Boosting Classifier. The model is assessed based on the ROC-AUC score, with a constraint of using no more than 10 features to ensure computational efficiency.

## Setup Instructions

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/akamal341/physiological-sensor-data-analysis.git
    cd physiological-sensor-data-analysis
    ```

2. **Install Dependencies**:
    ```sh
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3. **Run the Jupyter Notebook**:
    Open and execute the provided Jupyter Notebook to walk through the data analysis and model development process.

## Key Results

1. **Baseline Model:** Achieved initial accuracy of ~81.6%.
2. **Custom Train-Test Split:** Improved data separation to prevent leakage and ensure robustness.
3. **Feature Importance:** Identified key features contributing most to model performance.
4. **Optimized Model:** Developed a Gradient Boosting Classifier achieving a ROC-AUC score of 0.863 with only 10 selected features.

## Contact Information

For any questions or further information, please contact:
- **Name:** Asad Kamal
- **Email:** aakamal {/@/} umich {/dot/} edu
- **LinkedIn:** [LinkedIn Profile](https://linkedin.com/in/asadakamal)

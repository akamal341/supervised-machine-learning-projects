# Regression and Classification Analysis

## Project Overview

This project aims to build and evaluate several regression and classification models, exploring how model complexity impacts generalization performance. The analysis includes polynomial regression, k-Nearest Neighbors (kNN) regression, Lasso regression, and Support Vector Classification (SVC) applied to both synthetic data and the Breast Cancer Wisconsin dataset.

## Table of Contents
1. [Project Description](#project-description)
2. [Functionality](#functionality)
3. [Setup Instructions](#setup-instructions)
4. [Key Results](#key-results)
5. [Contact Information](#contact-information)

---

## Project Description

In this project, we follow these steps:

1. **Generate synthetic data:** We start by generating synthetic polynomial data with noise to simulate a real-world dataset with underlying patterns and inherent randomness.
  
2. **Evaluate polynomial regression models:** We fit polynomial regression models of various degrees (1, 3, 7, 11) to understand how well they capture the underlying patterns in the data.

3. **Compare kNN regression performance:** We fit a kNN regression model and compare its performance to polynomial regression models.

4. **Apply Lasso regression:** We use Lasso regression to constrain model complexity and avoid overfitting, evaluating its performance against polynomial regression of various degrees.

5. **Analyze the Breast Cancer Wisconsin dataset using Support Vector Classification (SVC):** We employ SVC to classify instances in the dataset, exploring the impact of the gamma parameter on the classifier's performance.

## Functionality

### Raw Data with Noise Generation
We generate a synthetic dataset for polynomial regression analysis. The independent variable x consists of 60 evenly spaced points from the interval [0, 20], and the dependent variable y is a polynomial function with noise.

---

### Polynomial Features and Quality of Fit

#### Task 1a - Polynomial Features
Fit polynomial expansions of the training data for degrees 1, 3, 7, 11 to a Linear Regression model.

#### Task 1b - Quality of Fit
Calculate and compare the R² scores for polynomial models of various degrees on both training and testing datasets.

---

### KNN Regression
Fit a kNN regression model with the training data and return the R² value on the testing data.

---

### Polynomial Fitting with Lasso Regression

#### Task 3a - Lasso Regression
Fit polynomial models of various degrees (1, 3, 7, 11) with Lasso regression to avoid overfitting.

#### Task 3b - Compare with Gold Standard
Compare the R² score of the Lasso models to a 'gold standard' test set generated from the true underlying polynomial model.

---

### Support Vector Classification on Breast Cancer Data

#### Task 4 - Applying SVC
Generate validation curves for SVC with varying gamma values to analyze their impact on model accuracy.

---

## Setup Instructions

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/akamal341/regression-classification-analysis.git
    cd regression-classification-analysis
    ```

2. **Install Dependencies**:
    ```sh
    pip install pandas numpy scikit-learn matplotlib
    ```

3. **Run the Jupyter Notebook**:
    Open and execute the provided Jupyter Notebook to see the data manipulation and analysis.

## Key Results

1. **Polynomial Regression:** Evaluated model performance for polynomial degrees 1, 3, 7, and 11, revealing insights into underfitting, overfitting, and optimal generalization.
2. **KNN Regression:** Compared the performance of kNN regression with polynomial regression models.
3. **Lasso Regression:** Demonstrated the use of regularization in improving model generalization.
4. **SVC Analysis:** Explored the impact of the gamma parameter on the performance of a Support Vector Classifier on the Breast Cancer Wisconsin dataset.

## Contact Information

For any questions or further information, please contact:
- **Name:** Asad Kamal
- **Email:** aakamal {/@/} umich {/dot/} edu
- **LinkedIn:** [LinkedIn Profile](https://linkedin.com/in/asadakamal)

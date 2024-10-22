# Breast Cancer Diagnosis Using k-Nearest Neighbors

## Project Overview

This project aims to develop a k-Nearest Neighbors (kNN) classifier to diagnose breast cancer using the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn. The analysis encompasses data transformation, model development, performance evaluation, and hyperparameter tuning, providing insights into diagnostic accuracy and feature importance.

## Table of Contents

1. [Project Description](#project-description)
2. [Data Sources](#data-sources)
3. [Functionality](#functionality)
   - [Load and Explore Dataset](#load-and-explore-dataset)
   - [Data Transformation](#data-transformation)
   - [Descriptive Statistics](#descriptive-statistics)
   - [Data Preparation for Model](#data-preparation-for-model)
   - [Model Training and Evaluation](#model-training-and-evaluation)
   - [Hyper-Parameter Tuning](#hyper-parameter-tuning)
   - [One-Hot Encoding Example](#one-hot-encoding-example)
4. [Setup Instructions](#setup-instructions)
5. [Key Results](#key-results)
6. [Contact Information](#contact-information)

---

## Project Description

This project involves the following steps:

1. **Data Loading and Exploration:** Load and explore the Breast Cancer Wisconsin dataset.
2. **Data Transformation:** Transform raw data into a more readable and usable format.
3. **Descriptive Statistics:** Calculate and display class distribution statistics.
4. **Data Preparation for Model:** Split data into features (X) and target (y).
5. **Model Training and Evaluation:** Train a kNN classifier, evaluate its performance, and predict outcomes for new data.
6. **Hyper-Parameter Tuning:** Optimize the kNN model by tuning hyper-parameters.
7. **One-Hot Encoding Example:** Demonstrate handling categorical data using one-hot encoding on a secondary dataset.

## Data Sources

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, readily accessible from scikit-learn. It includes attributes for several physical characteristics of breast cancer cell nuclei.

## Functionality

### 1. Load and Explore Dataset

The function `load_breast_cancer` from scikit-learn is used to load the dataset, followed by exploratory data analysis to understand the dataset's structure and attributes.

```python
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer Wisconsin dataset
cancer = load_breast_cancer()

# Display dataset keys
print(cancer.keys())

# Display dataset description
print(cancer.DESCR)
```

### 2. Data Transformation

Transform the dataset into a pandas DataFrame for easier manipulation.

```python
import pandas as pd

# Function to transform the dataset into a pandas DataFrame
def make_cancer_dataframe():
    cancer_df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
    cancer_df['target'] = cancer['target']
    return cancer_df

# Create DataFrame
cancer_df = make_cancer_dataframe()
print(f"DataFrame shape: {cancer_df.shape}")
cancer_df.head()
```

### 3. Descriptive Statistics

Calculate and display the class distribution of the dataset.

```python
# Function to calculate class distribution
def get_target_distro(df):
    malignant = (df['target'] == 0).sum()
    benign = (df['target'] == 1).sum()
    return pd.Series([malignant, benign], index=['malignant', 'benign'])

# Display class distribution
target_distro = get_target_distro(cancer_df)
print(target_distro)
```

### 4. Data Preparation for Model

Prepare the features (X) and target (y) for model training.

```python
# Function to prepare X (features) and y (target)
def prepare_X_y(df):
    X = df.iloc[:, :-1]
    y = df['target']
    return X, y

# Prepare the data
X, y = prepare_X_y(cancer_df)
print(f"Features shape: {X.shape}, Target shape: {y.shape}")
```

### 5. Model Training and Evaluation

#### Train-Test Split

Split the dataset into training and testing sets for model evaluation.

```python
from sklearn.model_selection import train_test_split

# Function to split the data into training and testing sets
def get_train_test(X, y, random_state=42):
    return train_test_split(X, y, random_state=random_state)

# Perform train-test split
X_train, X_test, y_train, y_test = get_train_test(X, y)
print(f"Train shapes: {X_train.shape, y_train.shape}, Test shapes: {X_test.shape, y_test.shape}")
```

#### Train kNN Classifier

Train a kNN classifier with k=1.

```python
from sklearn.neighbors import KNeighborsClassifier

# Function to train a kNN classifier
def k_nearest_neighbors(X_train, y_train, n_neighbors=1):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# Train kNN classifier
knn = k_nearest_neighbors(X_train, y_train)
print(knn)
```

#### Evaluate the Model

Predict the class label for the mean vector of the training data and evaluate the model on the test set.

```python
# Function to predict class label for the mean vector
def knn_predict_on_mean(knn, X_train):
    means = X_train.mean().values.reshape(1, -1)
    return knn.predict(means)

# Predict on mean vector
mean_prediction = knn_predict_on_mean(knn, X_train)
print(f"Prediction on mean vector: {mean_prediction}")

# Function to predict class labels on the test set
def knn_predict_on_test(knn, X_test):
    return knn.predict(X_test)

# Predict on test set
test_predictions = knn_predict_on_test(knn, X_test)
print(test_predictions)

# Function to compute model accuracy
def knn_score_prediction(knn, X_test, y_test):
    return knn.score(X_test, y_test)

# Evaluate model accuracy
accuracy = knn_score_prediction(knn, X_test, y_test)
print(f"Model accuracy: {accuracy}")
```

### 6. Hyper-Parameter Tuning

Perform a parameter sweep for odd values of k from 1 to 19 and find the optimal k.

```python
# Function to perform hyper-parameter tuning
def knn_hyperparameter_tuning(X_train, y_train, X_test, y_test):
    best_accuracy = 0
    k_best = None
    for k in range(1, 20, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            k_best = k
    return k_best

# Perform hyper-parameter tuning
best_k = knn_hyperparameter_tuning(X_train, y_train, X_test, y_test)
print(f"Optimal k: {best_k}")
```

### 7. One-Hot Encoding Example

Example of using one-hot encoding on a secondary dataset to handle categorical data.

```python
# Load a sample housing prices dataset for illustration
def get_house_prices_data():
    data = pd.read_csv("./assets/housing_prices.csv")
    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]
    X["Electrical"].fillna("Mix", inplace=True)
    return X, y

# Function to split the housing prices data
def get_house_prices_split(random_state=42):
    X, y = get_house_prices_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Function to perform one-hot encoding
from sklearn.preprocessing import OneHotEncoder

def one_hot_encoding(encode_features, handle_unk="error"):
    X_train, X_test, y_train, y_test = get_house_prices_split()
    encoder = OneHotEncoder(handle_unknown=handle_unk, sparse_output=False, dtype=int)
    ohe_indices = [X_train.columns.get_loc(feature) for feature in encode_features]
    encoder.fit(X_train.iloc[:, ohe_indices])
    ohe_features_array = encoder.transform(X_train.iloc[:, ohe_indices])
    ohe_feature_names = encoder.get_feature_names_out(input_features=encode_features)
    ohe_df_train = pd.DataFrame(ohe_features_array, columns=ohe_feature_names, index=X_train.index)
    X_train = pd.concat([X_train, ohe_df_train], axis=1)
    ohe_features_array_test = encoder.transform(X_test.iloc[:, ohe_indices])
    ohe_df_test = pd.DataFrame(ohe_features_array_test, columns=ohe_feature_names, index=X_test.index)
    X_test = pd.concat([X_test, ohe_df_test], axis=1)
    return X_train, X_test, y_train, y_test, ohe_feature_names

# Test one-hot encoding function
X_tr, X_te, y_tr, y_te, ohe_feat = one_hot_encoding(["BldgType"])
print(ohe_feat)
display(X_tr.head())
```

## Setup Instructions

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/akamal341/breast-cancer-diagnosis.git
    cd breast-cancer-diagnosis
    ```

2. **Install Dependencies**:
    ```sh
    pip install pandas numpy scikit-learn
    ```

3. **Run the Jupyter Notebook**:
    Open and run the provided Jupyter Notebook to see the data manipulation and analysis.

## Key Results

1. **Data Transformation:** Cleaned and transformed the dataset for analysis.
2. **Descriptive Statistics:** Calculated class distribution and other statistics.
3. **Model Training and Evaluation:** Developed a kNN classifier and evaluated its performance.
4. **Hyper-Parameter Tuning:** Identified the optimal hyperparameters for the kNN model.
5. **One-Hot Encoding Example:** Illustrated handling categorical data using one-hot encoding.

## Contact Information

For any questions or further information, please contact:
- **Name:** Asad Kamal
- **Email:** aakamal {/@/} umich {/dot/} edu
- **LinkedIn:** [LinkedIn Profile](https://linkedin.com/in/asadakamal)

---

Feel free to adjust the file paths, repository URL, and any additional details specific to your project. This README ensures that your project is informative and provides all necessary steps for replication and understanding.

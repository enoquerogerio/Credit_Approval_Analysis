# Credit Approval Analysis

This project demonstrates a detailed analysis of credit approval data using machine learning models and data visualization techniques.

## Table of Contents
1. [Introduction](#introduction)
2. [Technologies Used](#technologies-used)
3. [Setup and Installation](#setup-and-installation)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Results and Visualization](#results-and-visualization)
7. [Saving Results](#saving-results)

---

## Introduction

This project analyzes a dataset of credit approvals, leveraging machine learning models to predict credit approval outcomes. The models used include:
- Decision Tree Classifier (with hyperparameter tuning via GridSearchCV)
- Logistic Regression (for comparison)

The dataset was preprocessed to handle missing values and categorical variables. Feature importance and correlations were visualized to gain insights into the data.

---

## Technologies Used

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `json`

---

## Setup and Installation

This project was developed using Google Colab. To replicate the analysis:

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the script file (`Credit_Approval.ipynb`) to your Colab environment.
3. Upload the dataset (`crx.data`) to the Colab environment.
4. Run the script cell by cell.

No additional installation is required since Google Colab comes with most dependencies pre-installed. If any libraries are missing, install them using:
```python
!pip install library_name
```

---

## Data Preprocessing

The dataset contains both numeric and categorical features. Missing values were handled as follows:
- Numeric columns: Replaced with the mean value.
- Categorical columns: Replaced with the most frequent value and encoded using `LabelEncoder`.

Features were scaled using `StandardScaler` for better model performance.

---

## Model Training and Evaluation

1. **Decision Tree Classifier**
   - Hyperparameter tuning with GridSearchCV:
     - `max_depth`
     - `min_samples_split`
     - `min_samples_leaf`
   - Achieved an accuracy of **X.XXXX**.

2. **Logistic Regression**
   - Comparison model.
   - Achieved an accuracy of **X.XXXX**.

Evaluation metrics included:
- Accuracy
- Classification Report
- Confusion Matrix

---

## Results and Visualization

1. **Decision Tree Visualization**
   - A graphical representation of the optimized decision tree is provided.

2. **Feature Importance**
   - Features were ranked based on their importance in the decision tree model.

3. **Correlation Matrix**
   - Highlighted relationships between features.

4. **Histograms**
   - Showed the distribution of numeric features, categorized by the target class.

---

## Saving Results

Key results were saved to a JSON file (`resultados_analise_credito.json`) for easy sharing and further analysis. These include:
- Model accuracies
- Best hyperparameters for the Decision Tree
- Feature importances




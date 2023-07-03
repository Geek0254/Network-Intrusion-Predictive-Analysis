# Network-Intrusion-Predictive-Analysis
This repository contains code for a Random Forest Classification machine learning model that performs predictive analysis. The model is built using Python and leverages the scikit-learn library.

## Datasets
The model uses the following datasets:
- `Train_data.csv`: The training dataset.
- `Test_data.csv`: The test dataset.

## Preprocessing
The preprocessing steps include handling missing values and encoding categorical variables.

## Feature Selection
The model uses feature selection techniques to identify the most important features for prediction.

## Training and Testing
The model is trained using the RandomForestClassifier algorithm and tested on the test dataset. Accuracy scores and classification reports are generated for both training and test data.

## Saving the Model
The trained model is saved as `random_forest_model.pkl` using the joblib library.

## Saving the Scaler
The scaler used for feature scaling is saved as `scaler1.pkl` using the joblib library.

## Results
Classification Report: Provides precision, recall, F1-score, and support for each class.
ROC AUC Score: Measures the model's performance using the Receiver Operating Characteristic (ROC) curve.
Confusion Matrix: Visualizes the performance of the model by showing true positive, true negative, false positive, and false negative predictions.

## Prerequisites

- Python 3.x
- scikit-learn
- pandas
- numpy
- seaborn
- matplotlib

# Diabetes Prediction Web Application

This project is a Flask-based web application that predicts the likelihood of a patient having diabetes based on various health metrics. The machine learning model used for prediction is built using Logistic Regression and is trained on the `.

## Table of Contents

1. [Features](#features)
2. [Technologies](#technologies)
3. [Working Principle](#working-principle)
4. [Usage](#usage)
5. [Model Information](#model-information)
6. [Use Cases](#use-cases)

## Features

1. User-friendly form to input health-related data such as glucose levels, BMI, and age.
2. Predicts if a person is likely diabetic based on the input features.
3. Uses a machine learning model for real-time predictions.
4. Animated, modern UI built with Bootstrap and CSS animations.

## Technologies

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn, NumPy
- **Frontend**: HTML5, CSS3, Bootstrap 5
- **Model Storage**: Joblib

## Working Principle

1. **User Input**: The web application provides a form for users to input key health metrics such as pregnancies, glucose level, blood pressure, BMI, etc.
2. **Model Prediction**: The input data is processed and passed to the trained **Logistic Regression** model.
3. **Result**: The model predicts whether the individual is likely diabetic (Positive) or not (Negative), based on the input data.
4. **Feedback**: The result is displayed on the screen with an informative message like "Diabetic" or "Non-Diabetic".
5. **Backend Mechanism**:
   - The input data is normalized using a **StandardScaler** before being passed into the Logistic Regression model for prediction.
   - The trained model, stored as a `pkl` file, is loaded during the app initialization and used for real-time predictions.

## Usage

1. Input relevant health data such as:
      Number of pregnancies
      Glucose level
      Blood pressure
      Skin thickness
      Insulin level
      BMI
      Diabetes Pedigree Function
      Age
2. Click the "Predict" button to check whether the person is likely to have diabetes.
3. The model will return a result indicating whether the patient is "Diabetic" or "Non-Diabetic."
   

## Model Information

The prediction model is a Logistic Regression model trained using the [https://www.kaggle.com/datasets/rahulsah06/machine-learning-for-diabetes-with-python/data) Diabetes dataset.
The model pipeline includes a StandardScaler for feature normalization before passing the data into the Logistic Regression model.
The model achieves an accuracy of approximately 77.9% using cross-validation.

## Use Cases
**Medical Professionals:**
   Doctors and healthcare providers can use the app to quickly assess the likelihood of diabetes based on patient data during checkups.
   It can serve as a decision support tool, allowing doctors to make better-informed decisions when diagnosing diabetes.

**Health-Conscious Individuals:**
   Individuals who are concerned about their health can use this app to monitor and assess their risk of diabetes.
   It provides a simple way to input health metrics and get an immediate prediction, promoting awareness of diabetes risk factors.
   
**Health Clinics & Wellness Centers:**
   Clinics that focus on preventive healthcare can use this app to assess the diabetes risk of patients and suggest further testing or lifestyle changes if            necessary.
   Wellness centers can integrate it into their health checkup routines to offer risk assessment as part of the package.

**Healthcare Students and Researchers:**
   Students studying healthcare or machine learning can use this app to understand how machine learning models can be applied in a real-world healthcare setting.
   It can serve as a practical project or study resource, demonstrating how logistic regression models can predict medical conditions.

**Corporate Health Programs:**
   Companies that have corporate wellness programs can use this tool to assess the health of employees and promote preventive measures.
   It can be used in health camps or wellness drives to help detect early warning signs of diabetes.

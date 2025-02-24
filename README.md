# Medical Insurance Prediction Using Machine Learning

# Overview
This project uses Machine Learning and Deep Learning techniques to predict medical insurance costs based on user attributes such as age, gender, BMI, number of children, smoking habits, and region. By analyzing these factors, insurance companies can estimate premium charges more accurately.

# Features
✅ Data Preprocessing: Handles categorical and numerical data using One-Hot Encoding and Feature Scaling
✅ Exploratory Data Analysis (EDA): Uses Seaborn & Matplotlib for data visualization and correlation analysis
✅ Machine Learning Model: Implements Multiple Linear Regression for insurance cost prediction
✅ Deep Learning Model: Uses a Neural Network (ANN) with TensorFlow/Keras for advanced predictions
✅ Model Evaluation: Computes RMSE, R² Score for performance assessment
✅ Data Visualization: Heatmaps, scatter plots, and pair plots to analyze trends
✅ Model Saving & Deployment: Saves trained models for future use

# Technologies Used
- Python 
- Pandas, NumPy (Data Processing)
- Seaborn, Matplotlib (Visualization)
- Scikit-learn (Machine Learning)
- TensorFlow/Keras (Deep Learning)

# Dataset
The dataset used is insurance.csv, which contains:
- age: Age of the policyholder
- sex: Gender (Male/Female)
- bmi: Body Mass Index
- children: Number of dependents
- smoker: Smoking status (Yes/No)
- region: Residential area
- charges: Insurance cost (Target variable)

# Installation & Usage
Install dependencies
pip install -r requirements.txt

# Run the script
python Medical Insurance.py

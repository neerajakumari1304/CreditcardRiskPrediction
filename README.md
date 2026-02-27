# CreditcardRiskPrediction
Developed ML pipeline using Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, and XGBoost. 

## 🚀 Project Overview
The project implements a full ML lifecycle, including data cleaning, advanced feature engineering, handling class imbalance, model benchmarking, and deployment via a Flask web application.

## 🛠 Project StructureThe project is modularized into several Python scripts, each handling a specific stage of the pipeline:

**1. Data Ingestion & Preprocessing (main.py)**
- Data Loading: Loads the initial dataset (150,002 rows) and performs initial cleaning.
- Column Cleanup: Removes redundant columns like MonthlyIncome.1.
-  Data Splitting: Splits the data into training (120,000 rows) and testing (30,000 rows) sets. **2. Missing Value Imputation (random_sample.py)**
-  Technique: Uses Random Sample Imputation for features like MonthlyIncome and NumberOfDependents to maintain the original distribution of the data.
**3. Feature Engineering (var_out.py & main.py)**
- Transformation: Applies Yeo-Johnson transformations to numerical features to handle non-normal distributions.
- Outlier Handling: Implements trimming based on Interquartile Range (IQR) to mitigate the impact of extreme values.
- Categorical Encoding: Utilizes OneHotEncoder for nominal data (Gender, Region) and OrdinalEncoder for ranked data (Education, Occupation).
**4. Feature Selection (feature_selection.py)**
- Variance Thresholding: Removes constant and quasi-constant features that provide little predictive power.
- Hypothesis Testing: Uses Pearson correlation to identify the most significant features relative to the target variable.
**5. Imbalance Management (imbalance_data.py)**
- SMOTE: Employs Synthetic Minority Over-sampling Technique (SMOTE) to balance the "Good" and "Bad" customer classes.
- Scaling: Standardizes the balanced dataset using StandardScaler to ensure all features are on a comparable scale.
**6. Model Benchmarking (all_models.py)**
Trains and evaluates multiple classifiers, generating a comparative ROC curve for:
  - K-Nearest Neighbors (KNN)
  - Naive Bayes (NB)
  - Logistic Regression (LR)
  - Decision Tree (DT)
  - Random Forest (RF)
  - AdaBoost Gradient Boosting (GB)
  - XGBoost (XGB)
  - Support Vector Machine (SVM)
## 🌐 Web Application & Deployment (app.py)
A user-friendly Flask application is provided to serve the model.
- Model Loading: Loads the trained Logistic Regression model (creditcard.pkl) and the fitted scaler (scalar.pkl).
- Web Interface: An index.html form allows users to input details such as Age, Monthly Income, and Education.
- Real-time Prediction: Returns a classification of "Good Customer" or "Bad Customer" based on user input.
## 📊 Performance Summary
The top-performing models identified in the logs include:
- Naive Bayes: ~93.37% Test Accuracy
- Logistic Regression: ~92.57% Test Accuracy
## 💻 How to Run
## Install Dependencies:
`install flask numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn ` 
## Train the Models:
Run main.py to execute the full pipeline.

## Start the App:
`python app.py`
## Access: 
Open `http://127.0.0.1:5000/` in your browser.

## Project By Neeraja - Data Scientist

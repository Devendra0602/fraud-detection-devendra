# Fraud Detection System
### By Devendra Patil

## Project Overview
An end-to-end fraud detection system built using the IEEE-CIS dataset with 590,000 transactions. 
The system uses XGBoost with SMOTE balancing and SHAP explainability to detect credit card fraud.

## Live Dashboard
https://fraud-detection-devendra-bquvznqvqcnnkz7dv3awek.streamlit.app/

## Results
- Best Model: XGBoost (tuned with Optuna)
- ROC-AUC: 0.9666
- F1 Score: 0.7737
- Recall: 0.68
- Optimal Threshold: 0.35

## Project Structure
FraudDetection_DevendraPatil/
├── analysis.ipynb
├── data/
├── dashboard/
│   ├── app.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── results.csv
│   └── X_test.csv
├── charts/
├── requirements.txt
└── README.md

## Tools Used
- Python, Pandas, NumPy
- LightGBM, XGBoost, Scikit-learn
- SMOTE (imbalanced-learn)
- SHAP
- Optuna
- Streamlit
- Matplotlib, Seaborn

## How to Run Locally
cd dashboard
streamlit run app.py


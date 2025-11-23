💳🔍 Credit Card Fraud Detection 🔍💳

📌 Project Overview
This project uses **Logistic Regression** to detect fraudulent credit card transactions.  
Because the dataset is highly imbalanced, **SMOTE oversampling** is applied to improve the model’s ability to catch fraudulent cases.  
Visualizations such as class distribution help understand imbalance before modeling.


🧠 Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-Learn (SMOTE)  
- Matplotlib, Seaborn  
- Joblib  


🚀 Features
- Data cleaning & preprocessing  
- Imbalanced class visualization  
- Stratified Train/Test split  
- SMOTE oversampling  
- Logistic Regression model training  
- Model saved using **model.joblib**  
- Detailed evaluation:
  - Accuracy  
  - Classification Report  
  - Confusion Matrix  


📊 Model Performance
- **Accuracy:** ~99%  
- **High Recall** for fraud detection  
- Balanced classification after SMOTE  
- Model can effectively distinguish between genuine and fraudulent transactions  


📂 Project Structure
- project.ipynb
- data.csv
- model.joblib


📁 Dataset
- The `data.csv` file contains anonymized transaction features along with a fraud/non-fraud label.

## 📂 Project Structure

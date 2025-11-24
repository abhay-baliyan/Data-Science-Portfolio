❤️ Early Detection of Cardiovascular Disease Using Logistic Regression ❤️

A machine learning approach to predict the presence of heart disease using standard medical attributes such 
as age, cholesterol, resting blood pressure, and chest pain type.
This project demonstrates a complete ML workflow—data analysis, visualization, model building, and performance 
evaluation—aimed at supporting early diagnosis of cardiovascular conditions.



🧑‍💻 Author
Abhay Baliyan



📄 Abstract
Cardiovascular diseases (CVDs) are one of the leading causes of mortality worldwide. Early detection is crucial for 
enabling timely intervention and reducing healthcare risks.
This study presents a machine learning model built using Logistic Regression to predict heart disease based on patient 
clinical attributes.
A Kaggle dataset with clean and well-structured medical features was used, requiring minimal preprocessing.

The proposed model achieved:
Accuracy: 85.12%
Precision: 84.32%
Recall: 89.23%
F1-Score: 86.71%
The results show strong potential for integrating ML into clinical workflows for early screening and risk assessment.



🔑 Keywords
Cardiovascular Disease, Machine Learning, Logistic Regression, Healthcare Analytics, Classification, Predictive Modeling, Early Diagnosis.



🏥 1. Introduction
Cardiovascular diseases cause nearly 17.9 million deaths annually, according to the World Health Organization.
Early identification of at-risk individuals can drastically reduce mortality rates.
Traditional diagnosis relies on medical expertise and manual interpretation of diagnostic tests.
However, with advancements in digital health records and machine learning, predictive models can significantly enhance clinical decision-making.
This study builds an interpretable ML model—Logistic Regression—to classify whether a patient is likely to have heart disease based on their medical parameters.



📚 2. Literature Review
Machine learning has been widely explored for medical diagnosis:
Detrano et al. compared logistic regression and neural networks on coronary artery disease prediction.
SVMs and Ensemble models like XGBoost have achieved strong performance.
However, complex models sacrifice explainability—critical for healthcare adoption.
Thus, Logistic Regression remains a popular choice for medical decision support because:
✔ It is simple
✔ Highly interpretable
✔ Clinically explainable
This project builds on prior work using a well-known dataset and a transparent ML model.



📊 3. Methodology

3.1 Dataset
Dataset Source: Kaggle (Heart Disease Dataset)
Total Records: 303
Attributes: 14 medical features + target
Attribute	Meaning
age	Age
sex	Sex
cp	Chest pain type
trestbps	Resting blood pressure
chol	Serum cholesterol
fbs	Fasting blood sugar
restecg	Resting ECG results
thalach	Maximum heart rate
exang	Exercise-induced angina
oldpeak	ST depression
slope	Slope of ST segment
ca	Major vessels colored
thal	Heart health condition
target	1 = Disease, 0 = No Disease

3.2 Data Preprocessing
No missing values were found.
Dataset was clean and ready for modeling.
Exploratory Data Analysis (EDA) included:
Class distribution
Age–cholesterol visualizations
Chest pain vs. disease
Correlation heatmap

📌 Figure: Heart Disease Distribution (0 = No Disease, 1 = Disease)
📌 Figure: Correlation Matrix of Features



⚙️ 4. Model Development

4.1 Train-Test Split
Dataset split using 60:40 ratio:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=41)

4.2 Model Used — Logistic Regression
Reasons:
Binary classification
High interpretability
Clinical relevance
Low computational cost
model = LogisticRegression(max_iter=1500)
model.fit(X_train, y_train)



📈 5. Results
The model was evaluated using standard classification metrics:
Metric	Score (%)
Accuracy	85.12
Precision	84.32
Recall	89.23
F1-Score	86.71

📌 Figure: Confusion Matrix (Logistic Regression)



🔍 Interpretation
High recall (89.23%) → Model is excellent at identifying patients with heart disease.
Strong precision (84.32%) → Minimizes false alarms.
F1-score (86.71%) → Balanced and robust performance.
These metrics confirm the reliability of logistic regression for early detection.



💬 6. Discussion
Despite being a simple linear model, logistic regression achieved strong results:
Minimal preprocessing required
Clean medical dataset
Strong evaluation metrics
High interpretability → allows doctors to understand prediction factors
Potential improvements:
Add other ML models (Random Forest, XGBoost)
Hyperparameter tuning
Cross-validation
Feature engineering
Deploying the model with an API or healthcare dashboard



🏁 7. Conclusion
This study successfully developed a logistic regression model for predicting heart disease using patient medical attributes. 
Achieving an accuracy of 85.12% and F1-score of 86.71%, the model demonstrates strong potential for aiding clinicians in early detection.
The project shows how machine learning can:
Improve diagnostic accuracy
Reduce manual workload
Enable data-driven healthcare
Support preventive treatment
Future work may involve using advanced models and integrating the solution into real-time clinical systems.



📂 Project Files
cardiovascular-disease-prediction/
│
├── project.ipynb                 # Full EDA + Model Notebook
├── data.csv                      # Dataset
├── model.joblib                  # Trained Model
├── README.md                     # Project Documentation
│
└── research-paper/
    ├── heart_disease_prediction_research.pdf   # Final research paper
    └── heart_disease_prediction_research.docx 



▶️ How to Run
Install dependencies:
pip install numpy pandas matplotlib seaborn scikit-learn joblib
Run the notebook:
project.ipynb

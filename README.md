Telco Customer Churn Prediction (Machine Learning Project)
This project builds a customer churn prediction model using the Telco Customer Churn dataset.
It includes full exploratory data analysis (EDA), preprocessing, model training, evaluation, and a reusable inference pipeline.
The goal is to identify customers who are most likely to stop using telecom services, enabling proactive retention strategies.
🚀 Project Highlights
Full end-to-end machine learning pipeline
Clean and modular Python code (src/ folder)
Exploratory Data Analysis in Jupyter Notebook
Feature engineering and preprocessing pipeline
Machine learning model training and optimisation
Evaluation metrics and threshold selection
Saved models for reuse and deployment
Professional project structure suitable for portfolio use
📂 Repository Structure
telco-churn-ml/
│
├── data/                        # Dataset (CSV)
│
├── notebooks/
│   └── 01_eda_telco_churn.ipynb # EDA notebook
│
├── src/                         # Python modules (ML pipeline)
│   ├── preprocess.py            # Data cleaning & encoding
│   ├── train.py                 # Model training script
│   ├── evaluation.py            # Evaluation metrics
│   ├── predict.py               # Predict churn for new customers
│   └── threshold.py             # Threshold optimisation
│
├── models/
│   ├── telco_churn_model.pkl    # Final trained model
│   └── decision_threshold.pkl   # Optimized classification threshold
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
📊 Exploratory Data Analysis (EDA)
The EDA notebook includes:
Customer demographics analysis
Contract types, payment methods, and tenure patterns
Service usage analysis (internet, phone, streaming services)
Churn distribution and imbalance check
Correlation heatmaps
Visual patterns related to churn
Key Findings:
Month-to-month contracts → highest churn
Electronic check payment → most churned customers
Higher monthly charges → strong churn indicator
Senior citizens and short-tenure customers churn more frequently
🤖 Machine Learning Pipeline
1️⃣ Preprocessing (preprocess.py)
Categorical encoding
Numerical standardisation
Missing value handling
Feature selection
2️⃣ Model Training (train.py)
Train/test split
Logistic Regression / Random Forest (based on your script)
Hyperparameter choices
Model saved using joblib
3️⃣ Threshold Optimization (threshold.py)
Finds the best probability threshold for classification
Improves recall and precision for churn cases
4️⃣ Model Evaluation (evaluation.py)
Computes:
Accuracy
Precision
Recall
F1-score
Confusion matrix
5️⃣ Prediction Script (predict.py)
Loads:
telco_churn_model.pkl
decision_threshold.pkl
Then predicts churn for new customer data.
💾 Saved Models
File	Description
telco_churn_model.pkl	Final trained churn classifier
decision_threshold.pkl	Best probability threshold for classification
▶️ How to Run the Project
Install dependencies
pip install -r requirements.txt
Train the model
python src/train.py
Evaluate performance
python src/evaluation.py
Run predictions
python src/predict.py
🧠 Conclusion
This project demonstrates:
Strong understanding of the complete ML lifecycle
Clean, modular Python code suitable for real-world use
Reproducible pipeline with saved models
Solid EDA and feature engineering
Professional GitHub structure ideal for job applications
✔️ Great for Data Analyst roles
✔️ Great for Machine Learning Engineer roles
✔️ Excellent addition to your GitHub portfolio

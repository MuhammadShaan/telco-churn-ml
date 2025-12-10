Telco Customer Churn Prediction (Machine Learning Project)
This project builds a customer churn prediction model using the popular Telco Customer Churn dataset.
It includes full data exploration, preprocessing, model training, evaluation, and a reusable inference pipeline.
The goal is to identify customers who are most likely to stop using telecom services, enabling proactive retention strategies.
🚀 Project Highlights
Complete end-to-end ML pipeline
Clean and modular code (Python scripts in src/)
Exploratory Data Analysis notebook
Preprocessing pipeline
Model training + hyperparameters
Threshold optimisation
Evaluation metrics
Saved models ready for deployment (models/ folder)
📂 Repository Structure
telco-churn-ml/
│
├── data/                    # Dataset (CSV)
│
├── notebooks/
│   └── 01_eda_telco_churn.ipynb   # EDA notebook
│
├── src/                     # Python modules for ML pipeline
│   ├── preprocess.py        # Data cleaning and encoding
│   ├── train.py             # Model training script
│   ├── evaluation.py        # Evaluation metrics
│   ├── predict.py           # Make new predictions
│   └── threshold.py         # Threshold optimisation
│
├── models/
│   ├── telco_churn_model.pkl       # Final trained model
│   └── decision_threshold.pkl       # Optimized classification threshold
│
├── requirements.txt          # Python environment
└── README.md                 # Project documentation
📊 EDA Summary
Explored customer demographics, services, and billing
Identified missing values and outliers
Investigated churn patterns
Found important drivers such as:
Contract type
Monthly charges
Tenure
Payment method
Internet service
Visualisations include distributions, correlations, and churn comparisons.
🤖 Machine Learning Pipeline
1️⃣ Preprocessing
Handled in preprocess.py:
Convert numerical and categorical features
One-hot encoding
Missing value handling
Scaling of numeric columns
2️⃣ Model Training
Performed via train.py:
Logistic Regression / Random Forest (depending on your script)
Train-test split
Model saving using joblib
3️⃣ Threshold Optimization
threshold.py selects the best decision threshold for churn classification.
4️⃣ Evaluation
evaluation.py computes:
Accuracy
Precision
Recall
F1-score
Confusion matrix
5️⃣ Prediction Script
predict.py loads the model and predicts churn for new customer data.
📁 Saved Models
telco_churn_model.pkl → The trained model
decision_threshold.pkl → Best classification threshold
These are used by predict.py for inference.
▶️ How to Run the Project
Install dependencies:
pip install -r requirements.txt
Train the model:
python src/train.py
Evaluate the model:
python src/evaluation.py
Run predictions:
python src/predict.py
🧠 Conclusion
This project demonstrates:
Strong understanding of machine learning workflow
Ability to structure a real-world ML project professionally
Clear separation of concerns (EDA vs scripts vs models)
Reproducible code and saved models
It is suitable for inclusion in:
✔️ Data Analyst portfolio
✔️ Machine Learning Engineer applications
✔️ Python projects on GitHub

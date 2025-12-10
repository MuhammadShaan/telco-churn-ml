# Telco Customer Churn Prediction (Machine Learning Project)

This project builds a customer churn prediction model using the Telco Customer Churn dataset.  
It includes full exploratory data analysis (EDA), preprocessing, model training, evaluation, and a reusable inference pipeline.  
The goal is to identify customers who are most likely to stop using telecom services, enabling proactive retention strategies.  

---

## 🚀 Project Highlights

- Full end-to-end machine learning pipeline  
- Clean and modular Python code (`src/` folder)  
- Exploratory Data Analysis in Jupyter Notebook  
- Feature engineering and preprocessing pipeline  
- Model training and threshold optimisation  
- Evaluation metrics  
- Saved models ready for deployment  
- Professional project structure suitable for portfolio use  

---

## 📂 Repository Structure

## 📂 Repository Structure

telco-churn-ml/
│
├── data/ # Dataset (CSV)
│
├── notebooks/
│ └── 01_eda_telco_churn.ipynb # EDA notebook
│
├── src/ # Python modules (ML pipeline)
│ ├── preprocess.py # Data cleaning & encoding
│ ├── train.py # Model training script
│ ├── evaluation.py # Evaluation metrics
│ ├── predict.py # Predict churn for new customers
│ └── threshold.py # Threshold optimisation
│
├── models/
│ ├── telco_churn_model.pkl # Final trained model
│ └── decision_threshold.pkl # Optimized classification threshold
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 📊 Exploratory Data Analysis (EDA)

The EDA notebook includes:

- Customer demographics  
- Contract types, payment methods, tenure  
- Service usage analysis  
- Churn distribution & imbalance check  
- Correlation heatmaps  
- Visual patterns related to churn  

### Key Findings

- Month-to-month contracts → highest churn  
- Electronic check payment → major churn indicator  
- Higher monthly charges → higher churn probability  
- Senior citizens & short-tenure customers churn more  

---

## 🤖 Machine Learning Pipeline

### 1️⃣ Preprocessing (`preprocess.py`)  
- Categorical encoding  
- Numerical scaling  
- Missing value handling  
- Feature selection  

### 2️⃣ Model Training (`train.py`)  
- Logistic Regression / Random Forest  
- Train/test split  
- Model saved using joblib  

### 3️⃣ Threshold Optimization (`threshold.py`)  
- Selects best probability threshold for churn classification  

### 4️⃣ Evaluation (`evaluation.py`)  
Computes:  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix  

### 5️⃣ Prediction (`predict.py`)  
- Loads model + threshold  
- Predicts churn for new customer data  

---

## 💾 Saved Models

| File | Description |
|------|-------------|
| `telco_churn_model.pkl` | Final trained classifier |
| `decision_threshold.pkl` | Optimal probability threshold |

---

## ▶️ How to Run the Project

### Install dependencies
pip install -r requirements.txt

### Train model
python src/train.py

### Evaluate model
python src/evaluation.py

### Predict churn
python src/predict.py

---

## 🧠 Conclusion

This project demonstrates:

- Full ML lifecycle understanding  
- Clean modular Python scripts  
- Reproducible workflow with saved models  
- Strong EDA & feature engineering  
- Professional project structure  

Suitable for:

✔️ Data Analyst roles  
✔️ Machine Learning Engineer roles  
✔️ Portfolio / GitHub showcase  

---


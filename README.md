# 💰 EMI Prediction System

## 📌 Overview

This project predicts:

* Loan Eligibility (Eligible / High Risk / Not Eligible)
* Maximum EMI a user can afford

Built using Machine Learning and deployed with Streamlit.

---

## 🚀 Features

* Classification + Regression model
* Real-world financial feature engineering
* Handles imbalanced data
* Interactive UI

---

## 🧠 Models Used

* Logistic Regression
* Random Forest
* XGBoost

---

## 🖥️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
EMI_Predict/
├── app.py
├── processed_data/
├── notebooks/
├── data/
```
---
## 📊 Model Performance

### Classification (XGBoost)

* Accuracy: ~0.92
* F1 Score (Weighted): ~0.93
* Minority Class Recall: ~0.60

### Regression (XGBoost)

### Regression (XGBoost)

* R² Score: 0.9814  
* RMSE: ₹1052.83  
* MAE: ₹467.24  
* MAPE: 16.94%  

---

## 🧑‍💻 Author
Rama Naren  
Aspiring Data Scientist

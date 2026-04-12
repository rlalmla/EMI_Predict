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

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd EMI_Predict
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download dataset

Download the datasets from the **Dataset section below** and place them inside the `data/` folder.

Final structure should look like:

```
data/
├── emi_prediction_dataset.csv
├── cleaned_data.csv
```

### 4. Run the application

```bash
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
## 📂 Dataset

Due to size constraints, datasets are not included in this repository.

- Raw Dataset: https://drive.google.com/file/d/1yIXzljTpNkm-nItWDHKCybWJ6I-x87Vm/view?usp=sharing
- Cleaned Dataset: https://drive.google.com/file/d/12JUdSzyWFYAfHziehNsWZELDGZwfIXtf/view?usp=sharing

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

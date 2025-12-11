# 📈 Google Stock Price Prediction using LSTM

A deep learning project to forecast **Google (GOOGL) stock Open & Close prices** using a **Long Short-Term Memory (LSTM)** network.

---

## 📝 Project Overview

This project uses historical Google stock data to:

* Clean & preprocess time-series data
* Scale features using **MinMaxScaler**
* Create 50-day input sequences
* Train an **LSTM model** to predict next-day **open** and **close** prices
* Compare predicted vs. actual values
* Forecast **10 upcoming days** of stock prices

The notebook includes full visualization and evaluation.

---

## 📂 Dataset

Dataset used: **Google Stock Prediction Dataset**
🔗 [https://www.kaggle.com/datasets/shreenidhihipparagi/google-stock-prediction](https://www.kaggle.com/datasets/shreenidhihipparagi/google-stock-prediction)

Columns used:

* `date`
* `open`
* `close`

---

## 🧠 Model Architecture

The model is a **stacked LSTM**:

```text
LSTM (50 units, return_sequences=True)
Dropout (0.1)
LSTM (50 units)
Dense (2 units → open & close output)
Loss: MSE
Optimizer: Adam
Metric: MAE
```

This model captures long-term temporal dependencies in stock price movements.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* scikit-learn
* TensorFlow / Keras

---

## 🚀 Workflow Breakdown

### **1️⃣ Data Loading & Cleaning**

* Read CSV file
* Convert date to datetime
* Set date as index
* Plot Open & Close price trends

### **2️⃣ Data Preprocessing**

* Apply **MinMaxScaler**
* 80–20 train-test split
* Create 50-day sequences using a custom `create_sequence()` function

### **3️⃣ Model Training**

* Build the LSTM model using Keras
* Train for **80 epochs**
* Validate on test data

### **4️⃣ Predictions**

* Predict open & close prices on test set
* Inverse-scale predictions
* Merge with original data
* Visualize actual vs predicted values

### **5️⃣ Future Forecast**

Predict **next 10 days** using autoregressive rolling prediction.

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

---

## ▶️ Running the Notebook

```bash
jupyter notebook Google_Stock_prediction_using_LSTM.ipynb
```

---

## 📦 Requirements (`requirements.txt`)

```
pandas
numpy
matplotlib
scikit-learn
tensorflow
keras
```

---

## 🏁 Results Summary

* The LSTM successfully tracks the trend of real stock prices.
* Predictions closely follow the actual values after inverse-scaling.
* Future 10-day predictions provide a smooth continuation of the trend.

This project demonstrates the effectiveness of LSTMs for time-series forecasting.

---

## 🙌 Acknowledgements

Dataset by **Shreenidhi Hipparagi**
Kaggle link:
[https://www.kaggle.com/datasets/shreenidhihipparagi/google-stock-prediction](https://www.kaggle.com/datasets/shreenidhihipparagi/google-stock-prediction)

---

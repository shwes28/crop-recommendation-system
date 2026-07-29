# 🌾 CropSense AI — Crop Recommendation System

> AI-powered crop recommendation using soil chemistry & climate data. Trained on 2,200 farm records across 22 crop types using Random Forest, SVM, and Naive Bayes classifiers.

---

## 📋 Features

- **3 ML Models** — Random Forest (~99.3%), SVM, Naive Bayes
- **7 Input Features** — N, P, K, Temperature, Humidity, pH, Rainfall
- **22 Crop Types** — Rice, Maize, Mango, Banana, Cotton, and more
- **Interactive Streamlit UI** — Dark-themed, real-time predictions with confidence gauges
- **Comprehensive EDA** — Correlation heatmap, box plots, scatter plots, distributions
- **Model Comparison** — Radar charts, confusion matrices, classification reports
- **Crop Guide** — Searchable reference with seasonal and soil requirements

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train models

```bash
python train_models.py
```

### 3. Launch Streamlit app

```bash
streamlit run app/app.py
```

---

## 📁 Project Structure

```
crop-recommendation/
├── data/
│   └── crop_recommendation.csv     # Dataset (2200 rows × 8 cols)
├── models/
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── naive_bayes.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── src/
│   ├── data_loader.py              # Data loading & preprocessing
│   ├── eda.py                      # EDA chart functions
│   ├── train.py                    # Model training pipeline
│   └── predict.py                  # Inference utilities
├── app/
│   └── app.py                      # Streamlit application
├── generate_dataset.py             # Dataset generator
├── train_models.py                 # Main training script
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Feature      | Description                  | Unit   |
|--------------|------------------------------|--------|
| N            | Nitrogen content in soil     | mg/kg  |
| P            | Phosphorus content in soil   | mg/kg  |
| K            | Potassium content in soil    | mg/kg  |
| temperature  | Air temperature              | °C     |
| humidity     | Relative humidity            | %      |
| ph           | Soil pH                      | —      |
| rainfall     | Annual rainfall              | mm     |
| label        | Crop type (target)           | —      |

---

## 🤖 Model Results

| Model          | Test Accuracy | CV Accuracy |
|----------------|---------------|-------------|
| Random Forest  | ~99.3%        | ~99.2%      |
| SVM            | ~97–98%       | ~97%        |
| Naive Bayes    | ~93–95%       | ~93%        |

---

## 🌱 Supported Crops

Apple, Banana, Blackgram, Chickpea, Coconut, Coffee, Cotton, Grapes, Jute,
Kidney Beans, Lentil, Maize, Mango, Mothbeans, Mungbean, Muskmelon,
Orange, Papaya, Pigeon Peas, Pomegranate, Rice, Watermelon

---

## 📦 Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — Data handling
- **Scikit-learn** — ML models
- **Streamlit** — Web UI
- **Matplotlib / Seaborn** — Static charts
- **Plotly** — Interactive visualizations
- **Joblib** — Model serialization

# 🍃 Tea Quality Classifier

This project predicts the quality of tea leaves (T1 to T4) using an image-based classification model built with TensorFlow and deployed via Streamlit.

## 📦 Features

- Upload a tea leaf image (JPG/PNG)
- Model predicts one of four quality categories (T1 to T4)
- Confidence score and explanation of prediction
- Handles invalid or low-confidence inputs gracefully

## 📊 Dataset

Source: [TeaLeafAgeQuality Dataset – Mendeley](https://data.mendeley.com/datasets/7t964jmmy3/1)  
It contains labeled images of tea leaves grouped by age:  
- T1: 1–2 days (freshest)  
- T2: 3–4 days  
- T3: 5–7 days  
- T4: 7+ days (least suitable)

## 🛠 Requirements

Install dependencies:
```bash
pip install -r requirements.txt

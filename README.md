# 🏎️ F1 2025 Predictions 2025 - XGBoost Model

**F1 2025 Predictions ** repository! This project uses **machine learning, FastF1 API data, and historical F1 race results** to predict race outcomes for the 2025 Formula 1 season.

## 🚀 Project Overview
This repository contains an **XGBoost Machine Learning model** that predicts race results based on past performance, qualifying times, free practices and other structured F1 data.

### 🔧 The model uses:
- FastF1 API for historical race data
- 2024 race results
- 2025 qualifying and practice session results
- The goal is to add more meaningful data, such as team performance and weather, throughout the season to get better results
- Feature engineering techniques to improve predictions

## 📊 Data Sources
- **FastF1 API**: Fetches lap times, tyre data, race results, and telemetry data
- **2024 Race Data**: Used to judge the prediction
- **2025 Qualifying & Practice Data**: Used for training the model
- **Historical F1 Results**: Processed from FastF1 for training the model

### 🏁 How It Works
1. **Data Collection**: The script fatches relevant F1 data using the FastF1 API.
2. **Preprocessing & Feature Engineering**: Calculates lap times, gathers tyre info, and processes speeds.
3. **Model Training**: An **XGBoost Regressor** is trained using 2025 practice and qualifying sessions, and 2024 race results.
4. **Prediction**: The model predicts race times for 2025 and ranks drivers accordingly.
5. **Evaluation**: Model performance is measured using **Mean Absolute Error (MAE)**, **Root Mean Square Error (RMSE)**, **R2 Score (R2)**.

### 📊 Dependencies
- `fastf1`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `XGBoost`

## 📂 File Structure 
- For every race, we will create a folder that holds all the main and related files for that race.

## 📌 Future Improvements
- Add **pit stop strategies** into the model
- Incorporate **weather conditions** as a feature

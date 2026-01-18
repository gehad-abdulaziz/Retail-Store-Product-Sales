#  Retail Store Footfall Prediction

A comprehensive machine learning project that predicts customer footfall in retail stores using various regression models and feature engineering techniques.

##  Overview

This project analyzes retail store sales data to predict customer footfall (traffic) using machine learning algorithms. The analysis includes comprehensive exploratory data analysis, feature engineering, and evaluation of multiple regression models to identify the most effective approach for footfall prediction.

### Key Objectives
- Predict customer footfall based on various retail metrics
- Identify the most influential features affecting store traffic
- Compare different machine learning models for optimal performance
- Provide actionable insights for retail decision-making

##  Dataset

The dataset contains **15,000 observations** with the following features:

| Feature | Description |
|---------|-------------|
| `price` | Product price |
| `discount` | Discount percentage offered |
| `promotion_intensity` | Intensity of promotional activities |
| `ad_spend` | Advertising expenditure |
| `competitor_price` | Competitor pricing |
| `stock_level` | Inventory stock level |
| `weather_index` | Weather condition index |
| `customer_sentiment` | Customer sentiment score |
| `return_rate` | Product return rate |
| **`footfall`** | **Target variable: Customer traffic count** |

##  Features

### Data Processing
-  Outlier detection and treatment using IQR method
-  Feature scaling with StandardScaler
-  Train-test split (80-20)

### Feature Engineering
Created 5 new engineered features:
- `price_discount_ratio`: Competitor price × Discount
- `promotion_adspend`: Promotion intensity × Ad spend
- `stock_return_ratio`: Stock level / (Return rate + 1)
- `discount_squared`: Discount²
- `sentiment_squared`: Customer sentiment²

### Models Implemented
1. Linear Regression (Baseline)
2. Ridge Regression
3. Lasso Regression
4. Polynomial Regression (degrees 2-5)
5. Decision Tree Regressor
6. **Random Forest Regressor**  (Best Model)
7. Support Vector Regression (RBF kernel)


### Clone the Repository
```bash
git clone https://github.com/yourusername/retail-footfall-prediction.git
cd retail-footfall-prediction
```


### Required Libraries
```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

##  Usage

### Running the Analysis
```bash
python footfall_prediction.py
```

### Making Predictions
```python
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('best_model_footfall_predictor.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare new data
new_data = np.array([[
    98.24,      # price
    49.33,      # discount
    6.57,       # promotion_intensity
    4.96,       # ad_spend
    2536.42,    # competitor_price
    50.33,      # stock_level
    0.52,       # weather_index
    6.76,       # customer_sentiment
    1219.36,    # return_rate
    # Engineered features
    49.33 * 2536.42,    # price_discount_ratio
    6.57 * 4.96,        # promotion_adspend
    50.33 / (1219.36 + 1),  # stock_return_ratio
    49.33 ** 2,         # discount_squared
    6.76 ** 2           # sentiment_squared
]])

# Scale and predict
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(f"Predicted Footfall: {prediction[0]:.2f} customers")
```

##  Models Comparison

| Model | R² Train | R² Test | MSE Test | MAE Test | CV Mean |
|-------|----------|---------|----------|----------|---------|
| Linear Regression | 0.7409 | 0.7534 | 526.06 | 18.25 | 0.7398 |
| Ridge Regression | 0.7406 | 0.7536 | 525.74 | 18.24 | 0.7396 |
| Lasso Regression | 0.7306 | 0.7464 | 540.98 | 18.54 | 0.7297 |
| Polynomial (deg 3) | 0.8133 | 0.8021 | 422.28 | 15.99 | 0.7892 |
| Decision Tree | 0.8560 | 0.7055 | 628.34 | 18.89 | 0.6876 |
| **Random Forest**  | **0.9531** | **0.7941** | **439.36** | **16.08** | **0.7842** |
| SVR (RBF) | 0.8185 | 0.8075 | 410.79 | 15.65 | 0.7963 |

##  Results

### Best Model: Random Forest Regressor
- **R² Score (Test)**: 0.7941 (79.41% variance explained)
- **R² Score (Train)**: 0.9531
- **Mean Absolute Error**: 16.08 customers
- **Cross-Validation Score**: 0.7842 (±0.0104)

### Top 5 Most Important Features
1. **Ad Spend** (70.59%) - Dominant factor
2. **Stock Level** (7.03%)
3. **Price Discount Ratio** (3.61%)
4. **Promotion Ad Spend** (3.14%)
5. **Price** (3.13%)


##  Key Insights

1. **Advertising is Critical**: Ad spend shows the strongest correlation (0.79) with footfall, accounting for over 70% of feature importance.

2. **Promotion Matters**: Promotion intensity (0.53 correlation) is the second most influential factor after advertising.

3. **Price Sensitivity**: Higher prices and competitor prices show negative correlations, suggesting price-conscious customer behavior.

4. **Model Performance**: Random Forest achieves excellent generalization with minimal overfitting (R² gap of 0.16 between train and test).

5. **Polynomial Regression Warning**: Higher-degree polynomials (4-5) show severe overfitting, making them unsuitable for this dataset.

##  Acknowledgments

- Dataset source:https://www.kaggle.com/datasets/mabubakrsiddiq/retail-store-product-sales-simulation-dataset/data
- Inspired by real-world retail analytics challenges
- Thanks to the open-source community for the amazing tools


⭐ **If you find this project helpful, please give it a star!** ⭐

# 📊 Click-Through Rate (CTR) Prediction for Email Campaigns

This project aims to predict the Click-Through Rate (CTR) of email marketing campaigns using a combination of numerical, categorical, and behavioral metadata features. Accurate CTR predictions help businesses optimize their content, targeting, and scheduling to boost engagement.

## 🧠 Problem Statement

Given campaign attributes like subject length, body content features, number of CTAs, presence of images, and personalization, the goal is to build a regression model that accurately predicts the `click_rate` (numerical target). A secondary task involves binarizing the target for classification (clicked vs not clicked).

## 🛠️ Methodology

- **EDA & Preprocessing**
  - Null imputation (mean/mode)
  - Feature binarization (e.g., `is_price`)
  - One-hot encoding for `times_of_day`
  - Normalization using `MinMaxScaler`

- **Feature Selection**
  - `SelectKBest` using `f_classif` and `f_regression`
  - Selected top features for training

- **Modeling**
  - Used **Random Forest** and **XGBoost Regressors**
  - Tuned hyperparameters with `GridSearchCV`
  - Cross-validated with 5-fold CV and `RepeatedKFold`

## 🧪 Evaluation

- **Random Forest Regressor**
  - MAE (Validation): **0.021**
  - R² (Validation): **0.534**
  - Accuracy (Binary CTR): **99.6%**

- **XGBoost Regressor**
  - R² (Train): **0.790**
  - R² (Validation): **0.291**

- Classification metrics reported using `precision`, `recall`, `F1-score`, and `classification_report`.

## 🔍 Key Insights

- Top predictive features include:
  - `no_of_CTA`, `is_discount`, `is_price`, `category`, `target_audience`
- Visualizations include:
  - Pair plots, heatmaps, feature importance charts, and k-value tuning plots.

## 💾 Deployment

- Final model saved using `pickle` for production use
- Supports easy inference for new campaign records

## 🧰 Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Seaborn, Matplotlib
- Pickle

---

### 📎 Example Output

- Predicted CTR values for unseen test data
- Classification: Binary predictions with ~99.6% accuracy

---

### 📌 Future Improvements

- Incorporate NLP for subject/body analysis
- Add time-based campaign effectiveness (hour-of-day patterns)
- Deploy model with a Streamlit interface


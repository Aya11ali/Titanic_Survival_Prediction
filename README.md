# 🛳️ Titanic Survival Prediction

This project aims to predict passenger survival on the Titanic using machine learning models and real-world data from Kaggle. The pipeline includes data preprocessing, exploratory data analysis (EDA), feature engineering, and model tuning using Random Forest with cross-validation.

---

## 🚀 Project Goals

- Clean and preprocess real-world tabular data.
- Perform exploratory data analysis to uncover survival patterns.
- Build a modular machine learning pipeline using `scikit-learn`.
- Tune hyperparameters using `GridSearchCV`.
- Generate predictions for unseen test data and prepare a submission file for Kaggle.

---

## 🧠 Skills Demonstrated

| Category | Tools / Techniques |
|---------|--------------------|
| Data Analysis | `pandas`, `numpy`, `seaborn`, `matplotlib`, `plotly` |
| Preprocessing | Custom Transformers (`BaseEstimator`, `TransformerMixin`), Imputation, One-Hot Encoding |
| ML Modeling | `RandomForestClassifier`, `GridSearchCV`, `StratifiedShuffleSplit` |
| Feature Engineering | Age imputation, Log-transform on fare, Categorical encoding |
| Pipeline Design | Clean, modular pipeline using `Pipeline` from `scikit-learn` |
| Evaluation | Cross-validation, accuracy score, train/test split |
| File Handling | Kaggle API integration, CSV import/export |

---

## 📊 Exploratory Data Analysis (EDA)

- Used histograms and grouped bar plots to explore:
  - Survival rate by age (younger children had higher survival).
  - Survival by passenger class (1st class more likely to survive).
  - Log-transformed fare values to reveal skewness.
- Detected and visualized missing values using heatmaps.
- Computed correlation heatmap to explore feature relationships.

---

## 🛠️ Preprocessing Pipeline

Created a custom `Pipeline` that includes:
- **Age Imputer**: fills missing `age` values using the mean.
- **Feature Encoder**: one-hot encodes `sex` and `embarkation_port`.
- **Feature Dropper**: removes unnecessary or redundant columns like `name`, `ticket_number`, `cabin`, etc.

This pipeline was applied consistently across training and test datasets to ensure reproducibility and avoid data leakage.

---

## 🔍 Model Training

- Model: `RandomForestClassifier`
- Used `GridSearchCV` to tune:
  - `n_estimators` (number of trees)
  - `max_depth` (tree depth)
  - `min_samples_split`
- Applied `StratifiedShuffleSplit` to maintain survival class balance in train/test sets.
- Final model trained on the full dataset after testing accuracy on held-out set.

---



## 📚 What This Project Taught Me

- ✅ How to clean and explore real-world, messy data.
- ✅ How to build and reuse robust ML pipelines.
- ✅ How to tune hyperparameters effectively using cross-validation.
- ✅ How to avoid data leakage during model evaluation.
- ✅ The importance of consistent preprocessing between train/test data.
- ✅ Experience working with Kaggle datasets and submission formats.

---

## 📁 Files

- `titanic_survival_prediction.py`: Full end-to-end code including EDA, pipeline, training, prediction.
- `train.csv` : The original training dataset downloaded from Kaggle. Contains features and the target variable
- `test.csv` : The test dataset provided by Kaggle for prediction. It includes the same features as the training data (except for Survived)
- `submission.csv`: Final predictions on test data for Kaggle. 
- `README.md`: Project documentation.


---

## 🏁 Conclusion

This project was a hands-on application of the machine learning pipeline on a famous dataset. It helped strengthen my data analysis, preprocessing, modeling, and evaluation skills. I now have a deeper understanding of what it takes to build a real-world predictive model from scratch.


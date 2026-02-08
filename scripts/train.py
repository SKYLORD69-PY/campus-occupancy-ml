import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ------------------------------------------------
# Paths
# ------------------------------------------------

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "generated_data" / "campus_occupancy_dataset.csv"
MODEL_PATH = BASE / "models"

MODEL_PATH.mkdir(exist_ok=True)

# ------------------------------------------------
# Load dataset
# ------------------------------------------------

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.upper()

# ------------------------------------------------
# Feature engineering
# ------------------------------------------------

df["UTILIZATION"] = df["ACTUAL_ATTENDANCE"] / df["VENUE_CAPACITY"]
df["OVERCROWDED"] = (df["UTILIZATION"] > 1).astype(int)

# ------------------------------------------------
# Feature selection
# ------------------------------------------------

features = [
    "SCHOOL","VENUE","WEEKDAY","TIME",
    "VENUE_CAPACITY","SEMESTER_DAY","EXAM_WEEK",
    "ENROLLED_STUDENTS","PREV_ATTENDANCE",
    "ROLLING_AVG_3","ROLLING_AVG_5","TREND_DELTA"
]

X = df[features]
y_reg = df["ACTUAL_ATTENDANCE"]
y_clf = df["OVERCROWDED"]

# ------------------------------------------------
# Preprocessor
# ------------------------------------------------

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     ["SCHOOL","VENUE","WEEKDAY","TIME"]),
    ("num","passthrough",
     ["VENUE_CAPACITY","SEMESTER_DAY","EXAM_WEEK",
      "ENROLLED_STUDENTS","PREV_ATTENDANCE",
      "ROLLING_AVG_3","ROLLING_AVG_5","TREND_DELTA"])
])

X_proc = preprocessor.fit_transform(X)

joblib.dump(preprocessor, MODEL_PATH / "preprocessor_v1.joblib")

# ------------------------------------------------
# Train/test split
# ------------------------------------------------

X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X_proc, y_reg, test_size=0.2, random_state=42
)

_, _, y_train_clf, y_test_clf = train_test_split(
    X_proc, y_clf, test_size=0.2, random_state=42
)

# ------------------------------------------------
# Linear Regression
# ------------------------------------------------

lin_model = LinearRegression()
lin_model.fit(X_train, y_train_reg)
lin_pred = lin_model.predict(X_test)

joblib.dump(lin_model, MODEL_PATH / "linear_model_v1.joblib")

# ------------------------------------------------
# Random Forest Regressor
# ------------------------------------------------

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train_reg)
rf_pred = rf_model.predict(X_test)

joblib.dump(rf_model, MODEL_PATH / "rf_model_v1.joblib")

# ------------------------------------------------
# SVM Classifier with SMOTE
# ------------------------------------------------

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train_clf)

svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_bal, y_train_bal)

probs = svm_model.predict_proba(X_test)[:,1]
svm_pred = (probs > 0.35).astype(int)

joblib.dump(svm_model, MODEL_PATH / "svm_classifier_v1.joblib")
joblib.dump((y_test_clf, svm_pred), MODEL_PATH / "svm_eval.joblib")

# ------------------------------------------------
# Metrics
# ------------------------------------------------

metrics = {
    "linear": {
        "MAE": mean_absolute_error(y_test_reg, lin_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test_reg, lin_pred)),
        "R2": r2_score(y_test_reg, lin_pred)
    },
    "rf": {
        "MAE": mean_absolute_error(y_test_reg, rf_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test_reg, rf_pred)),
        "R2": r2_score(y_test_reg, rf_pred)
    },
    "svm": {
        "Accuracy": accuracy_score(y_test_clf, svm_pred),
        "Report": classification_report(y_test_clf, svm_pred)
    }
}

joblib.dump(metrics, MODEL_PATH / "model_metrics.joblib")

# ------------------------------------------------
# Finish
# ------------------------------------------------

print("\n✅ Training complete")
print("Artifacts saved in /models")

import pandas as pd
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "models"

# ------------------------------------------------
# Load models + preprocessor
# ------------------------------------------------

lin_model = joblib.load(MODEL_PATH / "linear_model_v1.joblib")
rf_model = joblib.load(MODEL_PATH / "rf_model_v1.joblib")
svm_model = joblib.load(MODEL_PATH / "svm_classifier_v1.joblib")
preprocessor = joblib.load(MODEL_PATH / "preprocessor_v1.joblib")

# ------------------------------------------------
# Example input (edit freely)
# ------------------------------------------------

sample = {
    "SCHOOL": "VSST",
    "VENUE": "Activity Room",
    "WEEKDAY": "MONDAY",
    "TIME": "09:00",
    "VENUE_CAPACITY": 50,
    "SEMESTER_DAY": 10,
    "EXAM_WEEK": 0,
    "ENROLLED_STUDENTS": 40,
    "PREV_ATTENDANCE": 30,
    "ROLLING_AVG_3": 30,
    "ROLLING_AVG_5": 30,
    "TREND_DELTA": 0
}

input_df = pd.DataFrame([sample])

# ------------------------------------------------
# Transform
# ------------------------------------------------

X = preprocessor.transform(input_df)

# ------------------------------------------------
# Predict
# ------------------------------------------------

lin_pred = lin_model.predict(X)[0]
rf_pred = rf_model.predict(X)[0]
svm_pred = svm_model.predict(X)[0]

print("\n=== Prediction Results ===")
print(f"Linear Regression Attendance: {int(lin_pred)}")
print(f"Random Forest Attendance: {int(rf_pred)}")

if svm_pred == 1:
    print("⚠ Overcrowded Class")
else:
    print("✅ Safe Class")

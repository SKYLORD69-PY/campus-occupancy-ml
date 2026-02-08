import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import snowflake.connector
from pathlib import Path
import os

st.set_page_config(layout="wide")
st.title("🎓 Campus Occupancy Prediction Dashboard")

# ------------------------------------------------
# Paths
# ------------------------------------------------

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "models"

# ------------------------------------------------
# Snowflake connection
# ------------------------------------------------

@st.cache_data
def load_data():

    conn = snowflake.connector.connect(
        user=os.getenv("SNOW_USER"),
        password=os.getenv("SNOW_PASS"),
        account=os.getenv("SNOW_ACCOUNT"),
        warehouse=os.getenv("SNOW_WH"),
        database=os.getenv("SNOW_DB"),
        schema=os.getenv("SNOW_SCHEMA")
    )

    query = "SELECT * FROM CLASS_SESSIONS"
    df = pd.read_sql(query, conn)
    conn.close()

    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
    return df

df = load_data()

# ------------------------------------------------
# Feature engineering
# ------------------------------------------------

df["UTILIZATION"] = df["ACTUAL_ATTENDANCE"] / df["VENUE_CAPACITY"]
df["OVERCROWDED"] = (df["ACTUAL_ATTENDANCE"] > df["VENUE_CAPACITY"]).astype(int)
df["ENROLL_PRESSURE"] = df["ENROLLED_STUDENTS"] / df["VENUE_CAPACITY"]

# ------------------------------------------------
# Load models
# ------------------------------------------------

metrics = joblib.load(MODEL_PATH / "model_metrics.joblib")
y_test_clf, svm_pred = joblib.load(MODEL_PATH / "svm_eval.joblib")

lin_model = joblib.load(MODEL_PATH / "linear_model_v1.joblib")
rf_model = joblib.load(MODEL_PATH / "rf_model_v1.joblib")
svm_model = joblib.load(MODEL_PATH / "svm_classifier_v1.joblib")
preprocessor = joblib.load(MODEL_PATH / "preprocessor_v1.joblib")

tab1, tab2, tab3 = st.tabs(["📈 EDA", "📊 Model Reports", "🔮 Predictions"])

# =================================================
# TAB 1 — EDA
# =================================================

with tab1:

    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6,4))
        sns.histplot(df["UTILIZATION"], bins=20, kde=True, ax=ax1)
        ax1.axvline(1, color="red", linestyle="--")
        st.pyplot(fig1)

    with col2:
        venue_util = df.groupby("VENUE")["UTILIZATION"].mean()
        fig2, ax2 = plt.subplots(figsize=(6,6))
        sns.heatmap(venue_util.to_frame(), annot=True, cmap="Reds", ax=ax2)
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        school_risk = df.groupby("SCHOOL")["OVERCROWDED"].mean()
        fig3, ax3 = plt.subplots(figsize=(6,4))
        school_risk.sort_values().plot(kind="barh", ax=ax3)
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(6,4))
        sns.boxplot(x="OVERCROWDED", y="ENROLL_PRESSURE", data=df, ax=ax4)
        st.pyplot(fig4)

# =================================================
# TAB 2 — REPORTS
# =================================================

with tab2:

    st.header("Model Training Snapshot")

    st.subheader("Linear Regression")
    st.write(metrics["linear"])

    st.subheader("Random Forest")
    st.write(metrics["rf"])

    st.subheader("SVM Classification")
    st.write(f"Accuracy: {metrics['svm']['Accuracy']}")
    st.text(metrics["svm"]["Report"])

    cm = confusion_matrix(y_test_clf, svm_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)

# =================================================
# TAB 3 — PREDICTIONS
# =================================================

with tab3:

    st.header("Predict Class Safety & Attendance")

    venue_capacity_map = df.groupby("VENUE")["VENUE_CAPACITY"].first().to_dict()

    school = st.selectbox("School", sorted(df["SCHOOL"].unique()))
    venue = st.selectbox("Venue", sorted(df["VENUE"].unique()))
    weekday = st.selectbox("Weekday", ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY"])
    time = st.selectbox("Time", ["09:00","10:00","11:00","12:00","13:00"])

    capacity = venue_capacity_map[venue]
    enrolled = st.number_input("Enrolled Students", 1, 200, 40)

    input_df = pd.DataFrame({
        "SCHOOL":[school],
        "VENUE":[venue],
        "WEEKDAY":[weekday],
        "TIME":[time],
        "VENUE_CAPACITY":[capacity],
        "SEMESTER_DAY":[10],
        "EXAM_WEEK":[0],
        "ENROLLED_STUDENTS":[enrolled],
        "PREV_ATTENDANCE":[30],
        "ROLLING_AVG_3":[30],
        "ROLLING_AVG_5":[30],
        "TREND_DELTA":[0]
    })

    colA, colB = st.columns(2)

    with colA:
        if st.button("Predict Overcrowding"):
            X_input = preprocessor.transform(input_df)
            pred = svm_model.predict(X_input)[0]

            if pred == 1:
                st.error("⚠️ Overcrowded")
            else:
                st.success("✅ Safe")

    with colB:
        if st.button("Predict Attendance"):
            X_input = preprocessor.transform(input_df)
            attendance = lin_model.predict(X_input)[0]
            st.info(f"Predicted Attendance: {int(attendance)}")

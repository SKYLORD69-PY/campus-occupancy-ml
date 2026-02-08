# 🎓 Campus Occupancy Prediction System

A production-style machine learning pipeline for predicting classroom attendance and overcrowding in a university campus.

This project demonstrates **data engineering + ML deployment + dashboarding + model lifecycle management**, not just model training. It simulates how a real institutional analytics system would be built and maintained.

---

## 🚀 Project Goals

* Predict classroom attendance
* Detect overcrowded classes
* Build a reusable ML pipeline
* Store and ingest data via SQL (Snowflake-ready)
* Provide an interactive dashboard
* Maintain versioned models
* Demonstrate model lifecycle engineering

This project focuses on **engineering robustness**, not just accuracy.

---

## 🧠 Models Used

* Linear Regression → attendance prediction
* Random Forest Regressor → attendance prediction
* SVM Classifier → overcrowding detection

Additional techniques:

* SMOTE class balancing
* feature engineering
* evaluation via MAE / RMSE / R² / Accuracy / Classification Report

---

## 📊 Dashboard Features

The Streamlit dashboard includes:

* exploratory data analysis
* utilization visualization
* model performance reports
* confusion matrix
* live predictions
* attendance estimation
* overcrowding warning system

Run dashboard:

```
streamlit run dashboard/app.py
```

---

## 🏗 Repository Structure

```
campus-occupancy-ml/
│
├── dashboard/         # Streamlit UI
├── database/          # Snowflake ingestion scripts
├── scripts/           # data + training pipeline
├── models/            # model metadata & versioning   
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🔁 Reproducible Pipeline

Generate dataset:

```
python scripts/generate_data.py
```

Train models:

```
python scripts/train.py
```

Run predictions:

```
python scripts/predict.py
```

Run dashboard:

```
streamlit run dashboard/app.py
```

---

## 🧩 Model Versioning

Models are versioned using metadata JSON.

Binary model files are **not committed** to the repository to keep it lightweight and reproducible.

Instead:

* models are regenerated via training script
* metadata describes model version
* lifecycle is tracked through commits

This mirrors real-world ML deployment practices.

---

## 🗄 SQL Integration (Snowflake)

The project includes a Snowflake ingestion template:

```
database/snowflake_ingest.py
```

This demonstrates how synthetic campus data would be uploaded into a SQL warehouse for continuous retraining.

Credentials are intentionally excluded for security.

---

## 🎯 Engineering Focus

This project emphasizes:

✔ pipeline design
✔ maintainability
✔ version control
✔ deployment readiness
✔ reproducibility
✔ modular architecture
✔ model lifecycle management

Not just model performance.

---

## 🧪 Synthetic Dataset

The dataset is fully synthetic and reproducible via script.

It simulates:

* multiple schools
* venues with capacity constraints
* time slots
* attendance behavior
* overcrowding patterns

No real student data is used.

---

## 🛠 Requirements

Install dependencies:

```
pip install -r requirements.txt
```

---

## 👨‍💻 Author

YOUR NAME
University ML Pipeline Project
2026

---

## 📌 Academic Note

This repository is designed as a demonstration of **production ML engineering**:

> Notebooks are replaced by scripts
> models are versioned
> pipelines are reproducible
> dashboards use latest artifacts
> SQL integration is demonstrated
> Git history tracks lifecycle

This mirrors industry ML systems.

---

## ⭐ Final Outcome

A complete end-to-end ML system:

data → SQL → training → versioning → dashboard → predictions

Built like a real deployment pipeline.

---

If you’re reviewing this project:

Thank you 🙌
Feel free to explore the scripts and dashboard.

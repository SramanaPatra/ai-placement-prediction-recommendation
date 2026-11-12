# AI-Based Placement Prediction and Job Recommendation System

An end-to-end Machine Learning web application combining **Classification** (predicting student placement probability & risk status) and **Content-Based Recommendation** (matching students with company openings and identifying skill gaps).

---

## 🌟 Key Features

1. **AI Placement Predictor (Classification Engine)**:
   - Evaluates academic parameters (CGPA, 10th %, 12th %, Backlogs) and technical profile (Internships, Projects, Coding Test Score, Soft Skills, Certifications).
   - Generates interactive Plotly probability gauges, risk classification (`High Placement Probability`, `Moderate`, `At Risk`), and actionable recommendations.
   - Saves evaluated student records to the SQLite database.

2. **AI Job Recommender & Skill Gap Analysis**:
   - Matches candidate skills against active job listings using **TF-IDF Vectorization** and **Cosine Similarity**.
   - Evaluates eligibility criteria (Minimum CGPA cutoffs & maximum backlogs).
   - Visualizes **Matched Required Skills**, **Missing Skill Gaps**, and **Bonus Preferred Skills**.

3. **Company & Job Openings Database (DB Explorer)**:
   - Interactive database explorer with keyword search and salary filters.
   - Recruiter / TPO portal to post new company campus recruitment drives into SQLite in real-time.

4. **Model Performance & Explainable AI**:
   - Multi-model evaluation comparing **Random Forest**, **Gradient Boosting**, and **Logistic Regression**.
   - Cross-validation metrics: Accuracy (~85%), Precision (~87%), Recall (~92%), ROC-AUC (~0.93).
   - Interactive ROC Curve, Confusion Matrix heatmap, and Feature Importance bar charts.

5. **Batch Placement Evaluation (TPO Dashboard)**:
   - Upload bulk student CSVs or generate simulated batch evaluations.
   - Batch summary statistics (Placement rate, At-risk counts) and exportable CSV reports.

---

## 📁 Project Architecture

```
ai-placement-prediction-recommendation/
├── .streamlit/
│   └── config.toml                # Streamlit theme & server configuration
├── app.py                         # Interactive Streamlit Web Application
├── requirements.txt               # Runtime dependencies
├── data/
│   ├── placement_system.db        # SQLite Database (jobs, student_profiles)
│   └── placement_training_data.csv # 3,000 synthetic records with realistic correlations
├── models/
│   └── placement_model.joblib     # Serialized tuned Pipeline bundle and metrics
├── reports/                       # Offline tuning, scaler, and sensitivity artifacts
├── src/
│   ├── database.py                # SQLite schema, seeding, duplicate-safe drive inserts
│   ├── generate_dataset.py        # Ground truth dataset generator
│   ├── train_model.py             # Tuned Pipeline training, CV evaluation & predictor
│   ├── tune_and_analyze.py        # Hyperparameter search & sensitivity report generator
│   └── recommender.py             # Content-based recommendation & skill gap engine
└── tests/
    ├── test_database.py           # SQLite initialization, seeding & profile persistence tests
    ├── test_predictor.py          # Prediction bounds & recommendation rule tests
    └── test_recommender.py        # Eligibility cutoffs & skill-gap matching tests
```

---

## 🚀 Getting Started

### 1. Run the Streamlit Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 2. Retrain or Evaluate the Model Manually
```bash
python -c "from src.train_model import train_and_evaluate_models; train_and_evaluate_models()"
python src/train_model.py
```

---

## 🧪 Running Automated Tests

Install `pytest` (if not already installed in your virtual environment) and run the full test suite:

```bash
pip install pytest
pytest -v
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push the repository to GitHub (including `app.py`, `requirements.txt`, `.streamlit/config.toml`, `src/`, `models/`, and `reports/`).
2. Sign in to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository and branch, and set **Main file path** to `app.py` (or `ai-placement-prediction-recommendation/app.py` if deploying from the parent repository root).
4. Under **Advanced settings**, select **Python 3.9+** and confirm `requirements.txt` is detected automatically.
5. Click **Deploy** — the app uses `.streamlit/config.toml` defaults and automatically initializes `data/placement_system.db` on startup.

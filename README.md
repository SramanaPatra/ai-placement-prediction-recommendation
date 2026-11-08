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
placement_system/
├── app.py                     # Interactive Streamlit Web Application
├── requirements.txt           # Dependencies
├── data/
│   ├── placement_system.db    # SQLite Database (jobs, student_profiles)
│   └── placement_training_data.csv # 3,000 synthetic records with realistic correlations
├── models/
│   └── placement_model.joblib # Serialized champion model, scaler, and metrics
└── src/
    ├── database.py            # SQLite schema, initial seeding, and queries
    ├── generate_dataset.py    # Ground truth dataset generator
    ├── train_model.py         # Multi-model training, evaluation & predictor
    └── recommender.py         # Content-based recommendation & skill gap engine
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
python src/train_model.py
```

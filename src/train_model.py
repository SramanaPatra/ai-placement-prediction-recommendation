import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

from generate_dataset import generate_placement_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(MODELS_DIR, "placement_model.joblib")

FEATURE_COLUMNS = [
    "cgpa",
    "ssc_percentage",
    "hsc_percentage",
    "internships",
    "projects",
    "coding_score",
    "soft_skills_score",
    "certifications_count",
    "backlogs"
]

def load_or_create_data():
    csv_path = os.path.join(DATA_DIR, "placement_training_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        df = generate_placement_dataset(3000, random_state=42)
        df.to_csv(csv_path, index=False)
    return df

def train_and_evaluate_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_or_create_data()
    
    X = df[FEATURE_COLUMNS]
    y = df["placed"]
    
    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    def make_pipe(estimator):
        return Pipeline([("scale", StandardScaler()), ("clf", estimator)])

    candidate_models = {
        "Random Forest": make_pipe(RandomForestClassifier(
            n_estimators=235,
            max_depth=9,
            min_samples_split=2,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42,
        )),
        "Gradient Boosting": make_pipe(GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.061556316037040225,
            max_depth=2,
            min_samples_leaf=13,
            subsample=0.718509402281633,
            random_state=42,
        )),
        "Logistic Regression": make_pipe(LogisticRegression(
            C=0.05,
            penalty="l2",
            max_iter=2000,
            random_state=42,
        )),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_evaluations = {}
    
    best_model_name = None
    best_roc_auc = -1.0
    best_model = None
    
    print("=" * 60)
    print("TRAINING & EVALUATION OF PLACEMENT PREDICTION MODELS")
    print("=" * 60)
    
    for name, pipe in candidate_models.items():
        # Train pipeline (scaler fits strictly inside training data/folds)
        pipe.fit(X_train, y_train)
        
        # Predictions on held-out test set
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        # 5-Fold Cross Validation (Pipeline prevents scaler leakage across folds)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        model_evaluations[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "confusion_matrix": cm.tolist()
        }
        
        print(f"\nModel: {name}")
        print(f"  Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f} | 5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_model = pipe
            
    print("=" * 60)
    print(f"Champion Model Selected: {best_model_name} with Test ROC-AUC: {best_roc_auc:.4f}")
    print("=" * 60)
    
    # Feature importances from the champion classifier step
    best_clf = best_model.named_steps["clf"]
    if hasattr(best_clf, "feature_importances_"):
        raw_imp = best_clf.feature_importances_
    elif hasattr(best_clf, "coef_"):
        raw_imp = np.abs(best_clf.coef_[0])
    else:
        raw_imp = np.ones(len(FEATURE_COLUMNS))
        
    feat_imp = {col: float(val) for col, val in zip(FEATURE_COLUMNS, raw_imp)}
    # Sort by importance descending
    feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
    
    # ROC curve points for dashboard visualization
    best_probs = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, best_probs)
    
    model_bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "scaler": best_model.named_steps["scale"],
        "feature_names": FEATURE_COLUMNS,
        "metrics": model_evaluations[best_model_name],
        "all_model_evaluations": model_evaluations,
        "feature_importances": feat_imp,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        },
        "test_distribution": {
            "total_test_samples": len(y_test),
            "placed_count": int(sum(y_test)),
            "unplaced_count": int(len(y_test) - sum(y_test))
        }
    }
    
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"Successfully serialized model bundle to: {MODEL_PATH}")
    return model_bundle

class PlacementPredictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            print("Model artifact not found. Training model now...")
            self.bundle = train_and_evaluate_models()
        else:
            self.bundle = joblib.load(MODEL_PATH)
            
        self.model = self.bundle["model"]
        self.scaler = self.bundle["scaler"]
        self.feature_names = self.bundle["feature_names"]
        
    def predict(self, profile: dict):
        """
        Expects a dictionary with keys matching FEATURE_COLUMNS.
        Returns:
            probability: float (0.0 - 1.0)
            prediction: int (1 for Placed, 0 for Unplaced)
            status: str ('High Placement Probability', 'Moderate Probability', 'At Risk / Needs Improvement')
            recommendations: list[str] of actionable suggestions based on gaps
        """
        features_df = pd.DataFrame([{col: float(profile.get(col, 0)) for col in self.feature_names}])
        if isinstance(self.model, Pipeline):
            prob = float(self.model.predict_proba(features_df)[0][1])
            pred = int(self.model.predict(features_df)[0])
        else:
            scaled_features = self.scaler.transform(features_df)
            prob = float(self.model.predict_proba(scaled_features)[0][1])
            pred = int(self.model.predict(scaled_features)[0])
        
        if prob >= 0.75:
            status = "High Placement Probability"
            color = "green"
        elif prob >= 0.45:
            status = "Moderate Placement Probability"
            color = "orange"
        else:
            status = "At Risk - Immediate Preparation Recommended"
            color = "red"
            
        # Generate targeted actionable advice
        recommendations = []
        if profile.get("backlogs", 0) > 0:
            recommendations.append("Priority 1: Clear pending backlog(s). Most Tier-1 companies require zero active backlogs at time of interview.")
        if profile.get("coding_score", 0) < 65:
            recommendations.append("Boost Coding Score: Practice DSA on LeetCode/HackerRank (aim for 150+ solved problems across Arrays, Strings, Trees, and Dynamic Programming).")
        if profile.get("internships", 0) == 0:
            recommendations.append("Industrial Exposure: Undertake at least one industrial internship or contribute to open-source to strengthen your resume.")
        if profile.get("projects", 0) < 2:
            recommendations.append("Full-Stack / Core Projects: Build 2-3 end-to-end deployed projects showcasing modern frameworks and database design.")
        if profile.get("cgpa", 0) < 7.0:
            recommendations.append("Academic Standing: Aim to elevate current semester GPA to cross the standard 7.0 / 7.5 cutoff for multinational company eligibility.")
        if profile.get("soft_skills_score", 0) < 7:
            recommendations.append("Communication & Interview Skills: Participate in mock HR interviews, group discussions, and technical articulation practice.")
        if not recommendations:
            recommendations.append("Outstanding Profile: Maintain your current GPA, practice system design and behavioral STAR interview questions!")
            
        return {
            "placement_probability": prob,
            "prediction": pred,
            "status": status,
            "color": color,
            "recommendations": recommendations,
            "feature_breakdown": {col: profile.get(col, 0) for col in self.feature_names}
        }

if __name__ == "__main__":
    predictor = PlacementPredictor()
    test_sample = {
        "cgpa": 8.4,
        "ssc_percentage": 88.0,
        "hsc_percentage": 85.0,
        "internships": 2,
        "projects": 3,
        "coding_score": 85.0,
        "soft_skills_score": 8.0,
        "certifications_count": 2,
        "backlogs": 0
    }
    result = predictor.predict(test_sample)
    print("\nTest Prediction Result:")
    print(f"Probability: {result['placement_probability']:.2%}")
    print(f"Status: {result['status']}")
    print(f"Recommendations: {result['recommendations']}")

"""
Model optimization, sensitivity analysis and preprocessing experiments for the
placement prediction prototype.

"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from train_model import FEATURE_COLUMNS, load_or_create_data  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
REPORT_DIR = os.path.join(os.path.dirname(SRC_DIR), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def save_fig(name):
    path = os.path.join(REPORT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"  saved {path}")


# ----------------------------------------------------------------------
# 0. Data
# ----------------------------------------------------------------------
df = load_or_create_data()
X, y = df[FEATURE_COLUMNS], df["placed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"Data: {len(df)} rows | placed rate = {y.mean():.2%}")

# ----------------------------------------------------------------------
# 1. Correlation with the outcome + outlier report
# ----------------------------------------------------------------------
print("\n[1] Correlation of each feature with 'placed'")
corr = df.corr()["placed"].drop("placed").sort_values(key=np.abs, ascending=False)
print(corr.round(3).to_string())
corr.to_csv(os.path.join(REPORT_DIR, "feature_correlation.csv"), header=["corr_with_placed"])

plt.figure(figsize=(6, 4))
corr.sort_values().plot(kind="barh", color=["#EF4444" if v < 0 else "#3B82F6" for v in corr.sort_values()])
plt.title("Correlation with placement outcome")
save_fig("feature_correlation.png")

print("\n    Outliers per feature (IQR rule, 1.5 x IQR)")
rows = []
for col in FEATURE_COLUMNS:
    q1, q3 = X[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = int(((X[col] < q1 - 1.5 * iqr) | (X[col] > q3 + 1.5 * iqr)).sum())
    rows.append({"feature": col, "outliers": n_out, "pct": round(100 * n_out / len(X), 2)})
outlier_df = pd.DataFrame(rows)
print(outlier_df.to_string(index=False))
outlier_df.to_csv(os.path.join(REPORT_DIR, "outlier_report.csv"), index=False)

# ----------------------------------------------------------------------
# 2. Hyperparameter tuning (scaler lives INSIDE the pipeline so each CV
#    fold fits its own scaler: no leakage)
# ----------------------------------------------------------------------
print("\n[2] Hyperparameter search (scoring = ROC-AUC, 5-fold stratified CV)")


def make_pipe(model, scaler=None):
    return Pipeline([("scale", scaler if scaler is not None else StandardScaler()), ("clf", model)])


rf_space = {
    "clf__n_estimators": randint(100, 500),
    "clf__max_depth": randint(3, 16),
    "clf__min_samples_split": randint(2, 20),
    "clf__min_samples_leaf": randint(1, 15),
    "clf__max_features": ["sqrt", "log2", 0.5, 0.8],
}
gb_space = {
    "clf__n_estimators": randint(80, 400),
    "clf__learning_rate": uniform(0.01, 0.19),
    "clf__max_depth": randint(2, 6),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__min_samples_leaf": randint(1, 20),
}

searches = {}
for name, pipe, space in [
    ("Random Forest", make_pipe(RandomForestClassifier(random_state=SEED, n_jobs=-1)), rf_space),
    ("Gradient Boosting", make_pipe(GradientBoostingClassifier(random_state=SEED)), gb_space),
]:
    rs = RandomizedSearchCV(
        pipe, space, n_iter=25, scoring="roc_auc", cv=CV,
        random_state=SEED, n_jobs=-1, return_train_score=True,
    )
    rs.fit(X_train, y_train)
    searches[name] = rs
    print(f"  {name}: best CV AUC = {rs.best_score_:.4f}")
    print(f"    params: { {k.replace('clf__', ''): v for k, v in rs.best_params_.items()} }")
    pd.DataFrame(rs.cv_results_).sort_values("rank_test_score").to_csv(
        os.path.join(REPORT_DIR, f"random_search_{name.replace(' ', '_').lower()}.csv"), index=False
    )

# Grid search for the regularized linear baseline (C = inverse regularization strength)
lr_grid = GridSearchCV(
    make_pipe(LogisticRegression(max_iter=2000, random_state=SEED)),
    {"clf__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50], "clf__penalty": ["l2"]},
    scoring="roc_auc", cv=CV, n_jobs=-1,
)
lr_grid.fit(X_train, y_train)
searches["Logistic Regression"] = lr_grid
print(f"  Logistic Regression: best CV AUC = {lr_grid.best_score_:.4f} | C = {lr_grid.best_params_['clf__C']}")

# Tuned-vs-baseline on the held-out test set
baseline = {
    "Random Forest": make_pipe(RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=4, random_state=SEED)),
    "Gradient Boosting": make_pipe(GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=SEED)),
    "Logistic Regression": make_pipe(LogisticRegression(max_iter=1000, random_state=SEED)),
}
cmp_rows = []
for name, search in searches.items():
    base = baseline[name].fit(X_train, y_train)
    tuned = search.best_estimator_
    for tag, est in [("baseline", base), ("tuned", tuned)]:
        prob = est.predict_proba(X_test)[:, 1]
        cmp_rows.append({
            "model": name, "version": tag,
            "test_auc": roc_auc_score(y_test, prob),
            "test_acc": accuracy_score(y_test, (prob >= 0.5).astype(int)),
            "test_f1": f1_score(y_test, (prob >= 0.5).astype(int)),
        })
cmp_df = pd.DataFrame(cmp_rows).round(4)
print("\n  Baseline vs tuned (held-out test set):")
print(cmp_df.to_string(index=False))
cmp_df.to_csv(os.path.join(REPORT_DIR, "baseline_vs_tuned.csv"), index=False)

# ----------------------------------------------------------------------
# 3. Validation curves: train vs CV score (spot over/underfitting)
# ----------------------------------------------------------------------
print("\n[3] Validation curves")
curve_specs = [
    ("Random Forest", "clf__max_depth", range(2, 18, 2), "RF: max_depth"),
    ("Random Forest", "clf__n_estimators", [25, 50, 100, 200, 300, 500], "RF: n_estimators"),
    ("Gradient Boosting", "clf__n_estimators", [25, 50, 100, 200, 300, 400], "GB: n_estimators"),
    ("Gradient Boosting", "clf__learning_rate", [0.01, 0.03, 0.05, 0.08, 0.12, 0.2], "GB: learning_rate"),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (model_name, param, values, title) in zip(axes.ravel(), curve_specs):
    est = searches[model_name].best_estimator_
    values = list(values)
    tr, cv_s = validation_curve(
        est, X_train, y_train, param_name=param, param_range=values,
        cv=CV, scoring="roc_auc", n_jobs=-1,
    )
    ax.plot(values, tr.mean(axis=1), "o-", label="train", color="#10B981")
    ax.plot(values, cv_s.mean(axis=1), "o-", label="cross-val", color="#2563EB")
    ax.fill_between(values, cv_s.mean(axis=1) - cv_s.std(axis=1), cv_s.mean(axis=1) + cv_s.std(axis=1), alpha=0.15, color="#2563EB")
    ax.set_title(title)
    ax.set_ylabel("ROC-AUC")
    ax.legend()
save_fig("validation_curves.png")

# ----------------------------------------------------------------------
# 4. Scaler comparison (+ robustness to injected outliers)
# ----------------------------------------------------------------------
print("\n[4] Scaler comparison (5-fold CV ROC-AUC on training split)")
scalers = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler}
best_params = {
    n: {k.replace("clf__", ""): v for k, v in s.best_params_.items()} for n, s in searches.items()
}
model_factories = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=SEED, **{"C": best_params["Logistic Regression"]["C"]}),
    "Random Forest": lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1, **best_params["Random Forest"]),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=SEED, **best_params["Gradient Boosting"]),
}

# Corrupt 5% of training rows with extreme values (typos such as CGPA 90, coding score 900)
rng = np.random.default_rng(SEED)
X_dirty = X_train.copy()
idx = rng.choice(len(X_dirty), size=int(0.05 * len(X_dirty)), replace=False)
X_dirty.loc[X_dirty.index[idx], "cgpa"] *= 10
X_dirty.loc[X_dirty.index[idx], "coding_score"] *= 5

scaler_rows = []
for m_name, factory in model_factories.items():
    for s_name, S in scalers.items():
        clean = cross_val_score(make_pipe(factory(), S()), X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1)
        # Train on dirty data, evaluate on the CLEAN test set -> measures robustness
        dirty_est = make_pipe(factory(), S()).fit(X_dirty, y_train)
        dirty_auc = roc_auc_score(y_test, dirty_est.predict_proba(X_test)[:, 1])
        scaler_rows.append({
            "model": m_name, "scaler": s_name,
            "cv_auc_clean": clean.mean(), "cv_std": clean.std(),
            "test_auc_trained_on_5pct_outliers": dirty_auc,
        })
scaler_df = pd.DataFrame(scaler_rows).round(4)
print(scaler_df.to_string(index=False))
scaler_df.to_csv(os.path.join(REPORT_DIR, "scaler_comparison.csv"), index=False)

# ----------------------------------------------------------------------
# 5. Feature importance on the champion model
# ----------------------------------------------------------------------
champion_name = max(searches, key=lambda n: searches[n].best_score_)
champion = searches[champion_name].best_estimator_
print(f"\n[5] Champion by CV AUC: {champion_name}")

perm = permutation_importance(
    champion, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED, n_jobs=-1
)
perm_df = (
    pd.DataFrame({"feature": FEATURE_COLUMNS, "auc_drop": perm.importances_mean, "std": perm.importances_std})
    .sort_values("auc_drop", ascending=False)
)
print("  Permutation importance (drop in test AUC when the feature is shuffled):")
print(perm_df.round(4).to_string(index=False))
perm_df.to_csv(os.path.join(REPORT_DIR, "permutation_importance.csv"), index=False)

plt.figure(figsize=(6.5, 4.5))
p = perm_df.sort_values("auc_drop")
plt.barh(p["feature"], p["auc_drop"], xerr=p["std"], color="#3B82F6")
plt.title(f"Permutation importance ({champion_name})")
plt.xlabel("Drop in ROC-AUC")
save_fig("permutation_importance.png")

# ----------------------------------------------------------------------
# 6. Sensitivity / threshold analysis (one feature at a time, rest fixed)
# ----------------------------------------------------------------------
print("\n[6] Threshold analysis: when does a student cross into Moderate (45%) / High (75%)?")

PROFILES = {
    "At-risk (Rohan-like)": dict(cgpa=6.2, ssc_percentage=65, hsc_percentage=62, internships=0, projects=1,
                                 coding_score=42, soft_skills_score=5, certifications_count=0, backlogs=2),
    "Average (Rahul-like)": dict(cgpa=7.6, ssc_percentage=78, hsc_percentage=75, internships=1, projects=2,
                                 coding_score=68, soft_skills_score=7, certifications_count=1, backlogs=0),
    "Borderline": dict(cgpa=7.0, ssc_percentage=72, hsc_percentage=70, internships=1, projects=2,
                       coding_score=55, soft_skills_score=6, certifications_count=1, backlogs=0),
}
SWEEPS = {
    "coding_score": np.arange(20, 101, 1.0),
    "cgpa": np.arange(5.0, 10.01, 0.05),
    "backlogs": np.arange(0, 6, 1.0),
    "internships": np.arange(0, 5, 1.0),
    "projects": np.arange(0, 8, 1.0),
}
MODERATE, HIGH = 0.45, 0.75


def sweep(base, feature, values):
    frame = pd.DataFrame([{**base, feature: v} for v in values])[FEATURE_COLUMNS]
    return champion.predict_proba(frame)[:, 1]


def first_crossing(values, probs, thr):
    hit = np.where(probs >= thr)[0]
    return None if len(hit) == 0 else float(values[hit[0]])


threshold_rows = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for pname, base in PROFILES.items():
    for feat, vals in SWEEPS.items():
        probs = sweep(base, feat, vals)
        threshold_rows.append({
            "profile": pname, "feature": feat,
            "baseline_value": base[feat],
            "baseline_prob": round(float(champion.predict_proba(pd.DataFrame([base])[FEATURE_COLUMNS])[0][1]), 3),
            f"min_{feat}_for_Moderate(45%)": first_crossing(vals, probs, MODERATE),
            f"min_{feat}_for_High(75%)": first_crossing(vals, probs, HIGH),
            "prob_at_min_value": round(float(probs[0]), 3),
            "prob_at_max_value": round(float(probs[-1]), 3),
        })

for ax, feat in zip(axes, ["coding_score", "cgpa", "backlogs"]):
    for pname, base in PROFILES.items():
        ax.plot(SWEEPS[feat], sweep(base, feat, SWEEPS[feat]), label=pname)
    ax.axhline(MODERATE, ls="--", color="orange", lw=1)
    ax.axhline(HIGH, ls="--", color="green", lw=1)
    ax.set_xlabel(feat)
    ax.set_ylabel("P(placed)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Sensitivity to {feat}")
axes[0].legend(fontsize=8)
save_fig("sensitivity_curves.png")

# Print a compact threshold table for coding_score (the question asked) and save everything
thr_df = pd.DataFrame(threshold_rows)
thr_df.to_csv(os.path.join(REPORT_DIR, "threshold_analysis.csv"), index=False)
coding_view = thr_df[thr_df["feature"] == "coding_score"][
    ["profile", "baseline_value", "baseline_prob", "min_coding_score_for_Moderate(45%)", "min_coding_score_for_High(75%)"]
]
print(coding_view.to_string(index=False))
print("  (None = even the max value of that feature cannot reach the threshold alone;"
      " that student must improve several features together.)")

# Two-feature heatmap: coding_score x cgpa for a student with 0 backlogs, 1 internship, 2 projects
cg_vals = np.arange(5.5, 10.01, 0.25)
cs_vals = np.arange(20, 101, 5.0)
base = {**PROFILES["Average (Rahul-like)"]}
grid = np.zeros((len(cs_vals), len(cg_vals)))
for j, cg in enumerate(cg_vals):
    for i, cs in enumerate(cs_vals):
        row = pd.DataFrame([{**base, "cgpa": cg, "coding_score": cs}])[FEATURE_COLUMNS]
        grid[i, j] = champion.predict_proba(row)[0][1]
plt.figure(figsize=(8, 5.5))
plt.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
           extent=[cg_vals[0], cg_vals[-1], cs_vals[0], cs_vals[-1]])
plt.colorbar(label="P(placed)")
plt.contour(cg_vals, cs_vals, grid, levels=[MODERATE, HIGH], colors=["orange", "darkgreen"], linewidths=1.5)
plt.xlabel("CGPA")
plt.ylabel("Coding score")
plt.title("Safe-zone map: CGPA x Coding score (other features = average profile)")
save_fig("safe_zone_heatmap.png")

# ----------------------------------------------------------------------
# 7. Summary to paste into the PR description
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Champion (CV AUC): {champion_name} = {searches[champion_name].best_score_:.4f}")
print("Best params to plug into train_model.py:")
for n, p in best_params.items():
    print(f"  {n}: {p}")
best_scaler_row = scaler_df.sort_values("cv_auc_clean", ascending=False).iloc[0]
print(f"Top clean-CV scaler/model combo: {best_scaler_row['scaler']} + {best_scaler_row['model']}")
print(f"Reports written to: {REPORT_DIR}")
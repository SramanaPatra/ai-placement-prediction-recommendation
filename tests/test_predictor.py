import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from train_model import PlacementPredictor  # noqa: E402


@pytest.fixture(scope="module")
def predictor():
    return PlacementPredictor()


@pytest.mark.parametrize(
    "profile",
    [
        {
            "cgpa": 9.5,
            "ssc_percentage": 96.0,
            "hsc_percentage": 95.0,
            "internships": 3,
            "projects": 5,
            "coding_score": 95.0,
            "soft_skills_score": 9.5,
            "certifications_count": 4,
            "backlogs": 0,
        },
        {
            "cgpa": 7.0,
            "ssc_percentage": 72.0,
            "hsc_percentage": 70.0,
            "internships": 1,
            "projects": 2,
            "coding_score": 65.0,
            "soft_skills_score": 7.0,
            "certifications_count": 1,
            "backlogs": 0,
        },
        {
            "cgpa": 5.2,
            "ssc_percentage": 50.0,
            "hsc_percentage": 48.0,
            "internships": 0,
            "projects": 0,
            "coding_score": 20.0,
            "soft_skills_score": 3.0,
            "certifications_count": 0,
            "backlogs": 5,
        },
    ],
)
def test_predict_probability_bounds_and_schema(predictor, profile):
    result = predictor.predict(profile)
    assert 0.0 <= result["placement_probability"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert result["color"] in ("green", "orange", "red")
    assert isinstance(result["recommendations"], list) and len(result["recommendations"]) >= 1
    assert set(result["feature_breakdown"].keys()) == set(predictor.feature_names)


def test_strong_profile_scores_higher_than_at_risk(predictor):
    strong = {
        "cgpa": 9.1,
        "ssc_percentage": 92.0,
        "hsc_percentage": 90.0,
        "internships": 2,
        "projects": 4,
        "coding_score": 90.0,
        "soft_skills_score": 8.5,
        "certifications_count": 2,
        "backlogs": 0,
    }
    weak = {
        "cgpa": 5.8,
        "ssc_percentage": 60.0,
        "hsc_percentage": 58.0,
        "internships": 0,
        "projects": 0,
        "coding_score": 35.0,
        "soft_skills_score": 4.5,
        "certifications_count": 0,
        "backlogs": 3,
    }
    strong_res = predictor.predict(strong)
    weak_res = predictor.predict(weak)
    assert strong_res["placement_probability"] > weak_res["placement_probability"]
    assert strong_res["status"] == "High Placement Probability"
    assert weak_res["status"] == "At Risk - Immediate Preparation Recommended"


def test_recommendation_rules_triggered_accurately(predictor):
    deficient = {
        "cgpa": 6.4,
        "ssc_percentage": 70.0,
        "hsc_percentage": 68.0,
        "internships": 0,
        "projects": 1,
        "coding_score": 50.0,
        "soft_skills_score": 5.5,
        "certifications_count": 0,
        "backlogs": 2,
    }
    recs = " ".join(predictor.predict(deficient)["recommendations"]).lower()
    assert "backlog" in recs
    assert "coding score" in recs
    assert "internship" in recs
    assert "project" in recs
    assert "7.0 / 7.5" in recs
    assert "communication" in recs


def test_outstanding_profile_recommendation(predictor):
    ideal = {
        "cgpa": 8.8,
        "ssc_percentage": 90.0,
        "hsc_percentage": 89.0,
        "internships": 2,
        "projects": 4,
        "coding_score": 88.0,
        "soft_skills_score": 8.5,
        "certifications_count": 2,
        "backlogs": 0,
    }
    recs = predictor.predict(ideal)["recommendations"]
    assert len(recs) == 1
    assert "Outstanding Profile" in recs[0]


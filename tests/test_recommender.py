import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import database  # noqa: E402
from recommender import JobRecommender  # noqa: E402


@pytest.fixture
def temp_jobs_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test_recommender.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_file))
    database.init_db()
    database.add_job_drive(
        company_name="AlphaCorp",
        role_title="Backend Developer",
        department="Software Engineering",
        min_cgpa=7.5,
        max_backlogs=0,
        required_skills="Python, SQL, Docker",
        preferred_skills="Kubernetes, AWS",
        salary_lpa=18.0,
        location="Bengaluru",
        description="Build scalable backend APIs and services.",
    )
    database.add_job_drive(
        company_name="BetaTech",
        role_title="Associate Engineer",
        department="IT Services",
        min_cgpa=6.0,
        max_backlogs=2,
        required_skills="Java, SQL, Git",
        preferred_skills="React.js, HTML5",
        salary_lpa=6.5,
        location="Hyderabad",
        description="Maintain enterprise web applications.",
    )
    return db_file


def test_eligibility_cutoffs_cgpa_and_backlogs(temp_jobs_db):
    recommender = JobRecommender()

    # Student fails AlphaCorp on both CGPA (7.0 < 7.5) and backlogs (1 > 0), passes BetaTech
    recs = recommender.recommend_jobs(
        student_skills=["Python", "SQL", "Docker", "Java", "Git"],
        student_cgpa=7.0,
        student_backlogs=1,
        top_n=5,
    )
    assert len(recs) == 2

    by_company = {r["company_name"]: r for r in recs}
    assert by_company["BetaTech"]["is_eligible"] is True
    assert by_company["BetaTech"]["eligibility_status"] == "Eligible"

    assert by_company["AlphaCorp"]["is_eligible"] is False
    assert by_company["AlphaCorp"]["eligibility_status"] == "Ineligible"
    reasons = " ".join(by_company["AlphaCorp"]["eligibility_reasons"])
    assert "Min CGPA required: 7.5" in reasons
    assert "Max backlogs allowed: 0" in reasons

    # Eligible jobs must sort ahead of ineligible jobs
    assert recs[0]["company_name"] == "BetaTech"
    assert recs[1]["company_name"] == "AlphaCorp"


def test_skill_gap_and_normalization_accuracy(temp_jobs_db):
    recommender = JobRecommender()
    recs = recommender.recommend_jobs(
        student_skills=["Python", "SQL", "React"],
        student_cgpa=8.5,
        student_backlogs=0,
        top_n=5,
    )
    by_company = {r["company_name"]: r for r in recs}

    alpha = by_company["AlphaCorp"]
    assert set(alpha["matched_required_skills"]) == {"Python", "SQL"}
    assert set(alpha["missing_required_skills"]) == {"Docker"}
    assert alpha["matched_preferred_skills"] == []
    assert set(alpha["missing_preferred_skills"]) == {"Kubernetes", "AWS"}

    # Normalized match: 'React' matches 'React.js'
    beta = by_company["BetaTech"]
    assert "React.js" in beta["matched_preferred_skills"]
    assert "SQL" in beta["matched_required_skills"]
    assert set(beta["missing_required_skills"]) == {"Java", "Git"}


def test_top_n_and_domain_filter(temp_jobs_db):
    recommender = JobRecommender()
    recs = recommender.recommend_jobs(
        student_skills=["Python", "SQL"],
        student_cgpa=8.0,
        student_backlogs=0,
        top_n=1,
        filter_domain="Software Engineering",
    )
    assert len(recs) == 1
    assert recs[0]["company_name"] == "AlphaCorp"


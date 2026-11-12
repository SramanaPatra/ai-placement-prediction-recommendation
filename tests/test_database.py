import os
import sqlite3
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import database  # noqa: E402


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test_placement.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_file))
    return db_file


def test_init_db_creates_expected_tables(temp_db):
    database.init_db()
    assert temp_db.exists()

    conn = sqlite3.connect(str(temp_db))
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    conn.close()

    assert "jobs" in tables
    assert "student_profiles" in tables


def test_insert_initial_jobs_seeds_and_is_idempotent(temp_db):
    database.init_db()
    database.insert_initial_jobs()
    first_count = len(database.get_all_jobs())
    assert first_count == 15

    # Calling insert_initial_jobs again should not duplicate seed records
    database.insert_initial_jobs()
    second_count = len(database.get_all_jobs())
    assert second_count == first_count


def test_add_job_drive_inserts_and_prevents_duplicates(temp_db):
    database.init_db()
    inserted, job_id = database.add_job_drive(
        company_name="Adobe",
        role_title="Member of Technical Staff",
        department="Software Engineering",
        min_cgpa=7.5,
        max_backlogs=0,
        required_skills="Python, Data Structures, Algorithms",
        preferred_skills="Docker, AWS",
        salary_lpa=22.0,
        location="Noida",
        description="Core product engineering role.",
    )
    assert inserted is True
    assert job_id is not None

    # Same company & role with different casing/whitespace should be rejected
    dup_inserted, dup_id = database.add_job_drive(
        company_name="  adobe ",
        role_title="member of technical staff",
        department="Software Engineering",
        min_cgpa=8.0,
        max_backlogs=0,
        required_skills="Python",
        preferred_skills="",
        salary_lpa=25.0,
        location="Bengaluru",
        description="Duplicate check.",
    )
    assert dup_inserted is False
    assert dup_id is None
    assert len(database.get_all_jobs()) == 1


def test_save_and_get_recent_profiles(temp_db):
    database.init_db()
    p1 = database.save_student_profile(
        "Aarav Sharma",
        "aarav@college.edu",
        "Computer Science & Engg",
        8.8,
        92.0,
        90.0,
        2,
        4,
        88,
        8,
        0,
        "Python, SQL, Docker",
        0.9621,
        "High Placement Probability",
    )
    p2 = database.save_student_profile(
        "Rohan Verma",
        "rohan@college.edu",
        "Information Technology",
        6.1,
        66.0,
        64.0,
        0,
        1,
        42,
        5,
        2,
        "HTML5, CSS3",
        0.0312,
        "At Risk - Immediate Preparation Recommended",
    )
    assert p2 > p1

    recent = database.get_recent_profiles(limit=5)
    assert len(recent) == 2
    # Ordered newest first
    assert recent.iloc[0]["full_name"] == "Rohan Verma"
    assert recent.iloc[1]["full_name"] == "Aarav Sharma"
    assert pytest.approx(recent.iloc[0]["placement_probability"], rel=1e-4) == 0.0312


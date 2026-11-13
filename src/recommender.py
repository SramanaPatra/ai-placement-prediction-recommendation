import os
import sys
import re
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import get_all_jobs

# ---------------------------------------------------------------------------
# Change 1: Skill Synonym & Alias Dictionary
# Maps common abbreviations / shorthand -> canonical full-form skill name.
# normalize_skill() expands these before stripping punctuation so that
# "ML" and "Machine Learning" both resolve to "machine learning", making
# student input match job requirements regardless of which form is used.
# ---------------------------------------------------------------------------
SKILL_ALIASES: dict = {
    # Machine Learning / AI family
    "ml":           "machine learning",
    "ai":           "artificial intelligence",
    "dl":           "deep learning",
    "cv":           "computer vision",
    "nlp":          "natural language processing",

    # Data Structures & Algorithms
    "dsa":          "data structures",
    "ds":           "data structures",

    # JavaScript / TypeScript ecosystem
    "js":           "javascript",
    "ts":           "typescript",
    "nodejs":       "nodejs",
    "reactjs":      "react",
    "vuejs":        "vue",
    "angularjs":    "angular",

    # Databases
    "postgres":     "postgresql",
    "pg":           "postgresql",
    "mongo":        "mongodb",
    "mysql":        "sql",
    "sqlite":       "sql",

    # Programming paradigms
    "oop":          "object oriented programming",
    "oops":         "object oriented programming",
    "fp":           "functional programming",

    # Cloud / Infrastructure
    "k8s":          "kubernetes",
    "gcp":          "google cloud",
    "aws":          "amazon web services",

    # General abbreviations
    "db":           "database management",
    "os":           "operating systems",
    "cn":           "computer networks",
    "se":           "software engineering",
    "cs":           "computer science",
    "rest":         "rest apis",
    "api":          "rest apis",
}


def normalize_skill(skill: str) -> str:
    """Normalize a skill string for consistent matching.

    Steps:
    1. Strip whitespace and lowercase.
    2. Remove punctuation (., -, _) so variants like 'React.js' -> 'reactjs'.
    3. Expand known abbreviation aliases (e.g. 'ml' -> 'machine learning').
    """
    s = skill.strip().lower()
    # Strip punctuation first so alias lookup works (e.g. "node.js" -> "nodejs")
    s_clean = re.sub(r'[\.\-_]', '', s)
    # Expand alias if the cleaned form is a known abbreviation
    s_expanded = SKILL_ALIASES.get(s_clean, s_clean)
    return s_expanded


def parse_skills_list(skills_str: str) -> set:
    if not skills_str:
        return set()
    raw = [s.strip() for s in re.split(r'[,|;/]+', str(skills_str)) if s.strip()]
    return {s for s in raw}


class JobRecommender:
    """Content-based job recommender using TF-IDF cosine similarity.

    Change 2: The fitted TfidfVectorizer and job TF-IDF matrix are cached
    as instance attributes (_cached_vectorizer, _cached_job_matrix,
    _cached_job_texts, _cache_job_count). The cache is invalidated only when
    the number of jobs in the database changes (i.e. a new drive is added in
    Tab 3). This avoids re-fitting the vectorizer on every slider move or
    Streamlit rerun, making filtering near-instantaneous.
    """

    def __init__(self):
        self._vectorizer_template = TfidfVectorizer(
            token_pattern=r'(?u)\b\w+\b', ngram_range=(1, 2)
        )
        # Cache state
        self._cached_vectorizer = None   # fitted TfidfVectorizer instance
        self._cached_job_matrix = None   # sparse matrix: shape (n_jobs, vocab)
        self._cached_job_texts = []      # raw job text strings used for fitting
        self._cache_job_count = -1       # number of jobs when cache was built

    def _build_job_text(self, row) -> str:
        """Combine all relevant job fields into a single text document."""
        return (
            f"{row['role_title']} {row['department']} "
            f"{row['required_skills']} {row['preferred_skills']} "
            f"{row['description']}"
        )

    def _ensure_cache(self, df_jobs: pd.DataFrame):
        """Rebuild the job TF-IDF cache if stale or uninitialized.

        The cache is considered stale when the number of rows in df_jobs
        differs from when the cache was last built, which happens exactly
        when a new drive is posted in Tab 3.
        """
        current_count = len(df_jobs)
        if self._cached_job_matrix is not None and self._cache_job_count == current_count:
            return  # Cache is fresh — nothing to do

        # Refit vectorizer on job texts alone (no student document)
        job_texts = [self._build_job_text(row) for _, row in df_jobs.iterrows()]
        import sklearn.base
        new_vectorizer = sklearn.base.clone(self._vectorizer_template)
        try:
            job_matrix = new_vectorizer.fit_transform(job_texts)
        except ValueError:
            # Edge case: empty corpus
            job_matrix = None

        self._cached_vectorizer = new_vectorizer
        self._cached_job_matrix = job_matrix
        self._cached_job_texts = job_texts
        self._cache_job_count = current_count

    def recommend_jobs(
        self,
        student_skills: list,
        student_cgpa: float,
        student_backlogs: int = 0,
        top_n: int = 8,
        filter_domain: str = "All",
    ):
        """Calculate content-based recommendation scores, check eligibility,
        and identify skill gaps.

        Uses the cached TF-IDF job matrix when possible (Change 2) and
        expanded skill aliases for robust matching (Change 1).
        """
        df_jobs = get_all_jobs()
        if df_jobs.empty:
            return []

        # Ensure the full-corpus cache is up-to-date
        self._ensure_cache(df_jobs)

        # Apply domain filter AFTER caching (cache always covers all jobs)
        df_filtered = df_jobs
        if filter_domain and filter_domain != "All":
            df_filtered = df_jobs[
                df_jobs["department"].str.contains(filter_domain, case=False, na=False)
            ].copy()
            if df_filtered.empty:
                df_filtered = df_jobs  # Fallback if filter yields 0

        # Normalize student skills
        student_skills_clean = [s.strip() for s in student_skills if s.strip()]
        student_skills_str = " ".join(student_skills_clean)
        student_skills_set = {normalize_skill(s) for s in student_skills_clean}

        # Compute cosine similarities using cached vectorizer
        cosine_sims_all = {}  # job_index -> cosine similarity
        if self._cached_job_matrix is not None and self._cached_vectorizer is not None:
            try:
                # Transform only the student document (cheap single transform)
                student_vector = self._cached_vectorizer.transform([student_skills_str])
                # Build a mapping: row position in df_jobs -> cosine sim
                sims = cosine_similarity(student_vector, self._cached_job_matrix)[0]
                for pos, (orig_idx, _) in enumerate(df_jobs.iterrows()):
                    cosine_sims_all[orig_idx] = float(sims[pos])
            except Exception:
                for orig_idx in df_jobs.index:
                    cosine_sims_all[orig_idx] = 0.0
        else:
            for orig_idx in df_jobs.index:
                cosine_sims_all[orig_idx] = 0.0

        recommendations = []

        for orig_idx, row in df_filtered.iterrows():
            req_skills_raw = parse_skills_list(row["required_skills"])
            pref_skills_raw = parse_skills_list(row["preferred_skills"])

            # Match calculation using alias-expanded normalization
            matched_req = []
            missing_req = []
            for rs in req_skills_raw:
                norm = normalize_skill(rs)
                if any(norm in s or s in norm for s in student_skills_set):
                    matched_req.append(rs)
                else:
                    missing_req.append(rs)

            matched_pref = []
            missing_pref = []
            for ps in pref_skills_raw:
                norm = normalize_skill(ps)
                if any(norm in s or s in norm for s in student_skills_set):
                    matched_pref.append(ps)
                else:
                    missing_pref.append(ps)

            req_coverage = len(matched_req) / max(len(req_skills_raw), 1)
            pref_coverage = len(matched_pref) / max(len(pref_skills_raw), 1)

            # Eligibility check
            cgpa_eligible = student_cgpa >= row["min_cgpa"]
            backlog_eligible = student_backlogs <= row["max_backlogs"]
            is_eligible = cgpa_eligible and backlog_eligible

            # Composite Match Score: 45% TF-IDF + 45% Required Skills + 10% Preferred
            cos_sim = cosine_sims_all.get(orig_idx, 0.0)
            base_score = (0.45 * cos_sim) + (0.45 * req_coverage) + (0.10 * pref_coverage)
            match_percentage = round(min(base_score * 100, 99.0), 1)

            # Eligibility reasons
            eligibility_notes = []
            if not cgpa_eligible:
                eligibility_notes.append(
                    f"Min CGPA required: {row['min_cgpa']} (You have: {student_cgpa})"
                )
            if not backlog_eligible:
                eligibility_notes.append(
                    f"Max backlogs allowed: {row['max_backlogs']} (You have: {student_backlogs})"
                )
            if is_eligible:
                eligibility_notes.append("Fully Eligible")

            recommendations.append({
                "job_id":                   row["id"],
                "company_name":             row["company_name"],
                "role_title":               row["role_title"],
                "department":               row["department"],
                "salary_lpa":               row["salary_lpa"],
                "location":                 row["location"],
                "min_cgpa":                 row["min_cgpa"],
                "max_backlogs":             row["max_backlogs"],
                "match_percentage":         match_percentage,
                "is_eligible":              is_eligible,
                "eligibility_status":       "Eligible" if is_eligible else "Ineligible",
                "eligibility_reasons":      eligibility_notes,
                "matched_required_skills":  matched_req,
                "missing_required_skills":  missing_req,
                "matched_preferred_skills": matched_pref,
                "missing_preferred_skills": missing_pref,
                "description":              row["description"],
            })

        # Sort: Eligible first, then descending match_percentage
        recommendations.sort(
            key=lambda x: (x["is_eligible"], x["match_percentage"]), reverse=True
        )
        return recommendations[:top_n]


if __name__ == "__main__":
    from database import init_db, insert_initial_jobs
    init_db()
    insert_initial_jobs()

    recommender = JobRecommender()
    sample_skills = ["Python", "Data Structures", "Algorithms", "Machine Learning", "SQL", "Pandas"]
    res = recommender.recommend_jobs(sample_skills, student_cgpa=8.1, student_backlogs=0, top_n=5)
    print("Top Recommendations:")
    for r in res:
        print(
            f"[{r['match_percentage']}%] {r['company_name']} - {r['role_title']} "
            f"({r['salary_lpa']} LPA) | {r['eligibility_status']}"
        )
        print(f"  Matched: {r['matched_required_skills']} | Missing: {r['missing_required_skills']}")

    # Test alias expansion
    print("\nAlias tests:")
    for abbr, expected in [("ML", "machine learning"), ("DSA", "data structures"), ("JS", "javascript"), ("postgres", "postgresql"), ("OOP", "object oriented programming")]:
        got = normalize_skill(abbr)
        status = "✓" if got == expected else "✗"
        print(f"  {status} normalize_skill({abbr!r}) = {got!r} (expected {expected!r})")

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import get_all_jobs

def normalize_skill(skill: str) -> str:
    """Normalize skill string (e.g., 'React.js' -> 'react', 'Node.js' -> 'node')"""
    s = skill.strip().lower()
    s = re.sub(r'[\.\-_]', '', s)
    return s

def parse_skills_list(skills_str: str) -> set:
    if not skills_str:
        return set()
    raw = [s.strip() for s in re.split(r'[,|;/]+', str(skills_str)) if s.strip()]
    return {s for s in raw}

class JobRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=(1, 2))
        
    def recommend_jobs(self, student_skills: list, student_cgpa: float, student_backlogs: int = 0, top_n: int = 8, filter_domain: str = "All"):
        """
        Calculates content-based recommendation scores, checks eligibility,
        and identifies skill gaps.
        """
        df_jobs = get_all_jobs()
        if df_jobs.empty:
            return pd.DataFrame()
            
        if filter_domain and filter_domain != "All":
            df_jobs = df_jobs[df_jobs["department"].str.contains(filter_domain, case=False, na=False)].copy()
            if df_jobs.empty:
                df_jobs = get_all_jobs() # Fallback if filter yields 0
                
        # Normalize student skills string
        student_skills_clean = [s.strip() for s in student_skills if s.strip()]
        student_skills_str = " ".join(student_skills_clean)
        student_skills_set = {normalize_skill(s) for s in student_skills_clean}
        
        # Prepare job text corpus for TF-IDF
        job_texts = []
        for _, row in df_jobs.iterrows():
            combined_text = f"{row['role_title']} {row['department']} {row['required_skills']} {row['preferred_skills']} {row['description']}"
            job_texts.append(combined_text)
            
        # Fit-transform TF-IDF on jobs corpus + student text
        corpus = [student_skills_str] + job_texts
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            student_vector = tfidf_matrix[0:1]
            job_vectors = tfidf_matrix[1:]
            cosine_sims = cosine_similarity(student_vector, job_vectors)[0]
        except ValueError:
            # Fallback if empty vocabulary
            cosine_sims = np.zeros(len(df_jobs))
            
        recommendations = []
        
        for idx, (_, row) in enumerate(df_jobs.iterrows()):
            req_skills_raw = parse_skills_list(row["required_skills"])
            pref_skills_raw = parse_skills_list(row["preferred_skills"])
            
            # Match calculation
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
            
            # Composite Match Score: 45% TF-IDF Cosine Sim + 45% Required Skills Ratio + 10% Preferred Skills
            base_score = (0.45 * float(cosine_sims[idx])) + (0.45 * req_coverage) + (0.10 * pref_coverage)
            match_percentage = round(min(base_score * 100, 99.0), 1)
            
            # Eligibility reasons
            eligibility_notes = []
            if not cgpa_eligible:
                eligibility_notes.append(f"Min CGPA required: {row['min_cgpa']} (You have: {student_cgpa})")
            if not backlog_eligible:
                eligibility_notes.append(f"Max backlogs allowed: {row['max_backlogs']} (You have: {student_backlogs})")
            if is_eligible:
                eligibility_notes.append("Fully Eligible")
                
            recommendations.append({
                "job_id": row["id"],
                "company_name": row["company_name"],
                "role_title": row["role_title"],
                "department": row["department"],
                "salary_lpa": row["salary_lpa"],
                "location": row["location"],
                "min_cgpa": row["min_cgpa"],
                "max_backlogs": row["max_backlogs"],
                "match_percentage": match_percentage,
                "is_eligible": is_eligible,
                "eligibility_status": "Eligible" if is_eligible else "Ineligible",
                "eligibility_reasons": eligibility_notes,
                "matched_required_skills": matched_req,
                "missing_required_skills": missing_req,
                "matched_preferred_skills": matched_pref,
                "missing_preferred_skills": missing_pref,
                "description": row["description"]
            })
            
        # Sort recommendations: Eligible first, then descending match_percentage
        recommendations.sort(key=lambda x: (x["is_eligible"], x["match_percentage"]), reverse=True)
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
        print(f"[{r['match_percentage']}%] {r['company_name']} - {r['role_title']} ({r['salary_lpa']} LPA) | {r['eligibility_status']}")
        print(f"  Matched: {r['matched_required_skills']} | Missing: {r['missing_required_skills']}")

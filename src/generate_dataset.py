import os
import numpy as np
import pandas as pd

def generate_placement_dataset(num_samples=2500, random_state=42):
    """
    Generates a realistic student placement dataset with ground-truth correlations.
    Features:
    - cgpa: 5.0 to 10.0
    - ssc_percentage (10th): 50.0 to 98.0
    - hsc_percentage (12th): 50.0 to 98.0
    - internships: 0 to 4
    - projects: 0 to 6
    - coding_score: 20 to 100 (DSA / coding evaluation)
    - soft_skills_score: 1 to 10
    - certifications_count: 0 to 5
    - backlogs: 0 to 5
    - target: placed (0 or 1)
    """
    np.random.seed(random_state)
    
    # Generate realistic correlated features
    cgpa = np.clip(np.random.normal(7.3, 1.1, num_samples), 5.0, 10.0)
    
    # SSC and HSC correlate with CGPA
    ssc_percentage = np.clip(cgpa * 8.5 + np.random.normal(15, 6, num_samples), 50.0, 99.0)
    hsc_percentage = np.clip(cgpa * 8.2 + np.random.normal(16, 7, num_samples), 48.0, 98.5)
    
    # Internships: more likely for higher CGPA students
    internships_prob = np.clip((cgpa - 5.0) / 5.0, 0.1, 0.9)
    internships = np.random.binomial(3, internships_prob)
    
    # Projects count
    projects = np.clip(np.random.poisson(lam=2.3, size=num_samples), 0, 7)
    
    # Coding test score (strongly predictive of modern placements)
    coding_base = (cgpa - 5.0) * 10 + (projects * 4) + (internships * 5)
    coding_score = np.clip(coding_base + np.random.normal(40, 12, num_samples), 20.0, 100.0)
    
    # Soft skills score (1 to 10)
    soft_skills = np.clip(np.random.normal(6.5, 1.8, num_samples), 1.0, 10.0)
    
    # Certifications
    certifications = np.random.binomial(4, 0.35, num_samples)
    
    # Backlogs: inversely related to CGPA
    backlog_prob = np.clip((9.5 - cgpa) / 10.0, 0.05, 0.8)
    backlogs = np.random.binomial(4, backlog_prob)
    
    # Log-odds formulation for realistic ground truth placement probability
    # Strong positive factors: CGPA, Coding Score, Internships, Projects, Soft skills
    # Strong negative factors: Backlogs
    z = (
        1.35 * (cgpa - 7.0) +
        0.045 * (coding_score - 60.0) +
        0.55 * internships +
        0.30 * projects +
        0.25 * (soft_skills - 6.0) +
        0.15 * certifications +
        0.02 * (ssc_percentage - 70.0) +
        0.02 * (hsc_percentage - 70.0) -
        0.85 * backlogs -
        0.65  # intercept
    )
    
    prob = 1 / (1 + np.exp(-z))
    placed = (np.random.rand(num_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        "cgpa": np.round(cgpa, 2),
        "ssc_percentage": np.round(ssc_percentage, 1),
        "hsc_percentage": np.round(hsc_percentage, 1),
        "internships": internships,
        "projects": projects,
        "coding_score": np.round(coding_score, 1),
        "soft_skills_score": np.round(soft_skills, 1),
        "certifications_count": certifications,
        "backlogs": backlogs,
        "placed": placed
    })
    
    return df

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "placement_training_data.csv")
    
    df = generate_placement_dataset(3000, random_state=42)
    df.to_csv(csv_path, index=False)
    print(f"Generated placement dataset with {len(df)} records saved to {csv_path}")
    print(f"Placement Distribution:\n{df['placed'].value_counts(normalize=True)}")

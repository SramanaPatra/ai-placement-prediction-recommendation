import os
import sys
import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src to python path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from database import init_db, insert_initial_jobs, get_all_jobs, save_student_profile, get_recent_profiles, get_connection
from train_model import PlacementPredictor, FEATURE_COLUMNS
from recommender import JobRecommender
from report_generator import generate_student_pdf

# Set page configuration
st.set_page_config(
    page_title="AI Placement Predictor & Job Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern responsive design (Desktop & Tablet)
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .card:hover {
        box-shadow: 0 8px 12px -2px rgba(0, 0, 0, 0.08);
    }
    .badge-eligible {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .badge-ineligible {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .tag-matched {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        display: inline-block;
        margin: 2px;
    }
    .tag-missing {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        display: inline-block;
        margin: 2px;
    }
    .tag-pref {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        display: inline-block;
        margin: 2px;
    }
    /* Responsive media queries for Desktop and Tablet */
    @media screen and (max-width: 992px) {
        .main-title {
            font-size: 1.85rem !important;
        }
        .subtitle {
            font-size: 0.95rem !important;
            margin-bottom: 1rem !important;
        }
        .card {
            padding: 14px 16px !important;
            margin-bottom: 0.85rem !important;
        }
        div[data-testid="column"] {
            min-width: 260px !important;
            flex: 1 1 260px !important;
        }
    }
    @media screen and (max-width: 768px) {
        .main-title {
            font-size: 1.5rem !important;
        }
        .subtitle {
            font-size: 0.88rem !important;
        }
        .card {
            padding: 12px 14px !important;
        }
        div[data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .tag-matched, .tag-missing, .tag-pref {
            font-size: 0.72rem !important;
            padding: 2px 6px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize database and model instances
@st.cache_resource
def load_system():
    init_db()
    insert_initial_jobs()
    predictor = PlacementPredictor()
    recommender = JobRecommender()
    return predictor, recommender

predictor, recommender = load_system()

# Popular skill list for multi-select
ALL_SKILLS = [
    "Python", "Java", "C++", "C#", "C", "Golang", "JavaScript", "TypeScript",
    "Data Structures", "Algorithms", "System Design", "Problem Solving",
    "SQL", "PostgreSQL", "MongoDB", "Redis", "Database Management",
    "React", "Node.js", "Next.js", "HTML5", "CSS3", "Redux", "Tailwind CSS",
    "Machine Learning", "Deep Learning", "Pandas", "Scikit-Learn", "TensorFlow", "PyTorch", "NLP",
    "AWS", "Azure", "Docker", "Kubernetes", "Linux", "Git", "REST APIs", "Microservices",
    "Computer Networks", "Cybersecurity", "Statistics", "Power BI", "Tableau", "Communication Skills"
]

# Sidebar
st.sidebar.markdown("### 🎓 Student Profile Setup")
preset = st.sidebar.selectbox(
    "Load Profile Preset",
    ["Custom Input", "🌟 High Achiever (Software)", "🤖 Data Science Enthusiast", "⚠️ Student Needing Guidance"]
)

# Set defaults based on preset
if preset == "🌟 High Achiever (Software)":
    default_name = "Aarav Sharma"
    default_cgpa = 8.8
    default_ssc = 92.0
    default_hsc = 89.5
    default_internships = 2
    default_projects = 4
    default_coding = 88.0
    default_soft = 8.5
    default_certifications = 3
    default_github = 12
    default_hackathons = 2
    default_backlogs = 0
    default_skills = ["Python", "Java", "Data Structures", "Algorithms", "System Design", "SQL", "Docker", "Git"]
elif preset == "🤖 Data Science Enthusiast":
    default_name = "Priya Patel"
    default_cgpa = 8.1
    default_ssc = 86.0
    default_hsc = 84.0
    default_internships = 1
    default_projects = 3
    default_coding = 76.0
    default_soft = 7.5
    default_certifications = 3
    default_github = 8
    default_hackathons = 1
    default_backlogs = 0
    default_skills = ["Python", "Machine Learning", "Pandas", "Scikit-Learn", "SQL", "Statistics", "Data Structures"]
elif preset == "⚠️ Student Needing Guidance":
    default_name = "Rohan Verma"
    default_cgpa = 6.2
    default_ssc = 65.0
    default_hsc = 62.0
    default_internships = 0
    default_projects = 1
    default_coding = 42.0
    default_soft = 5.0
    default_certifications = 0
    default_github = 1
    default_hackathons = 0
    default_backlogs = 2
    default_skills = ["C++", "HTML5", "CSS3"]
else:
    default_name = "Rahul Gupta"
    default_cgpa = 7.6
    default_ssc = 78.0
    default_hsc = 75.0
    default_internships = 1
    default_projects = 2
    default_coding = 68.0
    default_soft = 7.0
    default_certifications = 1
    default_github = 5
    default_hackathons = 1
    default_backlogs = 0
    default_skills = ["Java", "Python", "SQL", "Data Structures", "Git"]

# App Navigation
tabs = st.tabs([
    "🎯 Placement Prediction",
    "💼 Job Recommendation",
    "🏢 Company Database",
    "📊 Model Analytics",
    "👥 Batch Evaluation"
])

# -------------------------------------------------------------
# TAB 1: PLACEMENT PREDICTION
# -------------------------------------------------------------
with tabs[0]:
    st.markdown("<div class='main-title'>AI-Based Placement Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Evaluates academic credentials, practical projects, coding acumen, and experience to compute placement readiness.</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1.0])
    
    with col1:
        st.markdown("#### 📝 Candidate Assessment Form")
        with st.form("placement_form"):
            name = st.text_input("Full Name", value=default_name)
            branch = st.selectbox("Engineering Branch", ["Computer Science & Engg", "Information Technology", "AI & Data Science", "Electronics & Comm.", "Mechanical Engg", "Civil Engg"])
            
            c_a, c_b = st.columns(2)
            with c_a:
                cgpa = st.slider("Current CGPA (out of 10.0)", min_value=5.0, max_value=10.0, value=float(default_cgpa), step=0.05)
                ssc = st.number_input("10th Standard Marks (%)", min_value=40.0, max_value=100.0, value=float(default_ssc), step=0.5)
                internships = st.number_input("Internships Completed", min_value=0, max_value=5, value=int(default_internships))
                backlogs = st.number_input("Current Backlogs / Arrears", min_value=0, max_value=6, value=int(default_backlogs))
            with c_b:
                coding_score = st.slider("Coding / DSA Assessment Score (0 - 100)", min_value=10.0, max_value=100.0, value=float(default_coding), step=1.0)
                hsc = st.number_input("12th Standard Marks (%)", min_value=40.0, max_value=100.0, value=float(default_hsc), step=0.5)
                projects = st.number_input("Major / Minor Projects", min_value=0, max_value=8, value=int(default_projects))
                soft_skills = st.slider("Soft Skills & Interview Rating (1 - 10)", min_value=1.0, max_value=10.0, value=float(default_soft), step=0.5)
            
            st.markdown("##### 🏆 Portfolio & Extracurricular Excellence")
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                certifications = st.slider("📜 Technical Certifications", min_value=0, max_value=10, value=int(default_certifications), step=1, help="Industry certifications (AWS, Azure, GCP, Oracle Java, etc.)")
            with p_col2:
                github_repos = st.slider("🐙 GitHub Repositories", min_value=0, max_value=30, value=int(default_github), step=1, help="Public GitHub repositories showcasing project code and commit history")
            with p_col3:
                hackathons = st.slider("🥇 Hackathon Wins / Podiums", min_value=0, max_value=8, value=int(default_hackathons), step=1, help="Hackathon victories, finalist ranks, or podium finishes")
            
            skills_selected = st.multiselect(
                "Technical Skills & Tooling",
                options=ALL_SKILLS,
                default=[s for s in default_skills if s in ALL_SKILLS]
            )
            
            submit_btn = st.form_submit_button("🚀 Run AI Placement Prediction", use_container_width=True)
            
    # Process prediction
    profile_data = {
        "cgpa": cgpa,
        "ssc_percentage": ssc,
        "hsc_percentage": hsc,
        "internships": internships,
        "projects": projects,
        "coding_score": coding_score,
        "soft_skills_score": soft_skills,
        "certifications_count": certifications,
        "github_repos": github_repos,
        "hackathons_count": hackathons,
        "backlogs": backlogs
    }
    
    result = predictor.predict(profile_data)
    prob = result["placement_probability"]
    prob_percent = prob * 100
    status = result["status"]
    
    # Compute candidate recommended jobs for PDF report and recommendations
    candidate_recommended_jobs = recommender.recommend_jobs(
        student_skills=skills_selected,
        student_cgpa=cgpa,
        student_backlogs=backlogs,
        top_n=5
    )
    
    # Generate downloadable PDF report
    pdf_report_bytes = generate_student_pdf(
        student_name=name,
        branch=branch,
        profile_data=profile_data,
        skills_selected=skills_selected,
        prediction_result=result,
        recommended_jobs=candidate_recommended_jobs
    )
    
    with col2:
        st.markdown("#### 🎯 Placement Prediction Output")
        
        # Plotly Gauge Chart
        gauge_color = "#10B981" if prob >= 0.75 else ("#F59E0B" if prob >= 0.45 else "#EF4444")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Placement Readiness Score", 'font': {'size': 20, 'color': '#1E293B'}},
            number={'suffix': "%", 'font': {'size': 38, 'color': gauge_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                'bar': {'color': gauge_color},
                'bgcolor': "#F1F5F9",
                'borderwidth': 1,
                'bordercolor': "#E2E8F0",
                'steps': [
                    {'range': [0, 45], 'color': '#FEE2E2'},
                    {'range': [45, 75], 'color': '#FEF3C7'},
                    {'range': [75, 100], 'color': '#DCFCE7'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': prob_percent
                }
            }
        ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
        # Status Box
        if prob >= 0.75:
            st.success(f"**Status: {status}**\n\nThe candidate meets high placement probability benchmarks across academics, technical, and portfolio evaluation.")
        elif prob >= 0.45:
            st.warning(f"**Status: {status}**\n\nThe candidate has competitive baseline metrics but needs strengthening in key evaluated parameters.")
        else:
            st.error(f"**Status: {status}**\n\nHigh risk of campus placement rejection. Follow targeted guidance below to clear eligibility thresholds.")
            
        # Key Recommendations
        st.markdown("##### 📌 AI-Powered Personalized Recommendations:")
        for rec in result["recommendations"]:
            st.markdown(f"- {rec}")
            
        # Downloadable PDF Report Button
        st.markdown("---")
        st.markdown("##### 📄 Export Student Placement Report")
        st.download_button(
            label="📥 Download Placement Readiness & Job Report (PDF)",
            data=pdf_report_bytes,
            file_name=f"{name.replace(' ', '_')}_Placement_Readiness_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Download formal PDF report with placement readiness score, competency metrics, AI guidance, and matching jobs."
        )
            
        # Save to DB option
        if submit_btn:
            skills_str = ", ".join(skills_selected)
            saved_id = save_student_profile(
                name, f"{name.lower().replace(' ', '')}@college.edu", branch,
                cgpa, ssc, hsc, internships, projects, int(coding_score),
                int(soft_skills), backlogs, skills_str, round(prob, 4), status,
                certifications=int(certifications),
                github_repos=int(github_repos),
                hackathons=int(hackathons)
            )
            st.toast(f"✅ Assessment saved to database (Profile ID: {saved_id})")

# -------------------------------------------------------------
# TAB 2: AI JOB RECOMMENDATION & SKILL MATCHING
# -------------------------------------------------------------
with tabs[1]:
    st.markdown("<div class='main-title'>AI Job Recommender & Skill Gap Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Matches student profile against company openings using TF-IDF Content-Based Filtering, eligibility checking, and skill gap identification.</div>", unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns([1.5, 1, 1])
    with f1:
        rec_skills = st.multiselect(
            "Filter / Adjust Candidate Skills",
            options=ALL_SKILLS,
            default=[s for s in skills_selected if s in ALL_SKILLS] if skills_selected else ["Python", "SQL", "Data Structures", "Algorithms"]
        )
    with f2:
        domain_filter = st.selectbox(
            "Target Department / Domain",
            ["All", "Software Engineering", "Artificial Intelligence", "Cloud / Backend", "FinTech", "Security & Risk", "IT Services"]
        )
    with f3:
        top_k = st.slider("Max Job Recommendations", min_value=3, max_value=12, value=6)
        
    recommended_jobs = recommender.recommend_jobs(
        student_skills=rec_skills,
        student_cgpa=cgpa,
        student_backlogs=backlogs,
        top_n=top_k,
        filter_domain=domain_filter
    )
    
    if not recommended_jobs:
        st.info("No matching jobs found for the selected criteria.")
    else:
        st.markdown(f"### 🎯 Found {len(recommended_jobs)} Recommended Roles")
        
        for job in recommended_jobs:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                
                # Card Header
                h_col1, h_col2 = st.columns([2.5, 1])
                with h_col1:
                    st.markdown(f"### {job['role_title']} &nbsp;•&nbsp; <span style='color:#2563EB;'>{job['company_name']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Domain:** {job['department']} &nbsp;|&nbsp; **Location:** {job['location']} &nbsp;|&nbsp; **CTC:** `₹ {job['salary_lpa']} LPA`")
                with h_col2:
                    score = job['match_percentage']
                    badge_class = "badge-eligible" if job['is_eligible'] else "badge-ineligible"
                    st.markdown(f"<div style='text-align:right;'><span class='{badge_class}'>{job['eligibility_status']}</span></div>", unsafe_allow_html=True)
                    st.progress(score / 100.0)
                    st.markdown(f"<div style='text-align:right; font-weight:700; color:#1E3A8A;'>{score}% Profile Match</div>", unsafe_allow_html=True)
                
                # Skill badges
                st.markdown("---")
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    st.markdown("**✅ Matched Required Skills:**")
                    if job['matched_required_skills']:
                        tags_html = " ".join([f"<span class='tag-matched'>{s}</span>" for s in job['matched_required_skills']])
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.caption("No direct required skill matches.")
                
                with s_col2:
                    st.markdown("**⚠️ Skill Gap (Missing Required):**")
                    if job['missing_required_skills']:
                        tags_html = " ".join([f"<span class='tag-missing'>{s}</span>" for s in job['missing_required_skills']])
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:#10B981; font-weight:600;'>🎉 Full Required Skills Match!</span>", unsafe_allow_html=True)
                        
                if job['missing_preferred_skills']:
                    st.markdown("**💡 Bonus Skills to Boost Profile:**")
                    tags_pref = " ".join([f"<span class='tag-pref'>{s}</span>" for s in job['missing_preferred_skills']])
                    st.markdown(tags_pref, unsafe_allow_html=True)
                    
                # Eligibility details
                if not job['is_eligible']:
                    st.markdown(f"<div style='color:#B91C1C; font-size:0.85rem; margin-top:6px;'>⚠️ <b>Eligibility Notice:</b> {' | '.join(job['eligibility_reasons'])}</div>", unsafe_allow_html=True)
                
                with st.expander("📄 View Role Description"):
                    st.write(job['description'])
                    st.write(f"**Eligibility Criteria:** Minimum CGPA: `{job['min_cgpa']}`, Max Backlogs: `{job['max_backlogs']}`")
                    
                st.markdown("</div>", unsafe_allow_html=True)

        # Downloadable PDF report in Tab 2
        st.markdown("---")
        t2_pdf_bytes = generate_student_pdf(
            student_name=name,
            branch=branch,
            profile_data=profile_data,
            skills_selected=rec_skills,
            prediction_result=result,
            recommended_jobs=recommended_jobs
        )
        st.download_button(
            label="📥 Download Full Student Placement & Job Recommendation Report (PDF)",
            data=t2_pdf_bytes,
            file_name=f"{name.replace(' ', '_')}_Job_Recommendations_Report.pdf",
            mime="application/pdf",
            key="download_tab2_pdf",
            help="Download formal PDF report with complete skill gap analysis and recommended job matches."
        )

# -------------------------------------------------------------
# TAB 3: COMPANY & JOB OPENINGS DATABASE (DB)
# -------------------------------------------------------------
with tabs[2]:
    st.markdown("<div class='main-title'>Company & Job Openings Database</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Manage and explore real-time company recruitment drives stored in the SQLite database.</div>", unsafe_allow_html=True)
    
    df_all_jobs = get_all_jobs()
    
    st.markdown(f"#### 📋 Current Active Openings ({len(df_all_jobs)} Companies)")
    
    d1, d2 = st.columns([2, 1])
    with d1:
        search_query = st.text_input("🔍 Search by Company, Role, or Skill", placeholder="e.g. Google, Machine Learning, Java")
    with d2:
        min_salary_filter = st.slider("Minimum CTC (LPA)", 3.0, 30.0, 3.0, step=1.0)
        
    filtered_db_jobs = df_all_jobs[df_all_jobs["salary_lpa"] >= min_salary_filter]
    if search_query:
        mask = (
            filtered_db_jobs["company_name"].str.contains(search_query, case=False, na=False) |
            filtered_db_jobs["role_title"].str.contains(search_query, case=False, na=False) |
            filtered_db_jobs["required_skills"].str.contains(search_query, case=False, na=False) |
            filtered_db_jobs["department"].str.contains(search_query, case=False, na=False)
        )
        filtered_db_jobs = filtered_db_jobs[mask]
        
    st.dataframe(
        filtered_db_jobs[["id", "company_name", "role_title", "department", "salary_lpa", "min_cgpa", "max_backlogs", "location", "required_skills"]],
        use_container_width=True,
        hide_index=True
    )
    
    # Form to insert a new job posting
    with st.expander("➕ Post New Company Job Drive (Recruiter / TPO Portal)", expanded=False):
        st.markdown("""
        <div style='background-color:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px 16px; margin-bottom:12px; font-size:0.9rem; color:#475569;'>
            <b>Recruiter Portal:</b> Publish new campus placement drives directly to the database. All drives become immediately available for student recommendation and filtering.
        </div>
        """, unsafe_allow_html=True)
        with st.form("new_job_form"):
            c1, c2 = st.columns(2)
            with c1:
                new_company = st.text_input("Company Name *", placeholder="e.g. Adobe", help="Official legal or recruiting entity name")
                new_role = st.text_input("Role Title *", placeholder="e.g. Member of Technical Staff", help="Job designation / position title")
                new_dept = st.selectbox("Department / Domain *", ["Software Engineering", "Artificial Intelligence", "Cloud / Backend", "FinTech", "Security & Risk", "IT Services", "Analytics & Growth"])
                new_salary = st.number_input("CTC Package (LPA) *", min_value=1.0, max_value=120.0, value=18.0, step=0.5, help="Annual cost-to-company in Lakhs Per Annum")
            with c2:
                new_min_cgpa = st.slider("Minimum CGPA Cutoff", 5.0, 10.0, 7.5, 0.1, help="Academic cutoff required for initial screening")
                new_max_backlogs = st.number_input("Max Active Backlogs Allowed", min_value=0, max_value=10, value=0, help="Maximum number of unresolved arrears allowed")
                new_location = st.text_input("Job Location", value="Bengaluru / Remote")
                new_req_skills = st.text_input("Required Skills (comma-separated) *", placeholder="Python, Data Structures, Algorithms, Docker")
                new_pref_skills = st.text_input("Preferred Skills (comma-separated)", placeholder="AWS, Microservices, Kubernetes")
                new_desc = st.text_area("Role Description", placeholder="Brief description of key responsibilities and expectations...", height=80)
                
            post_submit = st.form_submit_button("🚀 Publish Job Drive to Database", use_container_width=True)
            
            if post_submit:
                validation_errors = []
                clean_company = new_company.strip()
                clean_role = new_role.strip()
                clean_location = new_location.strip() or "Bengaluru / Remote"
                clean_req_skills = new_req_skills.strip()
                clean_pref_skills = new_pref_skills.strip() if new_pref_skills else ""
                clean_desc = new_desc.strip() if new_desc else f"Campus recruitment opportunity with {clean_company} for {clean_role}."
                
                # Client-side field validations
                if not clean_company:
                    validation_errors.append("Company Name is required.")
                elif len(clean_company) < 2:
                    validation_errors.append("Company Name must be at least 2 characters.")
                    
                if not clean_role:
                    validation_errors.append("Role Title is required.")
                elif len(clean_role) < 2:
                    validation_errors.append("Role Title must be at least 2 characters.")
                    
                if not clean_req_skills:
                    validation_errors.append("At least one Required Skill must be specified.")
                else:
                    parsed_skills = [s.strip() for s in re.split(r'[,|;/]+', clean_req_skills) if s.strip()]
                    if not parsed_skills:
                        validation_errors.append("Please enter valid comma-separated Required Skills.")
                        
                if new_salary <= 0:
                    validation_errors.append("CTC Package must be greater than 0 LPA.")
                    
                if new_min_cgpa < 5.0 or new_min_cgpa > 10.0:
                    validation_errors.append("Minimum CGPA Cutoff must be between 5.0 and 10.0.")
                    
                if validation_errors:
                    for err in validation_errors:
                        st.toast(f"⚠️ {err}", icon="❌")
                    st.error("### ⚠️ Validation Issues:\n" + "\n".join([f"- {err}" for err in validation_errors]))
                else:
                    try:
                        conn = get_connection()
                        cursor = conn.cursor()
                        cursor.execute("""
                        INSERT INTO jobs (
                            company_name, role_title, department, min_cgpa, max_backlogs,
                            required_skills, preferred_skills, salary_lpa, location, description
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (clean_company, clean_role, new_dept, float(new_min_cgpa), int(new_max_backlogs), clean_req_skills, clean_pref_skills, float(new_salary), clean_location, clean_desc))
                        conn.commit()
                        conn.close()
                        
                        st.toast(f"🎉 Job Drive for {clean_company} ({clean_role}) published successfully!", icon="✅")
                        st.success(f"🎉 **{clean_company} - {clean_role}** drive published successfully! Refreshing database...")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.toast(f"Database error: {e}", icon="🚨")
                        st.error(f"Failed to publish job drive: {e}")

# -------------------------------------------------------------
# TAB 4: MODEL ANALYTICS & EXPLAINABLE AI
# -------------------------------------------------------------
with tabs[3]:
    st.markdown("<div class='main-title'>Model Performance & Explainability</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Rigorous evaluation metrics, comparison across classifier benchmarks, and feature impact analysis.</div>", unsafe_allow_html=True)
    
    bundle = predictor.bundle
    metrics = bundle["metrics"]
    all_models = bundle.get("all_model_evaluations", {})
    
    # KPI metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Champion Model", bundle["model_name"])
    m2.metric("Test Accuracy", f"{metrics['accuracy']:.2%}")
    m3.metric("Test Precision", f"{metrics['precision']:.2%}")
    m4.metric("Test Recall", f"{metrics['recall']:.2%}")
    m5.metric("ROC-AUC Score", f"{metrics['roc_auc']:.4f}")
    
    st.markdown("---")
    
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.markdown("#### 🏆 Multi-Model Benchmark Comparison")
        if all_models:
            comp_data = []
            for m_name, m_stats in all_models.items():
                comp_data.append({
                    "Model": m_name,
                    "Accuracy": m_stats["accuracy"],
                    "Precision": m_stats["precision"],
                    "Recall": m_stats["recall"],
                    "F1 Score": m_stats["f1"],
                    "ROC-AUC": m_stats["roc_auc"],
                    "5-Fold CV AUC": m_stats["cv_mean"]
                })
            df_comp = pd.DataFrame(comp_data)
            st.dataframe(df_comp.style.highlight_max(subset=["Accuracy", "F1 Score", "ROC-AUC"], color="#DCFCE7"), use_container_width=True)
            
            # Bar chart of ROC-AUC vs Accuracy
            fig_comp = px.bar(
                df_comp, x="Model", y=["Accuracy", "ROC-AUC", "F1 Score"],
                barmode="group",
                title="Classifier Metric Comparison",
                color_discrete_sequence=["#3B82F6", "#10B981", "#8B5CF6"]
            )
            fig_comp.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_comp, use_container_width=True)
            
    with c_right:
        st.markdown("#### 🔍 Feature Importance Ranking")
        feat_imp = bundle.get("feature_importances", {})
        if feat_imp:
            df_feat = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            fig_imp = px.bar(
                df_feat, x="Importance", y="Feature", orientation="h",
                title="Relative Impact on Placement Probability",
                color="Importance",
                color_continuous_scale="Blues"
            )
            fig_imp.update_layout(height=380, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_imp, use_container_width=True)
            
    # ROC Curve & Confusion Matrix
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("#### 📈 ROC (Receiver Operating Characteristic) Curve")
        roc_data = bundle.get("roc_curve", {})
        if roc_data:
            fpr = roc_data["fpr"]
            tpr = roc_data["tpr"]
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{bundle['model_name']} (AUC = {metrics['roc_auc']:.3f})", line=dict(color='#2563EB', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random Baseline", line=dict(color='#94A3B8', dash='dash')))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=320,
                margin=dict(t=30, b=30, l=30, r=30)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
    with r_col2:
        st.markdown("#### 🧮 Test Confusion Matrix")
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        cm_labels = ["Unplaced", "Placed"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Label", y="Actual Ground Truth", color="Count"),
            x=cm_labels,
            y=cm_labels,
            text_auto=True,
            color_continuous_scale="Teal"
        )
        fig_cm.update_layout(height=320, margin=dict(t=30, b=30, l=30, r=30))
        st.plotly_chart(fig_cm, use_container_width=True)

# -------------------------------------------------------------
# TAB 5: BATCH EVALUATION & TPO ADMIN
# -------------------------------------------------------------
with tabs[4]:
    st.markdown("<div class='main-title'>College Batch Placement Evaluator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Allows Training & Placement Officers (TPO) to assess an entire batch of graduating students simultaneously with branch-wise analytics and smart filters.</div>", unsafe_allow_html=True)
    
    if "evaluated_batch_df" not in st.session_state:
        st.session_state["evaluated_batch_df"] = None
        
    b_col1, b_col2 = st.columns([1, 1])
    with b_col1:
        st.markdown("##### 📥 Upload Student CSV")
        uploaded_file = st.file_uploader("Upload CSV formatted with student metrics", type=["csv"], key="tpo_csv_uploader")
        
    with b_col2:
        st.markdown("##### ⚡ Quick Demonstration")
        generate_demo = st.button("🎲 Generate & Evaluate Random Batch (40 Students)", use_container_width=True)
        
    if uploaded_file is not None:
        try:
            raw_batch_df = pd.read_csv(uploaded_file)
            st.toast(f"📥 Uploaded CSV with {len(raw_batch_df)} student rows", icon="ℹ️")
            
            # Check required columns
            missing_cols = [c for c in FEATURE_COLUMNS if c not in raw_batch_df.columns]
            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                with st.spinner("Evaluating batch with ML placement engine..."):
                    if "student_name" not in raw_batch_df.columns:
                        raw_batch_df.insert(0, "student_name", [f"Student_{i+101}" for i in range(len(raw_batch_df))])
                    if "branch" not in raw_batch_df.columns:
                        if "department" in raw_batch_df.columns:
                            raw_batch_df["branch"] = raw_batch_df["department"]
                        else:
                            dept_pool = ["Computer Science (CSE)", "Information Technology (IT)", "Electronics & Comm (ECE)", "AI & Data Science (AIDS)", "Mechanical Engg"]
                            raw_batch_df.insert(1, "branch", [dept_pool[i % len(dept_pool)] for i in range(len(raw_batch_df))])
                            
                    probs = []
                    preds = []
                    statuses = []
                    for _, row in raw_batch_df.iterrows():
                        res = predictor.predict(row.to_dict())
                        probs.append(round(res["placement_probability"] * 100, 2))
                        preds.append(res["prediction"])
                        statuses.append(res["status"])
                        
                    raw_batch_df["Placement_Probability_%"] = probs
                    raw_batch_df["Predicted_Outcome"] = ["Placed" if p == 1 else "Unplaced" for p in preds]
                    raw_batch_df["Risk_Category"] = statuses
                    st.session_state["evaluated_batch_df"] = raw_batch_df
                    st.toast("✅ Batch evaluated successfully!", icon="🎓")
        except Exception as e:
            st.error(f"Error reading or processing CSV: {e}")
            
    elif generate_demo:
        from generate_dataset import generate_placement_dataset
        raw_batch_df = generate_placement_dataset(num_samples=40, random_state=int(np.random.randint(1, 9999)))
        raw_batch_df.insert(0, "student_name", [f"Student_{i+101}" for i in range(len(raw_batch_df))])
        
        dept_pool = ["Computer Science (CSE)", "Information Technology (IT)", "Electronics & Comm (ECE)", "AI & Data Science (AIDS)", "Mechanical Engg"]
        raw_batch_df.insert(1, "branch", np.random.choice(dept_pool, size=len(raw_batch_df), p=[0.35, 0.25, 0.20, 0.12, 0.08]))
        
        with st.spinner("Evaluating 40 demo candidates across branches..."):
            probs = []
            preds = []
            statuses = []
            for _, row in raw_batch_df.iterrows():
                res = predictor.predict(row.to_dict())
                probs.append(round(res["placement_probability"] * 100, 2))
                preds.append(res["prediction"])
                statuses.append(res["status"])
                
            raw_batch_df["Placement_Probability_%"] = probs
            raw_batch_df["Predicted_Outcome"] = ["Placed" if p == 1 else "Unplaced" for p in preds]
            raw_batch_df["Risk_Category"] = statuses
            st.session_state["evaluated_batch_df"] = raw_batch_df
            st.toast("✅ Generated and evaluated 40 student cohort across 5 branches!", icon="🎲")
            
    batch_df = st.session_state.get("evaluated_batch_df")
    
    if batch_df is not None:
        total = len(batch_df)
        placed_count = int((batch_df["Predicted_Outcome"] == "Placed").sum())
        unplaced_count = total - placed_count
        rate = (placed_count / total) * 100 if total > 0 else 0
        avg_cgpa = float(batch_df["cgpa"].mean())
        avg_readiness = float(batch_df["Placement_Probability_%"].mean())
        
        st.markdown("---")
        # Summary KPI Metric Row
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Batch Students", f"{total}")
        s2.metric("Predicted Placements", f"{placed_count}")
        s3.metric("Placement Rate", f"{rate:.1f}%")
        s4.metric("At-Risk Students", f"{unplaced_count}")
        s5.metric("Avg Readiness Score", f"{avg_readiness:.1f}%")
        
        # ---------------------------------------------------------
        # INTERACTIVE TPO SUMMARY CHARTS: BRANCH-WISE BREAKDOWN
        # ---------------------------------------------------------
        st.markdown("#### 🏛️ College TPO Analytics: Branch-Wise Readiness Breakdown")
        
        # Compute branch-wise metrics
        branch_rows = []
        for b_name, grp in batch_df.groupby("branch"):
            b_total = len(grp)
            b_placed = int((grp["Predicted_Outcome"] == "Placed").sum())
            b_unplaced = b_total - b_placed
            b_rate = round((b_placed / b_total) * 100, 1) if b_total > 0 else 0
            b_avg_readiness = round(grp["Placement_Probability_%"].mean(), 1)
            b_avg_cgpa = round(grp["cgpa"].mean(), 2)
            b_avg_coding = round(grp["coding_score"].mean(), 1)
            branch_rows.append({
                "Branch": b_name,
                "Total Students": b_total,
                "Placed": b_placed,
                "Unplaced": b_unplaced,
                "Placement Rate (%)": b_rate,
                "Avg Readiness (%)": b_avg_readiness,
                "Avg CGPA": b_avg_cgpa,
                "Avg Coding Score": b_avg_coding
            })
        df_branch_stats = pd.DataFrame(branch_rows).sort_values("Placement Rate (%)", ascending=False)
        
        # Dual-Chart Layout
        ch1, ch2 = st.columns(2)
        with ch1:
            fig_branch_counts = px.bar(
                df_branch_stats,
                x="Branch",
                y=["Placed", "Unplaced"],
                barmode="group",
                title="Branch-Wise Placement Counts (Placed vs Unplaced)",
                labels={"value": "Number of Students", "variable": "Outcome"},
                color_discrete_map={"Placed": "#10B981", "Unplaced": "#EF4444"},
                text_auto=True
            )
            fig_branch_counts.update_layout(
                height=340,
                margin=dict(t=40, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_branch_counts, use_container_width=True)
            
        with ch2:
            fig_branch_rates = px.bar(
                df_branch_stats,
                x="Branch",
                y=["Placement Rate (%)", "Avg Readiness (%)"],
                barmode="group",
                title="Branch Placement Rate vs Avg Readiness Score",
                labels={"value": "Percentage (%)", "variable": "Metric"},
                color_discrete_map={"Placement Rate (%)": "#2563EB", "Avg Readiness (%)": "#8B5CF6"},
                text_auto='.1f'
            )
            fig_branch_rates.update_layout(
                height=340,
                margin=dict(t=40, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_branch_rates, use_container_width=True)
            
        with st.expander("📊 View Detailed Branch-Wise Metric Table & Probability Distribution"):
            sub_col1, sub_col2 = st.columns([1.3, 1.0])
            with sub_col1:
                st.markdown("##### 📋 Branch Benchmarking Matrix")
                st.dataframe(
                    df_branch_stats.style.highlight_max(subset=["Placement Rate (%)", "Avg Readiness (%)", "Avg CGPA"], color="#DCFCE7")
                                         .highlight_min(subset=["Unplaced"], color="#FEE2E2"),
                    use_container_width=True,
                    hide_index=True
                )
            with sub_col2:
                fig_batch = px.histogram(
                    batch_df, x="Placement_Probability_%", nbins=10,
                    color="Predicted_Outcome",
                    title="Batch Readiness Probability Distribution",
                    color_discrete_map={"Placed": "#10B981", "Unplaced": "#EF4444"}
                )
                fig_batch.update_layout(height=260, margin=dict(t=30, b=20, l=20, r=20))
                st.plotly_chart(fig_batch, use_container_width=True)
                
        # ---------------------------------------------------------
        # TPO BATCH CANDIDATE FILTERING CONTROLS
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("#### 🔍 Filter Batch Candidate Records")
        
        f1, f2, f3 = st.columns([1.3, 1.4, 1.3])
        
        # 1. Department Filter
        dept_options = ["All Departments"] + sorted(list(batch_df["branch"].dropna().unique()))
        selected_dept = f1.selectbox("Filter by Department / Branch", dept_options, key="tpo_filter_dept")
        
        # 2. CGPA Range Filter
        min_cgpa_bound = float(np.floor(batch_df["cgpa"].min() * 10) / 10)
        max_cgpa_bound = float(np.ceil(batch_df["cgpa"].max() * 10) / 10)
        selected_cgpa_range = f2.slider(
            "Filter by CGPA Range",
            min_value=max(4.0, min_cgpa_bound),
            max_value=min(10.0, max_cgpa_bound),
            value=(max(4.0, min_cgpa_bound), min(10.0, max_cgpa_bound)),
            step=0.1,
            key="tpo_filter_cgpa"
        )
        
        # 3. Status Filter
        status_options = ["All Outcomes", "Placed Only", "Unplaced Only", "At-Risk Only (< 45%)", "High Probability (>= 75%)"]
        selected_status = f3.selectbox("Filter by Placed / Risk Status", status_options, key="tpo_filter_status")
        
        # Apply Filters
        filtered_batch = batch_df.copy()
        
        if selected_dept != "All Departments":
            filtered_batch = filtered_batch[filtered_batch["branch"] == selected_dept]
            
        filtered_batch = filtered_batch[
            (filtered_batch["cgpa"] >= selected_cgpa_range[0]) &
            (filtered_batch["cgpa"] <= selected_cgpa_range[1])
        ]
        
        if selected_status == "Placed Only":
            filtered_batch = filtered_batch[filtered_batch["Predicted_Outcome"] == "Placed"]
        elif selected_status == "Unplaced Only":
            filtered_batch = filtered_batch[filtered_batch["Predicted_Outcome"] == "Unplaced"]
        elif selected_status == "At-Risk Only (< 45%)":
            filtered_batch = filtered_batch[filtered_batch["Placement_Probability_%"] < 45.0]
        elif selected_status == "High Probability (>= 75%)":
            filtered_batch = filtered_batch[filtered_batch["Placement_Probability_%"] >= 75.0]
            
        st.caption(f"Displaying **{len(filtered_batch)}** of **{len(batch_df)}** students matching active filters.")
        
        # Display Table
        st.dataframe(filtered_batch, use_container_width=True, hide_index=True)
        
        # Dual CSV Downloads
        d1, d2 = st.columns(2)
        with d1:
            filtered_csv = filtered_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download Filtered Cohort Report ({len(filtered_batch)} Students CSV)",
                data=filtered_csv,
                file_name="tpo_filtered_batch_report.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d2:
            full_csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download Complete Evaluated Batch Report ({len(batch_df)} Students CSV)",
                data=full_csv,
                file_name="tpo_full_evaluated_batch_report.csv",
                mime="text/csv",
                use_container_width=True
            )

import os
import sys
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

# Set page configuration
st.set_page_config(
    page_title="AI Placement Predictor & Job Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
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
    }
    .badge-eligible {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-ineligible {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
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
    default_certifications = 2
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
            
            certifications = st.number_input("Technical Certifications", min_value=0, max_value=6, value=int(default_certifications))
            
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
        "backlogs": backlogs
    }
    
    result = predictor.predict(profile_data)
    prob = result["placement_probability"]
    prob_percent = prob * 100
    status = result["status"]
    
    with col2:
        st.markdown("#### 🎯 Placement Prediction Output")
        
        # Plotly Gauge Chart
        gauge_color = "#10B981" if prob >= 0.75 else ("#F59E0B" if prob >= 0.45 else "#EF4444")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Placement Probability", 'font': {'size': 20, 'color': '#1E293B'}},
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
            st.success(f"**Status: {status}**\n\nThe candidate meets high placement probability benchmarks across academics and technical evaluation.")
        elif prob >= 0.45:
            st.warning(f"**Status: {status}**\n\nThe candidate has competitive baseline metrics but needs strengthening in key evaluated parameters.")
        else:
            st.error(f"**Status: {status}**\n\nHigh risk of campus placement rejection. Follow targeted guidance below to clear eligibility thresholds.")
            
        # Key Recommendations
        st.markdown("##### 📌 AI-Powered Personalized Recommendations:")
        for rec in result["recommendations"]:
            st.markdown(f"- {rec}")
            
        # Save to DB option
        if submit_btn:
            skills_str = ", ".join(skills_selected)
            saved_id = save_student_profile(
                name, f"{name.lower().replace(' ', '')}@college.edu", branch,
                cgpa, ssc, hsc, internships, projects, int(coding_score),
                int(soft_skills), backlogs, skills_str, round(prob, 4), status
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
    with st.expander("➕ Post New Company Job Drive (Recruiter / TPO Portal)"):
        with st.form("new_job_form"):
            c1, c2 = st.columns(2)
            with c1:
                new_company = st.text_input("Company Name", placeholder="e.g. Adobe")
                new_role = st.text_input("Role Title", placeholder="e.g. Member of Technical Staff")
                new_dept = st.selectbox("Department", ["Software Engineering", "Artificial Intelligence", "Cloud / Backend", "FinTech", "Security & Risk", "IT Services", "Analytics & Growth"])
                new_salary = st.number_input("CTC Package (LPA)", min_value=3.0, max_value=60.0, value=18.0, step=0.5)
            with c2:
                new_min_cgpa = st.slider("Minimum CGPA Cutoff", 5.0, 9.5, 7.5, 0.1)
                new_max_backlogs = st.number_input("Max Backlogs Allowed", 0, 5, 0)
                new_location = st.text_input("Job Location", value="Bengaluru / Remote")
                new_req_skills = st.text_input("Required Skills (comma-separated)", placeholder="Python, Data Structures, Algorithms, Docker")
                new_pref_skills = st.text_input("Preferred Skills (comma-separated)", placeholder="AWS, Microservices, Kubernetes")
                new_desc = st.text_area("Role Description", placeholder="Brief description of the responsibilities...")
                
            post_submit = st.form_submit_button("Publish Job Drive to Database")
            if post_submit:
                if new_company and new_role and new_req_skills:
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                    INSERT INTO jobs (
                        company_name, role_title, department, min_cgpa, max_backlogs,
                        required_skills, preferred_skills, salary_lpa, location, description
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (new_company, new_role, new_dept, new_min_cgpa, new_max_backlogs, new_req_skills, new_pref_skills or "", new_salary, new_location, new_desc))
                    conn.commit()
                    conn.close()
                    st.success(f"🎉 Job Drive for {new_company} ({new_role}) published successfully! Refreshing database.")
                    st.rerun()
                else:
                    st.error("Please fill in Company Name, Role Title, and Required Skills.")

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
    st.markdown("<div class='subtitle'>Allows placement officers (TPO) to assess an entire batch of graduating students simultaneously.</div>", unsafe_allow_html=True)
    
    b_col1, b_col2 = st.columns([1, 1])
    with b_col1:
        st.markdown("##### 📥 Upload Student CSV")
        uploaded_file = st.file_uploader("Upload CSV formatted with student metrics", type=["csv"])
        
    with b_col2:
        st.markdown("##### ⚡ Quick Demonstration")
        generate_demo = st.button("🎲 Generate & Evaluate Random Batch (30 Students)")
        
    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV with {len(batch_df)} student rows.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif generate_demo:
        from generate_dataset import generate_placement_dataset
        batch_df = generate_placement_dataset(num_samples=30, random_state=int(np.random.randint(1, 9999)))
        # Add random student names
        names = [f"Student_{i+101}" for i in range(len(batch_df))]
        batch_df.insert(0, "student_name", names)
        
    if batch_df is not None:
        # Check required columns
        missing_cols = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns in CSV: {missing_cols}")
        else:
            with st.spinner("Evaluating entire batch with ML placement engine..."):
                probs = []
                preds = []
                statuses = []
                
                for _, row in batch_df.iterrows():
                    res = predictor.predict(row.to_dict())
                    probs.append(round(res["placement_probability"] * 100, 2))
                    preds.append(res["prediction"])
                    statuses.append(res["status"])
                    
                batch_df["Placement_Probability_%"] = probs
                batch_df["Predicted_Outcome"] = ["Placed" if p == 1 else "Unplaced" for p in preds]
                batch_df["Risk_Category"] = statuses
                
            # Summary Metrics
            placed_count = sum(preds)
            total = len(batch_df)
            rate = (placed_count / total) * 100
            
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Batch Students", total)
            s2.metric("Predicted Placements", placed_count)
            s3.metric("Placement Rate", f"{rate:.1f}%")
            s4.metric("At-Risk Students", total - placed_count)
            
            # Distribution plot
            fig_batch = px.histogram(
                batch_df, x="Placement_Probability_%", nbins=10,
                color="Predicted_Outcome",
                title="Batch Placement Probability Distribution",
                color_discrete_map={"Placed": "#10B981", "Unplaced": "#EF4444"}
            )
            fig_batch.update_layout(height=280)
            st.plotly_chart(fig_batch, use_container_width=True)
            
            st.dataframe(batch_df, use_container_width=True)
            
            # Download CSV
            csv_export = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Evaluated Batch Report (CSV)",
                data=csv_export,
                file_name="evaluated_batch_placement_report.csv",
                mime="text/csv"
            )

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

# -------------------------------------------------------------
# APPLE LIQUID GLASS / VISIONOS DESIGN SYSTEM STYLESHEET
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --glass-bg: rgba(255, 255, 255, 0.62);
        --glass-bg-hover: rgba(255, 255, 255, 0.80);
        --glass-border: rgba(255, 255, 255, 0.85);
        --glass-border-subtle: rgba(255, 255, 255, 0.6);
        --glass-shadow: 0 16px 36px -6px rgba(31, 38, 135, 0.07), inset 0 1.5px 2px rgba(255, 255, 255, 0.95);
        --glass-shadow-hover: 0 22px 45px -4px rgba(37, 99, 235, 0.12), inset 0 1.5px 2px rgba(255, 255, 255, 1);
        --brand-blue: #2563EB;
        --brand-indigo: #4F46E5;
        --brand-purple: #7C3AED;
        --brand-cyan: #06B6D4;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
    }

    /* 1. LIQUID GLASS BACKGROUND ENVIRONMENT */
    .stApp {
        background:
            radial-gradient(1200px circle at 8% 10%, rgba(224, 231, 255, 0.85) 0%, transparent 55%),
            radial-gradient(900px circle at 92% 12%, rgba(219, 234, 254, 0.8) 0%, transparent 50%),
            radial-gradient(1000px circle at 50% 45%, rgba(240, 249, 255, 0.75) 0%, transparent 60%),
            radial-gradient(1200px circle at 85% 88%, rgba(238, 242, 255, 0.82) 0%, transparent 55%),
            radial-gradient(900px circle at 15% 92%, rgba(224, 242, 254, 0.75) 0%, transparent 50%),
            linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 50%, #EEF2F6 100%) !important;
        background-attachment: fixed !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* Top Header Bar Polish */
    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.35) !important;
        backdrop-filter: blur(24px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(190%) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.65) !important;
    }

    /* 2. TRANSLUCENT FLOATING GLASS SIDEBAR */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.52) !important;
        backdrop-filter: blur(35px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(35px) saturate(200%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.8) !important;
        box-shadow: 12px 0 35px -5px rgba(31, 38, 135, 0.04) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding: 1.5rem 1.25rem !important;
    }
    .sidebar-header-card {
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(255, 255, 255, 0.68);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 18px;
        padding: 14px 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 18px rgba(37, 99, 235, 0.06), inset 0 1px 1px rgba(255, 255, 255, 0.9);
    }
    .sidebar-brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 50%, #06B6D4 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        color: white;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.32);
    }
    .sidebar-brand-title {
        font-weight: 800;
        font-size: 1.12rem;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .sidebar-brand-sub {
        font-size: 0.72rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .sidebar-info-card {
        background: rgba(255, 255, 255, 0.58);
        border: 1px solid rgba(255, 255, 255, 0.85);
        border-radius: 16px;
        padding: 14px 16px;
        margin-top: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.02);
    }

    /* 3. HERO HEADER SECTION */
    .hero-header-container {
        margin-bottom: 1.8rem;
        padding: 4px 0;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(255, 255, 255, 0.95);
        border-radius: 9999px;
        padding: 5px 14px;
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        color: #2563EB;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.08), inset 0 1px 1px rgba(255, 255, 255, 1);
        backdrop-filter: blur(14px);
        margin-bottom: 12px;
    }
    .badge-glow-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 8px #10B981;
        display: inline-block;
    }
    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 35%, #4F46E5 70%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.35rem;
        filter: drop-shadow(0 2px 6px rgba(79, 70, 229, 0.08));
    }
    .subtitle {
        color: #475569;
        font-size: 1.02rem;
        font-weight: 400;
        line-height: 1.5;
        max-width: 920px;
    }

    /* 4. VISIONOS FLOATING PILL TABS */
    [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(25px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(25px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.85) !important;
        border-radius: 9999px !important;
        padding: 6px 8px !important;
        gap: 6px !important;
        box-shadow: 0 10px 25px -3px rgba(37, 99, 235, 0.08), inset 0 1px 1px rgba(255, 255, 255, 0.9) !important;
        margin-bottom: 2rem !important;
    }
    [data-baseweb="tab"] {
        border-radius: 9999px !important;
        padding: 9px 20px !important;
        font-weight: 600 !important;
        color: #475569 !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        font-size: 0.92rem !important;
    }
    [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.45) !important;
        color: #1E3A8A !important;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.92) 100%) !important;
        color: #1E40AF !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.15), inset 0 1px 1.5px rgba(255, 255, 255, 1) !important;
    }

    /* 5. LIQUID GLASS CARDS & FORMS */
    .liquid-glass-card, .card {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(28px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(190%) !important;
        border: 1px solid rgba(255, 255, 255, 0.85) !important;
        border-radius: 22px !important;
        padding: 24px 28px !important;
        box-shadow: 0 14px 35px -4px rgba(31, 38, 135, 0.06), inset 0 1.5px 2px rgba(255, 255, 255, 0.95), inset 0 -1px 2px rgba(0, 0, 0, 0.02) !important;
        margin-bottom: 1.25rem !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        position: relative !important;
    }
    .liquid-glass-card:hover, .card:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 20px 45px -4px rgba(37, 99, 235, 0.1), inset 0 1.5px 2px rgba(255, 255, 255, 1) !important;
    }

    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.62) !important;
        backdrop-filter: blur(32px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(32px) saturate(200%) !important;
        border: 1px solid rgba(255, 255, 255, 0.85) !important;
        border-radius: 24px !important;
        box-shadow: 0 16px 40px -4px rgba(31, 38, 135, 0.07), inset 0 1.5px 2px rgba(255, 255, 255, 1) !important;
        padding: 26px 28px !important;
    }

    /* 6. FROSTED GLASS INPUTS */
    div[data-baseweb="input"] {
        background: rgba(255, 255, 255, 0.72) !important;
        border: 1px solid rgba(203, 213, 225, 0.65) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(14px) !important;
        transition: all 0.25s ease !important;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }
    div[data-baseweb="input"]:focus-within {
        background: rgba(255, 255, 255, 0.95) !important;
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.18), inset 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }
    div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.72) !important;
        border: 1px solid rgba(203, 213, 225, 0.65) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(14px) !important;
    }
    div[data-baseweb="select"]:focus-within > div {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.18) !important;
    }

    /* 7. SLIDERS */
    div[data-testid="stSlider"] [role="slider"] {
        background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%) !important;
        box-shadow: 0 3px 12px rgba(37, 99, 235, 0.45), 0 0 0 3px rgba(255, 255, 255, 0.95) !important;
        border: none !important;
        height: 20px !important;
        width: 20px !important;
        transition: transform 0.15s ease !important;
    }
    div[data-testid="stSlider"] [role="slider"]:hover {
        transform: scale(1.18) !important;
    }

    /* 8. TAGS & MULTISELECT */
    div[data-baseweb="tag"] {
        background: rgba(238, 242, 255, 0.85) !important;
        border: 1px solid rgba(199, 210, 254, 0.8) !important;
        border-radius: 10px !important;
        color: #3730A3 !important;
        font-weight: 600 !important;
        backdrop-filter: blur(8px) !important;
        padding: 3px 8px !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="tag"]:hover {
        background: rgba(224, 231, 255, 1) !important;
        transform: translateY(-1px) !important;
    }

    /* Custom Glass Badges */
    .badge-eligible {
        background: linear-gradient(135deg, rgba(220, 252, 231, 0.9) 0%, rgba(209, 250, 229, 0.7) 100%) !important;
        color: #065F46 !important;
        border: 1px solid rgba(110, 231, 183, 0.8) !important;
        padding: 5px 12px !important;
        border-radius: 9999px !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.12) !important;
    }
    .badge-ineligible {
        background: linear-gradient(135deg, rgba(254, 226, 226, 0.9) 0%, rgba(254, 202, 202, 0.7) 100%) !important;
        color: #991B1B !important;
        border: 1px solid rgba(252, 165, 165, 0.8) !important;
        padding: 5px 12px !important;
        border-radius: 9999px !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.12) !important;
    }
    .tag-matched {
        background: rgba(209, 250, 229, 0.82) !important;
        color: #065F46 !important;
        border: 1px solid rgba(167, 243, 208, 0.9) !important;
        padding: 4px 10px !important;
        border-radius: 8px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        display: inline-block !important;
        margin: 3px !important;
        backdrop-filter: blur(8px) !important;
        transition: all 0.2s ease !important;
    }
    .tag-matched:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.18) !important;
    }
    .tag-missing {
        background: rgba(254, 226, 226, 0.82) !important;
        color: #991B1B !important;
        border: 1px solid rgba(254, 202, 202, 0.9) !important;
        padding: 4px 10px !important;
        border-radius: 8px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        display: inline-block !important;
        margin: 3px !important;
        backdrop-filter: blur(8px) !important;
    }
    .tag-pref {
        background: rgba(224, 231, 255, 0.82) !important;
        color: #3730A3 !important;
        border: 1px solid rgba(199, 210, 254, 0.9) !important;
        padding: 4px 10px !important;
        border-radius: 8px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        display: inline-block !important;
        margin: 3px !important;
        backdrop-filter: blur(8px) !important;
    }

    /* 9. PREDICTION STATUS BANNERS */
    .glass-status-banner {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        border-radius: 18px;
        padding: 16px 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.03);
        transition: all 0.25s ease;
    }
    .status-high {
        background: linear-gradient(135deg, rgba(220, 252, 231, 0.82) 0%, rgba(209, 250, 229, 0.6) 100%);
        border: 1px solid rgba(110, 231, 183, 0.8);
        color: #065F46;
    }
    .status-moderate {
        background: linear-gradient(135deg, rgba(254, 243, 199, 0.82) 0%, rgba(253, 230, 138, 0.6) 100%);
        border: 1px solid rgba(251, 191, 36, 0.8);
        color: #92400E;
    }
    .status-risk {
        background: linear-gradient(135deg, rgba(254, 226, 226, 0.82) 0%, rgba(254, 202, 202, 0.6) 100%);
        border: 1px solid rgba(248, 113, 113, 0.8);
        color: #991B1B;
    }
    .status-icon {
        font-size: 1.5rem;
        line-height: 1;
    }
    .status-title {
        font-weight: 700;
        font-size: 0.98rem;
        margin-bottom: 4px;
    }
    .status-desc {
        font-size: 0.86rem;
        line-height: 1.45;
        opacity: 0.95;
    }

    /* 10. AI RECOMMENDATION CARDS */
    .glass-rec-card {
        background: rgba(255, 255, 255, 0.68);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.85);
        border-left: 4px solid #3B82F6;
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        box-shadow: 0 4px 14px rgba(31, 38, 135, 0.03), inset 0 1px 1px rgba(255, 255, 255, 0.9);
        transition: all 0.25s ease;
    }
    .glass-rec-card:hover {
        transform: translateX(4px);
        background: rgba(255, 255, 255, 0.9);
        border-left-color: #6366F1;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.12);
    }
    .rec-bullet {
        color: #3B82F6;
        font-weight: 800;
        font-size: 1.1rem;
        line-height: 1.2;
    }
    .rec-text {
        font-size: 0.87rem;
        color: #1E293B;
        line-height: 1.45;
    }

    /* 11. PORTFOLIO & EXTRACURRICULAR GLASS PANEL */
    .glass-subpanel {
        background: linear-gradient(135deg, rgba(248, 250, 252, 0.75) 0%, rgba(241, 245, 249, 0.55) 100%);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 18px;
        padding: 18px 22px;
        margin-top: 14px;
        margin-bottom: 16px;
        backdrop-filter: blur(20px);
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.9), 0 4px 16px rgba(0, 0, 0, 0.02);
    }
    .glass-summary-pill {
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.85);
        border-radius: 12px;
        padding: 10px 16px;
        margin-bottom: 14px;
        font-size: 0.86rem;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
    }

    /* 12. BUTTONS (VISIONOS LIQUID BUTTONS) */
    div.stButton > button, [data-testid="stForm"] button {
        background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 45%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        letter-spacing: 0.015em !important;
        padding: 11px 24px !important;
        box-shadow: 0 6px 20px -2px rgba(59, 130, 246, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.45) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div.stButton > button:hover, [data-testid="stForm"] button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 10px 28px -2px rgba(59, 130, 246, 0.55), inset 0 1px 1.5px rgba(255, 255, 255, 0.6) !important;
    }
    div.stButton > button:active, [data-testid="stForm"] button:active {
        transform: translateY(0) scale(0.99) !important;
    }

    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #059669 0%, #10B981 50%, #06B6D4 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        box-shadow: 0 6px 20px -2px rgba(16, 185, 129, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.45) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 10px 28px -2px rgba(16, 185, 129, 0.5), inset 0 1px 1.5px rgba(255, 255, 255, 0.6) !important;
    }

    /* 13. METRIC CARDS */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.62) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 18px !important;
        padding: 16px 20px !important;
        box-shadow: 0 8px 24px -2px rgba(31, 38, 135, 0.05), inset 0 1px 1px rgba(255, 255, 255, 0.9) !important;
        transition: transform 0.2s ease !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
    }

    /* 14. EXPANDERS */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.75) !important;
        border-radius: 18px !important;
        backdrop-filter: blur(18px) !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.02) !important;
        overflow: hidden !important;
        margin-bottom: 1rem !important;
    }

    /* 15. RESPONSIVE BREAKPOINTS */
    @media screen and (max-width: 992px) {
        .main-title {
            font-size: 1.95rem !important;
        }
        .subtitle {
            font-size: 0.94rem !important;
        }
        .liquid-glass-card, .card {
            padding: 16px 18px !important;
        }
        div[data-testid="column"] {
            min-width: 260px !important;
            flex: 1 1 260px !important;
        }
    }
    @media screen and (max-width: 768px) {
        .main-title {
            font-size: 1.6rem !important;
        }
        .subtitle {
            font-size: 0.88rem !important;
        }
        div[data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        [data-baseweb="tab-list"] {
            border-radius: 16px !important;
            flex-wrap: wrap !important;
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

# Sidebar with Liquid Glass Aesthetics
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header-card">
        <div class="sidebar-brand-icon">🎓</div>
        <div>
            <div class="sidebar-brand-title">Placement AI</div>
            <div class="sidebar-brand-sub">Apple VisionOS Edition</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### ⚡ Student Profile Setup")
    preset = st.selectbox(
        "Load Profile Preset",
        ["Custom Input", "🌟 High Achiever (Software)", "🤖 Data Science Enthusiast", "⚠️ Student Needing Guidance"],
        help="Quickly populate candidate credentials with calibrated student archetypes"
    )
    
    st.markdown("""
    <div class="sidebar-info-card">
        <div style="font-weight: 700; color: #1E40AF; font-size: 0.82rem; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
            <span>✨</span> Liquid Glass Engine
        </div>
        <div style="font-size: 0.77rem; color: #64748B; line-height: 1.45;">
            Dynamic ML inference pipeline with real-time risk classification, skill gap vectorization, and TPO batch analytics.
        </div>
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown("""
    <div class="hero-header-container">
        <div class="hero-badge">
            <span class="badge-glow-dot"></span>
            <span>APPLE VISIONOS DESIGN SYSTEM • AI PLACEMENT PREDICTOR</span>
        </div>
        <div class="main-title">AI-Based Placement Prediction</div>
        <div class="subtitle">Evaluates academic credentials, practical projects, algorithmic acumen, and experience to compute placement readiness.</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1.0])
    
    with col1:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
            <h4 style="margin:0; font-weight:800; color:#0F172A; display:flex; align-items:center; gap:8px;">
                <span>📝</span> Candidate Assessment Form
            </h4>
            <span class="hero-badge" style="margin:0; padding:3px 10px; font-size:0.68rem;">LIVE PROFILER</span>
        </div>
        """, unsafe_allow_html=True)
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
            
            st.markdown("""
            <div class="glass-subpanel">
                <div style="font-weight:700; color:#1E3A8A; font-size:0.92rem; margin-bottom:8px; display:flex; align-items:center; gap:8px;">
                    <span>🏆</span> Portfolio & Extracurricular Excellence
                </div>
            """, unsafe_allow_html=True)
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                certifications = st.slider("📜 Certifications", min_value=0, max_value=10, value=int(default_certifications), step=1, help="Industry certifications (AWS, Azure, GCP, Oracle Java, etc.)")
            with p_col2:
                github_repos = st.slider("🐙 GitHub Repos", min_value=0, max_value=30, value=int(default_github), step=1, help="Public GitHub repositories showcasing project code and commit history")
            with p_col3:
                hackathons = st.slider("🥇 Hackathon Wins", min_value=0, max_value=8, value=int(default_hackathons), step=1, help="Hackathon victories, finalist ranks, or podium finishes")
            st.markdown("</div>", unsafe_allow_html=True)
            
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
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
            <h4 style="margin:0; font-weight:800; color:#0F172A; display:flex; align-items:center; gap:8px;">
                <span>🎯</span> Placement Readiness Output
            </h4>
            <span class="badge-eligible" style="font-size:0.72rem;">ML INFERENCE V2.4</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Plotly Gauge Chart with transparent glass integration
        gauge_color = "#10B981" if prob >= 0.75 else ("#F59E0B" if prob >= 0.45 else "#EF4444")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Placement Readiness Score", 'font': {'size': 18, 'color': '#0F172A', 'family': 'Plus Jakarta Sans, sans-serif'}},
            number={'suffix': "%", 'font': {'size': 38, 'color': gauge_color, 'family': 'Plus Jakarta Sans, sans-serif'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                'bar': {'color': gauge_color, 'thickness': 0.28},
                'bgcolor': "rgba(241, 245, 249, 0.45)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.8)",
                'steps': [
                    {'range': [0, 45], 'color': 'rgba(254, 226, 226, 0.55)'},
                    {'range': [45, 75], 'color': 'rgba(254, 243, 199, 0.55)'},
                    {'range': [75, 100], 'color': 'rgba(220, 252, 231, 0.55)'}
                ],
                'threshold': {
                    'line': {'color': gauge_color, 'width': 4},
                    'thickness': 0.8,
                    'value': prob_percent
                }
            }
        ))
        fig.update_layout(
            height=250,
            margin=dict(l=15, r=15, t=35, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans, sans-serif")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Status Box (Apple Liquid Glass status card)
        if prob >= 0.75:
            st.markdown(f"""
            <div class="glass-status-banner status-high">
                <div class="status-icon">🌟</div>
                <div>
                    <div class="status-title">Status: {status}</div>
                    <div class="status-desc">The candidate meets tier-1 placement probability benchmarks across academics, algorithmic coding, and portfolio evaluation.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif prob >= 0.45:
            st.markdown(f"""
            <div class="glass-status-banner status-moderate">
                <div class="status-icon">⚡</div>
                <div>
                    <div class="status-title">Status: {status}</div>
                    <div class="status-desc">The candidate has competitive baseline metrics but needs strengthening in key evaluated parameters before company visits.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="glass-status-banner status-risk">
                <div class="status-icon">⚠️</div>
                <div>
                    <div class="status-title">Status: {status}</div>
                    <div class="status-desc">Elevated risk of screening rejection. Follow targeted guidance below to clear company eligibility cutoffs.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Summary pill for extracurriculars
        st.markdown(f"""
        <div class="glass-summary-pill">
            <span>📜 <b>{certifications}</b> Certifications</span> &nbsp;•&nbsp;
            <span>🐙 <b>{github_repos}</b> Repos</span> &nbsp;•&nbsp;
            <span>🥇 <b>{hackathons}</b> Hackathons</span>
        </div>
        """, unsafe_allow_html=True)
            
        # Key Recommendations in Liquid Glass cards
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-top:14px; margin-bottom:10px;">
            <span style="font-size:1.1rem;">💡</span>
            <span style="font-weight:700; color:#1E3A8A; font-size:0.95rem;">AI-Powered Action Plan & Guidance:</span>
        </div>
        """, unsafe_allow_html=True)
        for rec in result["recommendations"]:
            st.markdown(f"""
            <div class="glass-rec-card">
                <div class="rec-bullet">⚡</div>
                <div class="rec-text">{rec}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Downloadable PDF Report Panel
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.55); border: 1px solid rgba(255, 255, 255, 0.85); border-radius: 16px; padding: 14px 18px; margin-top: 14px; backdrop-filter: blur(16px);">
            <div style="font-weight: 700; color: #1E3A8A; font-size: 0.92rem; margin-bottom: 6px; display: flex; align-items: center; gap: 8px;">
                <span>📄</span> Official Placement Readiness Report
            </div>
            <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 12px; line-height: 1.4;">
                Download an official student PDF portfolio containing ML readiness score, academic metrics, AI guidance, and matched company roles.
            </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Placement Readiness & Job Report (PDF)",
            data=pdf_report_bytes,
            file_name=f"{name.replace(' ', '_')}_Placement_Readiness_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Download formal PDF report with placement readiness score, competency metrics, AI guidance, and matching jobs."
        )
        st.markdown("</div>", unsafe_allow_html=True)
            
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
    st.markdown("""
    <div class="hero-header-container">
        <div class="hero-badge">
            <span class="badge-glow-dot"></span>
            <span>TF-IDF COSINE SIMILARITY • REAL-TIME SKILL GAP VECTORIZATION</span>
        </div>
        <div class="main-title">AI Job Recommender & Skill Gap Analysis</div>
        <div class="subtitle">Matches candidate profile against company openings using Content-Based Filtering, eligibility cutoffs, and skill gap identification.</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
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
                st.markdown("<div class='liquid-glass-card'>", unsafe_allow_html=True)
                
                # Card Header
                h_col1, h_col2 = st.columns([2.5, 1])
                with h_col1:
                    st.markdown(f"""
                    <h3 style='margin:0 0 6px 0; font-weight:800; font-size:1.3rem; color:#0F172A;'>
                        {job['role_title']} &nbsp;•&nbsp; <span style='background:linear-gradient(135deg, #2563EB, #7C3AED); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>{job['company_name']}</span>
                    </h3>
                    <div style='color:#475569; font-size:0.88rem;'>
                        <b>Domain:</b> {job['department']} &nbsp;|&nbsp; <b>Location:</b> {job['location']} &nbsp;|&nbsp; <b>CTC:</b> <span style='background:rgba(238,242,255,0.9); border:1px solid rgba(199,210,254,0.8); padding:3px 10px; border-radius:8px; font-weight:700; color:#1E40AF;'>₹ {job['salary_lpa']} LPA</span>
                    </div>
                    """, unsafe_allow_html=True)
                with h_col2:
                    score = job['match_percentage']
                    badge_class = "badge-eligible" if job['is_eligible'] else "badge-ineligible"
                    st.markdown(f"<div style='text-align:right;'><span class='{badge_class}'>{job['eligibility_status']}</span></div>", unsafe_allow_html=True)
                    st.progress(score / 100.0)
                    st.markdown(f"<div style='text-align:right; font-weight:700; color:#1E3A8A; font-size:0.88rem;'>{score}% Profile Match</div>", unsafe_allow_html=True)
                
                # Skill badges
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
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
                    st.markdown(f"<div style='color:#B91C1C; font-size:0.85rem; margin-top:8px;'>⚠️ <b>Eligibility Notice:</b> {' | '.join(job['eligibility_reasons'])}</div>", unsafe_allow_html=True)
                
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
    st.markdown("""
    <div class="hero-header-container">
        <div class="hero-badge">
            <span class="badge-glow-dot"></span>
            <span>SQLITE DATABASE EXPLORER • CAMPUS RECRUITMENT DRIVES</span>
        </div>
        <div class="main-title">Company & Job Openings Database</div>
        <div class="subtitle">Manage and explore real-time company recruitment drives stored in the SQLite database with instant query filtering.</div>
    </div>
    """, unsafe_allow_html=True)
    
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
        <div style='background:rgba(255,255,255,0.7); border:1px solid rgba(255,255,255,0.9); border-radius:12px; padding:12px 16px; margin-bottom:12px; font-size:0.9rem; color:#475569;'>
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
    st.markdown("""
    <div class="hero-header-container">
        <div class="hero-badge">
            <span class="badge-glow-dot"></span>
            <span>MULTI-CLASSIFIER BENCHMARK • EXPLAINABLE AI</span>
        </div>
        <div class="main-title">Model Performance & Explainability</div>
        <div class="subtitle">Rigorous evaluation metrics, comparison across classifier benchmarks, and feature impact analysis.</div>
    </div>
    """, unsafe_allow_html=True)
    
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
            
            # Bar chart of ROC-AUC vs Accuracy with transparent glass theme
            fig_comp = px.bar(
                df_comp, x="Model", y=["Accuracy", "ROC-AUC", "F1 Score"],
                barmode="group",
                title="Classifier Metric Comparison",
                color_discrete_sequence=["#3B82F6", "#10B981", "#8B5CF6"]
            )
            fig_comp.update_layout(
                height=320,
                margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans, sans-serif")
            )
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
            fig_imp.update_layout(
                height=380,
                margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans, sans-serif")
            )
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
                margin=dict(t=30, b=30, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans, sans-serif")
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
        fig_cm.update_layout(
            height=320,
            margin=dict(t=30, b=30, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans, sans-serif")
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# -------------------------------------------------------------
# TAB 5: BATCH EVALUATION & TPO ADMIN
# -------------------------------------------------------------
with tabs[4]:
    st.markdown("""
    <div class="hero-header-container">
        <div class="hero-badge">
            <span class="badge-glow-dot"></span>
            <span>COHORT PLACEMENT INTELLIGENCE • TPO COMMAND CENTER</span>
        </div>
        <div class="main-title">College Batch Placement Evaluator</div>
        <div class="subtitle">Allows Training & Placement Officers (TPO) to assess an entire batch of graduating students simultaneously with branch-wise analytics and smart filters.</div>
    </div>
    """, unsafe_allow_html=True)
    
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
        # Summary KPI Metric Row with frosted glass styling
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
        
        # Dual-Chart Layout with transparent canvas
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
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans, sans-serif"),
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
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans, sans-serif"),
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
                fig_batch.update_layout(
                    height=260,
                    margin=dict(t=30, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans, sans-serif")
                )
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

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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

# Custom CSS for modern design (Highlighted Neural Network, Glowing Wavy Lines & Calm Aurora)
st.markdown(
    '<div class="aurora-bg-layer" aria-hidden="true">'
    '<div class="orb-wrap orb-wrap-1"><div class="orb orb-1"></div></div>'
    '<div class="orb-wrap orb-wrap-2"><div class="orb orb-2"></div></div>'
    '<div class="orb-wrap orb-wrap-3"><div class="orb orb-3"></div></div>'
    '<div class="orb-wrap orb-wrap-4"><div class="orb orb-4"></div></div>'
    '<div class="orb-wrap orb-wrap-5"><div class="orb orb-5"></div></div>'
    '<div class="geo-ring gr-1"></div>'
    '<div class="geo-ring gr-2"></div>'
    '<svg class="bipartite-arcs" viewBox="0 0 1440 900" preserveAspectRatio="none">'
    '<defs>'
    '<filter id="waveGlow" x="-10%" y="-10%" width="120%" height="120%">'
    '<feGaussianBlur stdDeviation="3.5" result="blur"/>'
    '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
    '</filter>'
    '<linearGradient id="arcGrad1" x1="0%" y1="0%" x2="100%" y2="0%">'
    '<stop offset="0%" stop-color="#1E3A8A" stop-opacity="0.65"/>'
    '<stop offset="38%" stop-color="#3B82F6" stop-opacity="0.72"/>'
    '<stop offset="70%" stop-color="#8B5CF6" stop-opacity="0.62"/>'
    '<stop offset="100%" stop-color="#10B981" stop-opacity="0.75"/>'
    '</linearGradient>'
    '<linearGradient id="arcGrad2" x1="0%" y1="0%" x2="100%" y2="0%">'
    '<stop offset="0%" stop-color="#06B6D4" stop-opacity="0.58"/>'
    '<stop offset="50%" stop-color="#6366F1" stop-opacity="0.65"/>'
    '<stop offset="100%" stop-color="#10B981" stop-opacity="0.68"/>'
    '</linearGradient>'
    '<linearGradient id="ribbonGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
    '<stop offset="0%" stop-color="#3B82F6" stop-opacity="0.14"/>'
    '<stop offset="50%" stop-color="#A78BFA" stop-opacity="0.16"/>'
    '<stop offset="100%" stop-color="#10B981" stop-opacity="0.14"/>'
    '</linearGradient>'
    '</defs>'
    '<path class="wave-ribbon wr-1" d="M -40,170 Q 340,60 720,190 T 1480,150" stroke="url(#ribbonGrad)"/>'
    '<path class="wave-ribbon wr-2" d="M -40,680 Q 380,800 760,650 T 1480,710" stroke="url(#ribbonGrad)"/>'
    '<path class="flow-arc arc-1" filter="url(#waveGlow)" d="M -20,140 C 320,40 620,290 960,150 S 1260,60 1460,170" stroke="url(#arcGrad1)"/>'
    '<path class="flow-arc arc-2" filter="url(#waveGlow)" d="M -20,340 C 380,470 720,180 1060,360 S 1290,480 1460,320" stroke="url(#arcGrad2)"/>'
    '<path class="flow-arc arc-3" filter="url(#waveGlow)" d="M -20,560 C 340,440 690,710 1020,530 S 1280,430 1460,590" stroke="url(#arcGrad1)"/>'
    '<path class="flow-arc arc-4" filter="url(#waveGlow)" d="M -20,760 C 420,660 780,860 1120,720 S 1340,630 1460,770" stroke="url(#arcGrad2)"/>'
    '<path class="flow-arc-solid" filter="url(#waveGlow)" d="M -20,250 Q 360,150 720,260 T 1460,230" stroke="url(#arcGrad2)"/>'
    '</svg>'
    '<div class="bg-chip chip-1">🎓 Placement AI</div>'
    '<div class="bg-chip chip-2">💼 Role Match</div>'
    '<div class="bg-chip chip-3">⚡ ML Score</div>'
    '<div class="bg-chip chip-4">&lt;/&gt; DSA</div>'
    '<div class="bg-chip chip-5">📊 CGPA Fit</div>'
    '<div class="bg-chip chip-6">✓ Eligible</div>'
    '<div class="bg-chip chip-7">🚀 Career Path</div>'
    '<div class="bg-chip chip-8">🔍 Skill Graph</div>'
    '<div class="bg-chip chip-9">🧠 Neural Match</div>'
    '<div class="bg-chip chip-10">🎯 93% ROC-AUC</div>'
    '<div class="bg-chip chip-11">💼 Tier-1 Offers</div>'
    '<div class="bg-chip chip-12">☁️ Cloud &amp; AI</div>'
    '<div class="scroll-gradient-veil"></div>'
    '<div class="grid-shimmer"></div>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown("""<style>
/* ============================================================
   Scroll-Reactive Multi-Stop Aurora & Neural Glassmorphism Theme
   ============================================================ */
:root {
    --bg-c1: #CFE4FF;
    --bg-c2: #DDD6FE;
    --bg-c3: #B8F5DB;
    --bg-c4: #FFF1E6;
    --bg-angle: -45deg;
    --scroll-hue: 0deg;
    --veil-x: 22%;
    --veil-y: 24%;
}
/* ---- Layer 1: Enhanced Scroll-Reactive Aurora Gradient Canvas ---- */
[data-testid="stAppViewContainer"],
.stApp {
    background: linear-gradient(var(--bg-angle), var(--bg-c1), var(--bg-c2), var(--bg-c3), var(--bg-c4), var(--bg-c2)) !important;
    background-size: 380% 380% !important;
    animation: auroraShift 20s ease-in-out infinite !important;
    transition: background 0.35s ease;
    position: relative;
}
@keyframes auroraShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
/* ---- Scroll-Reactive Radial & Conic Veil ---- */
.scroll-gradient-veil {
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at var(--veil-x) var(--veil-y), rgba(99, 102, 241, 0.24) 0%, transparent 52%),
        radial-gradient(circle at calc(100% - var(--veil-x)) calc(100% - var(--veil-y)), rgba(16, 185, 129, 0.24) 0%, transparent 55%),
        conic-gradient(from var(--bg-angle) at 50% 50%, rgba(59, 130, 246, 0.08), rgba(167, 139, 250, 0.14), rgba(251, 191, 146, 0.12), rgba(16, 185, 129, 0.11), rgba(59, 130, 246, 0.08));
    filter: hue-rotate(var(--scroll-hue));
    pointer-events: none;
    z-index: 0;
    transition: filter 0.35s ease;
}
/* ---- Layer 2: Calm Floating Blurred Mesh Orbs + Scroll Parallax Wrappers ---- */
.aurora-bg-layer {
    position: fixed;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
}
.orb-wrap {
    position: absolute;
    inset: 0;
    pointer-events: none;
    will-change: transform;
    filter: hue-rotate(var(--scroll-hue));
    transition: transform 0.55s cubic-bezier(0.22, 1, 0.36, 1), filter 0.35s ease;
}
.orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(55px);
    will-change: transform, opacity;
}
.orb-1 {
    width: 560px; height: 560px;
    top: -120px; left: -60px;
    background: radial-gradient(circle at center, rgba(59, 130, 246, 0.48) 0%, rgba(59, 130, 246, 0) 70%);
    animation: orbFloat1 32s ease-in-out infinite;
}
.orb-2 {
    width: 620px; height: 620px;
    top: 10%; right: -120px;
    background: radial-gradient(circle at center, rgba(16, 185, 129, 0.42) 0%, rgba(16, 185, 129, 0) 70%);
    animation: orbFloat2 38s ease-in-out infinite;
    animation-delay: -5s;
}
.orb-3 {
    width: 540px; height: 540px;
    bottom: -140px; left: 20%;
    background: radial-gradient(circle at center, rgba(167, 139, 250, 0.48) 0%, rgba(167, 139, 250, 0) 70%);
    animation: orbFloat3 34s ease-in-out infinite;
    animation-delay: -10s;
}
.orb-4 {
    width: 460px; height: 460px;
    top: 42%; left: 38%;
    background: radial-gradient(circle at center, rgba(6, 182, 212, 0.38) 0%, rgba(6, 182, 212, 0) 70%);
    animation: orbFloat4 40s ease-in-out infinite;
    animation-delay: -14s;
}
.orb-5 {
    width: 440px; height: 440px;
    bottom: 6%; right: 10%;
    background: radial-gradient(circle at center, rgba(251, 191, 146, 0.42) 0%, rgba(244, 114, 182, 0.16) 40%, rgba(251, 191, 146, 0) 70%);
    animation: orbFloat1 36s ease-in-out infinite reverse;
    animation-delay: -8s;
}
@keyframes orbFloat1 {
    0%   { transform: translate3d(0, 0, 0) scale(1); }
    50%  { transform: translate3d(220px, 140px, 0) scale(1.14); }
    100% { transform: translate3d(0, 0, 0) scale(1); }
}
@keyframes orbFloat2 {
    0%   { transform: translate3d(0, 0, 0) scale(1.08); }
    50%  { transform: translate3d(-240px, 170px, 0) scale(0.94); }
    100% { transform: translate3d(0, 0, 0) scale(1.08); }
}
@keyframes orbFloat3 {
    0%   { transform: translate3d(0, 0, 0) scale(0.96); }
    50%  { transform: translate3d(170px, -190px, 0) scale(1.18); }
    100% { transform: translate3d(0, 0, 0) scale(0.96); }
}
@keyframes orbFloat4 {
    0%   { transform: translate3d(0, 0, 0) scale(1); }
    33%  { transform: translate3d(-180px, 100px, 0) scale(1.12); }
    66%  { transform: translate3d(150px, -120px, 0) scale(0.96); }
    100% { transform: translate3d(0, 0, 0) scale(1); }
}
/* ---- Placement & Recommendation Thematic Floating Skill/Job Chips ---- */
.bg-chip {
    position: absolute;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    color: rgba(30, 58, 138, 0.55);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.72), rgba(219, 234, 254, 0.56), rgba(209, 250, 229, 0.54));
    border: 1px solid rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.12);
    will-change: transform, opacity;
    white-space: nowrap;
}
.chip-1 { left: 6%;  bottom: -40px; animation: chipRise 26s linear infinite 0s; color: rgba(30, 58, 138, 0.72); }
.chip-2 { left: 19%; bottom: -40px; animation: chipRise 31s linear infinite -7s; color: rgba(4, 120, 87, 0.72); }
.chip-3 { left: 32%; bottom: -40px; animation: chipRise 28s linear infinite -14s; color: rgba(67, 56, 202, 0.70); }
.chip-4 { left: 46%; bottom: -40px; animation: chipRise 33s linear infinite -4s; color: rgba(29, 78, 216, 0.70); }
.chip-5 { left: 58%; bottom: -40px; animation: chipRise 29s linear infinite -18s; color: rgba(15, 118, 110, 0.72); }
.chip-6 { left: 71%; bottom: -40px; animation: chipRise 27s linear infinite -11s; color: rgba(4, 120, 87, 0.72); }
.chip-7 { left: 83%; bottom: -40px; animation: chipRise 34s linear infinite -22s; color: rgba(109, 40, 217, 0.68); }
.chip-8 { left: 91%; bottom: -40px; animation: chipRise 36s linear infinite -26s; color: rgba(30, 58, 138, 0.70); }
.chip-9 { left: 14%; bottom: -40px; animation: chipRise 30s linear infinite -16s; color: rgba(67, 56, 202, 0.70); }
.chip-10 { left: 40%; bottom: -40px; animation: chipRise 32s linear infinite -24s; color: rgba(29, 78, 216, 0.72); }
.chip-11 { left: 66%; bottom: -40px; animation: chipRise 35s linear infinite -9s; color: rgba(4, 120, 87, 0.70); }
.chip-12 { left: 79%; bottom: -40px; animation: chipRise 29s linear infinite -19s; color: rgba(14, 116, 144, 0.72); }
@keyframes chipRise {
    0%   { transform: translate3d(0, 0, 0) scale(0.92); opacity: 0; }
    12%  { opacity: 0.78; }
    85%  { opacity: 0.70; }
    100% { transform: translate3d(35px, -112vh, 0) scale(1.04); opacity: 0; }
}
/* ---- Rotating Geometric Tech Rings ---- */
.geo-ring {
    position: absolute;
    border-radius: 50%;
    border: 1.5px dashed rgba(59, 130, 246, 0.28);
    box-shadow: inset 0 0 36px rgba(99, 102, 241, 0.08);
    pointer-events: none;
    will-change: transform;
}
.gr-1 {
    width: 440px; height: 440px;
    top: -110px; right: -90px;
    animation: spinSlow 46s linear infinite;
}
.gr-2 {
    width: 380px; height: 380px;
    bottom: -100px; left: 16%;
    border-color: rgba(16, 185, 129, 0.28);
    animation: spinSlow 54s linear infinite reverse;
}
@keyframes spinSlow {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
/* ---- Live Neural Skill-to-Job Matching Canvas ---- */
#placement-net-canvas,
#sidebar-net-canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 0;
}
/* ---- Layer 3: Architectural Dot Grid Shimmer ---- */
.grid-shimmer {
    position: absolute;
    inset: 0;
    background-image: radial-gradient(rgba(30, 58, 138, 0.14) 1.2px, transparent 1.2px);
    background-size: 26px 26px;
    opacity: 0.36;
    animation: gridBreathe 10s ease-in-out infinite;
}
@keyframes gridBreathe {
    0%, 100% { opacity: 0.24; }
    50%      { opacity: 0.44; }
}
/* ---- Zero-height helper iframe cleanup ---- */
iframe[height="0"] {
    display: none !important;
}
/* ---- Transparent Header + Strict Foreground Layering ---- */
[data-testid="stHeader"],
[data-testid="stHeader"] > div {
    background: transparent !important;
}
[data-testid="stMainBlockContainer"],
.block-container {
    position: relative;
    z-index: 1;
}
[data-testid="stElementContainer"]:has(.aurora-bg-layer) {
    position: fixed !important;
    inset: 0 !important;
    z-index: 0 !important;
    pointer-events: none !important;
}
[data-testid="stElementContainer"]:not(:has(.aurora-bg-layer)) {
    position: relative;
    z-index: 2;
}
/* ---- Animated Sidebar ("Student Profile Setup") ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(var(--bg-angle), var(--bg-c3), var(--bg-c1), var(--bg-c2), var(--bg-c4)) !important;
    background-size: 350% 350% !important;
    animation: auroraShift 20s ease-in-out infinite !important;
    transition: background 0.35s ease;
    border-right: 1px solid rgba(255, 255, 255, 0.80);
    box-shadow: 4px 0 28px rgba(30, 58, 138, 0.10);
    overflow: hidden !important;
}
[data-testid="stSidebarContent"] {
    background: transparent !important;
    position: relative;
    z-index: 2;
}
.sidebar-bg-layer {
    position: absolute !important;
    inset: 0;
    width: 100%;
    height: 100%;
    min-height: 100vh;
    overflow: hidden;
    pointer-events: none;
    z-index: 0 !important;
}
.sidebar-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(42px);
    will-change: transform;
}
.s-orb-1 {
    width: 240px; height: 240px;
    top: -40px; left: -50px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.45) 0%, rgba(59, 130, 246, 0) 70%);
    animation: orbFloat1 26s ease-in-out infinite;
}
.s-orb-2 {
    width: 260px; height: 260px;
    bottom: 6%; right: -60px;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.40) 0%, rgba(16, 185, 129, 0) 70%);
    animation: orbFloat3 30s ease-in-out infinite;
}
.s-orb-3 {
    width: 220px; height: 220px;
    top: 42%; left: 10%;
    background: radial-gradient(circle, rgba(167, 139, 250, 0.40) 0%, rgba(167, 139, 250, 0) 70%);
    animation: orbFloat2 28s ease-in-out infinite;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.90), rgba(224, 242, 254, 0.80), rgba(237, 233, 254, 0.75)) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.85) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(30, 58, 138, 0.08);
}
/* ---- Main Title: Slow, Elegant Gradient Shimmer (9s) ---- */
.main-title {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
    background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981, #3B82F6, #1E3A8A);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: titleShimmer 9s linear infinite;
}
@keyframes titleShimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
/* ---- Subtitle ---- */
.subtitle {
    color: #1E293B;
    font-size: 1.05rem;
    font-weight: 500;
    opacity: 0.88;
    margin-bottom: 1.5rem;
}
/* ---- Frosted Glass Cards ---- */
.card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.90), rgba(239, 246, 255, 0.80), rgba(236, 253, 245, 0.78)) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.85);
    border-radius: 16px;
    padding: 18px 22px;
    margin-bottom: 1rem;
    box-shadow: 0 8px 28px rgba(30, 58, 138, 0.08);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(30, 58, 138, 0.14);
}
/* ---- Frosted Glass Gradient Forms, Metrics, Charts, Expanders & Uploader ---- */
[data-testid="stForm"],
[data-testid="stMetric"],
[data-testid="stPlotlyChart"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stFileUploader"] section,
div[data-testid="stExpander"] {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.90), rgba(239, 246, 255, 0.80), rgba(236, 253, 245, 0.78)) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.88) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 28px rgba(30, 58, 138, 0.08);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(224, 242, 254, 0.80), rgba(237, 233, 254, 0.76)) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    animation: fadeSlideUp 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}
[data-testid="stMetric"] {
    padding: 14px 16px;
}
[data-testid="stPlotlyChart"] {
    padding: 8px;
}
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] summary {
    background: transparent !important;
}
[data-testid="stForm"]:hover,
[data-testid="stMetric"]:hover,
[data-testid="stVerticalBlockBorderWrapper"]:hover,
[data-testid="stFileUploader"] section:hover,
div[data-testid="stExpander"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(30, 58, 138, 0.14);
}
button[kind="secondary"],
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(219, 234, 254, 0.82), rgba(209, 250, 229, 0.78)) !important;
    border: 1px solid rgba(255, 255, 255, 0.90) !important;
    border-radius: 12px !important;
    color: #1E3A8A !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(30, 58, 138, 0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
button[kind="secondary"]:hover,
[data-testid="stDownloadButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.20);
}
/* ---- Job Role Cards with Background Company Logo Watermarks ---- */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .role-bg-watermark) {
    position: relative !important;
    overflow: hidden !important;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.90), rgba(224, 242, 254, 0.76), rgba(236, 253, 245, 0.76)) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.88) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 28px rgba(30, 58, 138, 0.08);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeSlideUp 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .role-bg-watermark):hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(30, 58, 138, 0.14);
}
[data-testid="stElementContainer"]:has(.role-bg-watermark) {
    position: absolute !important;
    inset: 0 !important;
    z-index: 0 !important;
    pointer-events: none !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stElementContainer"]:has(.role-bg-watermark) > div,
[data-testid="stElementContainer"]:has(.role-bg-watermark) [data-testid="stMarkdown"],
[data-testid="stElementContainer"]:has(.role-bg-watermark) [data-testid="stMarkdownContainer"] {
    width: 100% !important;
    height: 100% !important;
    position: static !important;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .role-bg-watermark) > *:not(:has(.role-bg-watermark)) {
    position: relative;
    z-index: 2;
}
.role-bg-watermark {
    position: absolute;
    right: 44px;
    top: 50%;
    width: 205px;
    height: 205px;
    opacity: 0.38;
    filter: saturate(1.75) contrast(1.15) drop-shadow(0 14px 26px rgba(30, 58, 138, 0.25));
    transform-style: preserve-3d;
    will-change: transform, opacity;
    pointer-events: none;
    transition: opacity 0.4s ease, filter 0.4s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: logoSpin3D 18s linear infinite;
}
@keyframes logoSpin3D {
    0%   { transform: translateY(-50%) perspective(950px) rotateX(12deg) rotateY(0deg) rotateZ(-6deg) scale(1); }
    25%  { transform: translateY(-52%) perspective(950px) rotateX(-8deg) rotateY(90deg) rotateZ(0deg) scale(1.04); }
    50%  { transform: translateY(-50%) perspective(950px) rotateX(12deg) rotateY(180deg) rotateZ(6deg) scale(1.07); }
    75%  { transform: translateY(-48%) perspective(950px) rotateX(-8deg) rotateY(270deg) rotateZ(0deg) scale(1.04); }
    100% { transform: translateY(-50%) perspective(950px) rotateX(12deg) rotateY(360deg) rotateZ(-6deg) scale(1); }
}
.role-bg-watermark svg {
    width: 100%;
    height: 100%;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .role-bg-watermark):hover .role-bg-watermark {
    opacity: 0.54;
    filter: saturate(1.85) contrast(1.18) drop-shadow(0 18px 34px rgba(59, 130, 246, 0.36));
}
/* ---- Primary Submit Button Shimmer ---- */
button[kind="primary"],
.stFormSubmitButton button {
    background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981, #3B82F6, #1E3A8A) !important;
    background-size: 200% auto !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600;
    border-radius: 10px !important;
    transition: background-position 0.6s ease, transform 0.25s ease, box-shadow 0.25s ease;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.35);
}
button[kind="primary"]:hover,
.stFormSubmitButton button:hover {
    background-position: right center !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.45);
}
/* ---- Sleek Glass Pill Tabs (Streamlit 1.65 React Aria + BaseWeb) ---- */
[data-testid="stTabs"] [role="tablist"],
[data-baseweb="tab-list"] {
    background: linear-gradient(90deg, rgba(255, 255, 255, 0.88), rgba(224, 242, 254, 0.78), rgba(237, 233, 254, 0.78), rgba(209, 250, 229, 0.78)) !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.80) !important;
    border-radius: 999px !important;
    padding: 6px !important;
    box-shadow: 0 4px 18px rgba(30, 58, 138, 0.08);
    gap: 6px;
}
[data-testid="stTabs"] [role="tab"],
[data-baseweb="tab"] {
    border-radius: 999px !important;
    padding: 8px 16px !important;
    transition: background 0.25s ease, color 0.25s ease, transform 0.2s ease !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"],
[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.30);
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p {
    color: #FFFFFF !important;
}
[data-testid="stTabs"] [role="tab"]::after,
[data-testid="stTabs"] [role="tab"]::before,
[data-baseweb="tab-highlight"] {
    display: none !important;
}
[data-testid="stTabs"] [role="tab"] {
    border-bottom: none !important;
}
/* ---- Badges & Tags ---- */
.badge-eligible {
    background: rgba(16, 185, 129, 0.16);
    color: #047857;
    border: 1px solid rgba(16, 185, 129, 0.35);
    backdrop-filter: blur(8px);
    border-radius: 9999px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-ineligible {
    background: rgba(239, 68, 68, 0.14);
    color: #B91C1C;
    border: 1px solid rgba(239, 68, 68, 0.32);
    backdrop-filter: blur(8px);
    border-radius: 9999px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.tag-matched {
    background: rgba(59, 130, 246, 0.16);
    color: #1D4ED8;
    border: 1px solid rgba(59, 130, 246, 0.32);
    backdrop-filter: blur(8px);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
    margin: 2px;
}
.tag-missing {
    background: rgba(245, 158, 11, 0.16);
    color: #B45309;
    border: 1px solid rgba(245, 158, 11, 0.32);
    backdrop-filter: blur(8px);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
    margin: 2px;
}
.tag-pref {
    background: rgba(99, 102, 241, 0.14);
    color: #4338CA;
    border: 1px solid rgba(99, 102, 241, 0.30);
    backdrop-filter: blur(8px);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
    margin: 2px;
}
/* ---- DataFrames & Plotly: keep crisp over background ---- */
[data-testid="stDataFrame"],
.stDataFrame,
.js-plotly-plot,
.plotly-notifier {
    position: relative;
    z-index: 1;
}
/* ---- Toast alerts clean over background ---- */
[data-testid="stToast"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(224, 242, 254, 0.90), rgba(209, 250, 229, 0.88)) !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.70);
    box-shadow: 0 8px 28px rgba(30, 58, 138, 0.14);
    z-index: 9999;
}
/* ---- Highlighted Glowing Wavy Lines & Ribbons ---- */
.bipartite-arcs,
.sidebar-waves {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 0;
    filter: hue-rotate(var(--scroll-hue));
    transition: filter 0.35s ease;
}
.wave-ribbon {
    fill: none;
    stroke-width: 24;
    stroke-linecap: round;
    animation: ribbonDrift 14s ease-in-out infinite alternate;
}
.wr-2 { animation-duration: 18s; animation-direction: alternate-reverse; }
@keyframes ribbonDrift {
    0%   { transform: translateY(-10px); opacity: 0.75; }
    100% { transform: translateY(14px); opacity: 1; }
}
.flow-arc {
    fill: none;
    stroke-width: 3;
    stroke-dasharray: 18 14;
    stroke-linecap: round;
    animation: arcFlow 12s linear infinite;
}
.arc-2 { stroke-width: 2.6; animation-duration: 16s; animation-direction: reverse; stroke-dasharray: 14 16; }
.arc-3 { stroke-width: 3.2; animation-duration: 19s; stroke-dasharray: 22 14; }
.arc-4 { stroke-width: 2.4; animation-duration: 15s; animation-direction: reverse; stroke-dasharray: 12 18; }
.flow-arc-solid {
    fill: none;
    stroke-width: 2.2;
    opacity: 0.70;
    animation: ribbonDrift 12s ease-in-out infinite alternate;
}
@keyframes arcFlow {
    0%   { stroke-dashoffset: 384; }
    100% { stroke-dashoffset: 0; }
}
/* ---- Sidebar Eligibility Radar / Sonar Pulse ---- */
.sonar-ring {
    position: absolute;
    top: 86px;
    left: 50%;
    width: 140px;
    height: 140px;
    margin-left: -70px;
    border-radius: 50%;
    border: 2px solid rgba(59, 130, 246, 0.42);
    box-shadow: 0 0 18px rgba(59, 130, 246, 0.20);
    pointer-events: none;
    animation: sonarPulse 5.5s ease-out infinite;
}
.sr-2 {
    border-color: rgba(16, 185, 129, 0.42);
    animation-delay: 2.75s;
}
@keyframes sonarPulse {
    0%   { transform: scale(0.45); opacity: 0.85; }
    100% { transform: scale(2.4); opacity: 0; }
}
/* ---- UI Helper Pills & Sidebar Status Card ---- */
.sidebar-status-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.88), rgba(224, 242, 254, 0.78), rgba(237, 233, 254, 0.75));
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.90);
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 14px;
    box-shadow: 0 6px 20px rgba(30, 58, 138, 0.08);
    font-size: 0.86rem;
    color: #1E293B;
}
.sidebar-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px dashed rgba(148, 163, 184, 0.35);
}
.sidebar-stat-row:last-child { border-bottom: none; }
.kpi-mini-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 12px;
}
.kpi-mini-pill {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(219, 234, 254, 0.78), rgba(209, 250, 229, 0.75));
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.90);
    border-radius: 12px;
    padding: 10px 8px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(30, 58, 138, 0.07);
}
.kpi-mini-label {
    font-size: 0.72rem;
    color: #64748B;
    font-weight: 600;
    text-transform: uppercase;
}
.kpi-mini-val {
    font-size: 1.08rem;
    font-weight: 800;
    color: #1E3A8A;
    margin-top: 2px;
}
/* ---- Prediction Risk-State Ambient Glows & Staggered Recommendations ---- */
.risk-banner {
    border-radius: 14px;
    padding: 14px 18px;
    margin: 10px 0 14px 0;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    animation: fadeSlideUp 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.risk-glow-high {
    background: linear-gradient(135deg, rgba(220, 252, 231, 0.86), rgba(167, 243, 208, 0.68), rgba(209, 250, 229, 0.80));
    border: 1px solid rgba(16, 185, 129, 0.45);
    color: #065F46;
    box-shadow: 0 0 24px rgba(16, 185, 129, 0.22);
    animation: fadeSlideUp 0.55s ease both, pulseGreen 4s ease-in-out infinite;
}
.risk-glow-moderate {
    background: linear-gradient(135deg, rgba(254, 243, 199, 0.86), rgba(253, 230, 138, 0.68), rgba(255, 247, 237, 0.82));
    border: 1px solid rgba(245, 158, 11, 0.45);
    color: #92400E;
    box-shadow: 0 0 22px rgba(245, 158, 11, 0.20);
}
.risk-glow-low {
    background: linear-gradient(135deg, rgba(254, 226, 226, 0.88), rgba(254, 205, 211, 0.72), rgba(255, 241, 242, 0.82));
    border: 1px solid rgba(239, 68, 68, 0.45);
    color: #991B1B;
    box-shadow: 0 0 22px rgba(239, 68, 68, 0.22);
    animation: fadeSlideUp 0.55s ease both, pulseRed 3.2s ease-in-out infinite;
}
@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 4px 18px rgba(16, 185, 129, 0.16); }
    50%      { box-shadow: 0 6px 28px rgba(16, 185, 129, 0.34); }
}
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 4px 18px rgba(239, 68, 68, 0.16); }
    50%      { box-shadow: 0 6px 28px rgba(239, 68, 68, 0.34); }
}
.rec-item {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.86), rgba(239, 246, 255, 0.78), rgba(237, 233, 254, 0.74));
    border-left: 3px solid #3B82F6;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 8px;
    font-size: 0.92rem;
    color: #1E293B;
    opacity: 0;
    animation: fadeSlideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}
/* ---- Staggered Job Card Entrance & Animated Match Progress Bars ---- */
@keyframes fadeSlideUp {
    0%   { opacity: 0; transform: translate3d(0, 16px, 0); }
    100% { opacity: 1; transform: translate3d(0, 0, 0); }
}
.job-card-anim {
    opacity: 0;
    animation: fadeSlideUp 0.55s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}
.match-bar-track {
    width: 100%;
    height: 10px;
    background: rgba(203, 213, 225, 0.55);
    border-radius: 999px;
    overflow: hidden;
    margin: 8px 0 4px 0;
}
.match-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
    transform-origin: left center;
    animation: barGrow 0.9s cubic-bezier(0.22, 1, 0.36, 1) both;
}
@keyframes barGrow {
    0%   { transform: scaleX(0); }
    100% { transform: scaleX(1); }
}
.tag-matched, .tag-missing, .tag-pref {
    animation: tagPop 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
}
@keyframes tagPop {
    0%   { opacity: 0; transform: scale(0.88); }
    100% { opacity: 1; transform: scale(1); }
}
/* ---- Input & Slider Focus Glow Micro-Interactions ---- */
[data-baseweb="input"]:focus-within,
[data-baseweb="select"] > div:focus-within,
[data-baseweb="textarea"]:focus-within {
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.22) !important;
    border-color: #3B82F6 !important;
    transition: box-shadow 0.25s ease, border-color 0.25s ease !important;
}
/* ---- Reduced Motion Fallback ---- */
@media (prefers-reduced-motion: reduce) {
    [data-testid="stAppViewContainer"], .stApp, [data-testid="stSidebar"] {
        animation: none !important;
        background: linear-gradient(120deg, #DBEAFE 0%, #EDE9FE 50%, #FFF5EB 100%) !important;
    }
    .orb, .orb-wrap, .sidebar-orb, .sonar-ring, .flow-arc, .flow-arc-solid, .wave-ribbon,
    .geo-ring, .bg-chip, .grid-shimmer, .main-title, .risk-glow-high, .risk-glow-low,
    .rec-item, .job-card-anim, .match-bar-fill, .tag-matched, .tag-missing, .tag-pref,
    .role-bg-watermark, button[kind="primary"], .stFormSubmitButton button {
        animation: none !important;
        opacity: 1 !important;
        transform: none !important;
    }
    .bg-chip { opacity: 0.35 !important; }
    .role-bg-watermark { opacity: 0.30 !important; transform: translateY(-50%) rotate(-8deg) !important; }
    .grid-shimmer { opacity: 0.30; }
    * { transition: none !important; }
}
</style>""", unsafe_allow_html=True)

# AI Skill-to-Job Constellation Network (Main + Student Profile Setup Sidebar) + Scroll Parallax
components.html("""
<script>
(function() {
    const parentWin = window.parent;
    const doc = parentWin.document;
    const reducedMotion = parentWin.matchMedia && parentWin.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (parentWin.__auroraCleanup) {
        parentWin.__auroraCleanup();
    }

    // 1. Scroll parallax + orb direction reversal
    const factors = [
        { sel: '.orb-wrap-1', vx: 0.06,  vy: -0.10 },
        { sel: '.orb-wrap-2', vx: -0.08, vy: 0.08 },
        { sel: '.orb-wrap-3', vx: 0.05,  vy: -0.12 },
        { sel: '.orb-wrap-4', vx: -0.04, vy: -0.06 },
        { sel: '.orb-wrap-5', vx: 0.07,  vy: 0.09 }
    ];
    let lastScroll = 0;
    let dir = 1;
    let ticking = false;

    function getScrollMetrics(target) {
        const main = doc.querySelector('[data-testid="stMain"]') || doc.querySelector('[data-testid="stAppViewContainer"]');
        const scroller = (target && typeof target.scrollTop === 'number' && target !== doc && target.scrollHeight > target.clientHeight) ? target : main;
        const top = (scroller ? scroller.scrollTop : 0) || parentWin.scrollY || doc.documentElement.scrollTop || 0;
        const maxRange = scroller ? Math.max(scroller.scrollHeight - scroller.clientHeight, 450) : 650;
        return { top: top, maxRange: maxRange };
    }

    // 4-stop RGB palettes that smoothly morph as the user scrolls
    const scrollPalettes = [
        [[207, 228, 255], [221, 214, 254], [184, 245, 219], [255, 241, 230]], // Top: Azure, Lavender, Mint, Pearl
        [[186, 230, 253], [233, 213, 255], [165, 243, 252], [245, 208, 254]], // 33%: Electric Sky, Violet, Cyan, Orchid
        [[167, 243, 208], [254, 215, 170], [251, 207, 232], [199, 210, 254]], // 66%: Emerald, Peach Pearl, Rose Quartz, Periwinkle
        [[199, 210, 254], [254, 205, 211], [153, 246, 228], [253, 230, 138]]  // Bottom: Dusk Indigo, Coral Rose, Teal, Warm Gold
    ];

    function lerpRgb(cA, cB, t) {
        const r = Math.round(cA[0] + (cB[0] - cA[0]) * t);
        const g = Math.round(cA[1] + (cB[1] - cA[1]) * t);
        const b = Math.round(cA[2] + (cB[2] - cA[2]) * t);
        return 'rgb(' + r + ', ' + g + ', ' + b + ')';
    }

    function applyScrollColors(scrollY, maxRange) {
        const progress = Math.max(0, Math.min(1, scrollY / Math.max(maxRange, 1)));
        const scaled = progress * (scrollPalettes.length - 1);
        const idx = Math.min(scrollPalettes.length - 2, Math.floor(scaled));
        const localT = scaled - idx;
        const pA = scrollPalettes[idx];
        const pB = scrollPalettes[idx + 1];
        const rootStyle = doc.documentElement.style;

        rootStyle.setProperty('--bg-c1', lerpRgb(pA[0], pB[0], localT));
        rootStyle.setProperty('--bg-c2', lerpRgb(pA[1], pB[1], localT));
        rootStyle.setProperty('--bg-c3', lerpRgb(pA[2], pB[2], localT));
        rootStyle.setProperty('--bg-c4', lerpRgb(pA[3], pB[3], localT));
        rootStyle.setProperty('--bg-angle', (-45 + progress * 165).toFixed(1) + 'deg');
        rootStyle.setProperty('--scroll-hue', (progress * 95).toFixed(1) + 'deg');
        rootStyle.setProperty('--veil-x', (22 + progress * 56).toFixed(1) + '%');
        rootStyle.setProperty('--veil-y', (24 + Math.sin(progress * Math.PI) * 44).toFixed(1) + '%');
    }

    function applyParallax(scrollY, maxRange) {
        factors.forEach(function(f) {
            const el = doc.querySelector(f.sel);
            if (!el) return;
            const dx = Math.max(-90, Math.min(90, scrollY * f.vx * dir));
            const dy = Math.max(-90, Math.min(90, scrollY * f.vy));
            el.style.transform = 'translate3d(' + dx.toFixed(1) + 'px, ' + dy.toFixed(1) + 'px, 0)';
        });
        applyScrollColors(scrollY, maxRange);
    }

    let lastMaxRange = 650;
    function onScroll(e) {
        if (reducedMotion) return;
        const metrics = getScrollMetrics(e.target);
        lastMaxRange = metrics.maxRange;
        if (Math.abs(metrics.top - lastScroll) > 2) {
            dir = metrics.top >= lastScroll ? 1 : -1;
            lastScroll = metrics.top;
        }
        if (!ticking) {
            ticking = true;
            parentWin.requestAnimationFrame(function() {
                applyParallax(lastScroll, lastMaxRange);
                ticking = false;
            });
        }
    }

    // Apply initial scroll palette on load
    const initMetrics = getScrollMetrics(null);
    applyScrollColors(initMetrics.top, initMetrics.maxRange);

    // 2. Ensure Sidebar animated layer exists inside [data-testid="stSidebar"]
    function ensureSidebarLayer() {
        const sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (!sidebar) return null;
        let layer = sidebar.querySelector('.sidebar-bg-layer');
        if (!layer) {
            layer = doc.createElement('div');
            layer.className = 'sidebar-bg-layer';
            layer.setAttribute('aria-hidden', 'true');
            layer.innerHTML =
                '<div class="sidebar-orb s-orb-1"></div>' +
                '<div class="sidebar-orb s-orb-2"></div>' +
                '<div class="sidebar-orb s-orb-3"></div>' +
                '<div class="sonar-ring sr-1"></div>' +
                '<div class="sonar-ring sr-2"></div>' +
                '<svg class="sidebar-waves" viewBox="0 0 320 900" preserveAspectRatio="none">' +
                '<defs><linearGradient id="sWaveGrad" x1="0%" y1="0%" x2="100%" y2="100%">' +
                '<stop offset="0%" stop-color="#3B82F6" stop-opacity="0.55"/>' +
                '<stop offset="50%" stop-color="#8B5CF6" stop-opacity="0.50"/>' +
                '<stop offset="100%" stop-color="#10B981" stop-opacity="0.60"/>' +
                '</linearGradient></defs>' +
                '<path class="flow-arc arc-1" d="M -10,160 C 90,80 210,260 330,180" stroke="url(#sWaveGrad)"/>' +
                '<path class="flow-arc arc-2" d="M -10,470 C 110,560 200,360 330,490" stroke="url(#sWaveGrad)"/>' +
                '<path class="flow-arc arc-3" d="M -10,740 C 80,660 230,830 330,720" stroke="url(#sWaveGrad)"/>' +
                '</svg>' +
                '<canvas id="sidebar-net-canvas"></canvas>';
            sidebar.insertBefore(layer, sidebar.firstChild);
        }
        return layer.querySelector('#sidebar-net-canvas');
    }

    // 3. Ensure Main Constellation Canvas exists inside .aurora-bg-layer
    function ensureMainCanvas() {
        const layer = doc.querySelector('.aurora-bg-layer');
        if (!layer) return null;
        let canvas = layer.querySelector('#placement-net-canvas');
        if (!canvas) {
            canvas = doc.createElement('canvas');
            canvas.id = 'placement-net-canvas';
            layer.appendChild(canvas);
        }
        return canvas;
    }

    const pointer = { x: -9999, y: -9999, active: false };

    // 4. Highlighted Neural Network & Skill-to-Job Matching Renderer
    function createConstellation(canvas, opts) {
        if (!canvas) return null;
        const ctx = canvas.getContext('2d');
        let width = 0, height = 0;
        const nodes = [];

        function seedNodes() {
            nodes.length = 0;
            for (let i = 0; i < opts.count; i++) {
                const kind = i % 2 === 0 ? 'skill' : 'job';
                const isHub = i % 4 === 0;
                nodes.push({
                    x: Math.random() * width,
                    y: Math.random() * height,
                    vx: (Math.random() - 0.5) * opts.speed,
                    vy: (Math.random() - 0.5) * opts.speed,
                    r: isHub ? (5.2 + Math.random() * 1.8) : (kind === 'job' ? (3.6 + Math.random() * 1.4) : (2.8 + Math.random() * 1.2)),
                    kind: kind,
                    isHub: isHub,
                    phase: Math.random() * Math.PI * 2
                });
            }
        }

        function resize() {
            const parentEl = canvas.parentElement;
            const hostEl = parentEl.parentElement || parentEl;
            const rect = parentEl.getBoundingClientRect();
            const hostRect = hostEl.getBoundingClientRect();
            const dpr = Math.min(parentWin.devicePixelRatio || 1, 2);
            const prevH = height;
            width = Math.max(rect.width, hostRect.width, 1);
            height = Math.max(rect.height, hostRect.height, parentWin.innerHeight || 1);
            canvas.width = Math.floor(width * dpr);
            canvas.height = Math.floor(height * dpr);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            if (nodes.length === 0 || (prevH <= 10 && height > 50)) {
                seedNodes();
            }
        }
        resize();

        let ro = null;
        if (typeof parentWin.ResizeObserver === 'function') {
            ro = new parentWin.ResizeObserver(function() { resize(); });
            ro.observe(canvas.parentElement);
            if (canvas.parentElement.parentElement) {
                ro.observe(canvas.parentElement.parentElement);
            }
        }

        function step(time) {
            ctx.clearRect(0, 0, width, height);
            const hostRect = canvas.getBoundingClientRect();
            const localPtrX = pointer.x - hostRect.left;
            const localPtrY = pointer.y - hostRect.top;

            if (!reducedMotion) {
                for (let i = 0; i < nodes.length; i++) {
                    const n = nodes[i];
                    n.x += n.vx;
                    n.y += n.vy;
                    if (n.x < -12) n.x = width + 12;
                    if (n.x > width + 12) n.x = -12;
                    if (n.y < -12) n.y = height + 12;
                    if (n.y > height + 12) n.y = -12;
                }
            }

            // Draw highlighted neural synaptic edges & glowing signal impulses
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const a = nodes[i];
                    const b = nodes[j];
                    const dx = b.x - a.x;
                    const dy = b.y - a.y;
                    const dist = Math.hypot(dx, dy);
                    if (dist < opts.linkDist) {
                        const strength = 1 - dist / opts.linkDist;
                        const isMatch = a.kind !== b.kind;
                        ctx.beginPath();
                        ctx.moveTo(a.x, a.y);
                        ctx.lineTo(b.x, b.y);
                        if (isMatch) {
                            ctx.strokeStyle = 'rgba(16, 185, 129, ' + Math.min(0.85, strength * opts.matchAlpha).toFixed(3) + ')';
                            ctx.lineWidth = 1.45 + strength * 0.9;
                        } else {
                            ctx.strokeStyle = 'rgba(79, 70, 229, ' + Math.min(0.65, strength * opts.peerAlpha).toFixed(3) + ')';
                            ctx.lineWidth = 1.0 + strength * 0.6;
                        }
                        ctx.stroke();

                        // Bright neural impulse + trailing glow along Skill <-> Job edges
                        if (isMatch && !reducedMotion && strength > 0.25) {
                            const t = ((time * 0.00048 + (i * 0.17 + j * 0.11)) % 1);
                            const px = a.x + dx * t;
                            const py = a.y + dy * t;
                            ctx.beginPath();
                            ctx.arc(px, py, 2.7, 0, Math.PI * 2);
                            ctx.fillStyle = 'rgba(16, 185, 129, ' + Math.min(0.95, strength * 0.95).toFixed(3) + ')';
                            ctx.shadowColor = 'rgba(16, 185, 129, 0.85)';
                            ctx.shadowBlur = 8;
                            ctx.fill();
                            ctx.shadowBlur = 0;
                        }
                    }
                }
            }

            // Subtle pointer proximity synaptic lines
            if (pointer.active && !reducedMotion && localPtrX >= 0 && localPtrX <= width && localPtrY >= 0 && localPtrY <= height) {
                for (let i = 0; i < nodes.length; i++) {
                    const n = nodes[i];
                    const d = Math.hypot(localPtrX - n.x, localPtrY - n.y);
                    if (d < opts.linkDist * 0.95) {
                        const s = 1 - d / (opts.linkDist * 0.95);
                        ctx.beginPath();
                        ctx.moveTo(n.x, n.y);
                        ctx.lineTo(localPtrX, localPtrY);
                        ctx.strokeStyle = 'rgba(6, 182, 212, ' + (s * 0.45).toFixed(3) + ')';
                        ctx.lineWidth = 1.2;
                        ctx.stroke();
                    }
                }
            }

            // Draw glowing Skill & Job neural nodes + hub orbital rings
            for (let i = 0; i < nodes.length; i++) {
                const n = nodes[i];
                const pulse = reducedMotion ? 1 : 0.86 + 0.16 * Math.sin(time * 0.0024 + n.phase);
                const radius = n.r * pulse;

                // Outer halo
                ctx.beginPath();
                ctx.arc(n.x, n.y, radius * 2.1, 0, Math.PI * 2);
                ctx.fillStyle = n.kind === 'job' ? 'rgba(16, 185, 129, 0.14)' : 'rgba(59, 130, 246, 0.14)';
                ctx.fill();

                // Hub rotating dashed orbital ring
                if (n.isHub) {
                    ctx.save();
                    ctx.translate(n.x, n.y);
                    ctx.rotate(time * 0.0007 + n.phase);
                    ctx.setLineDash([4, 4]);
                    ctx.beginPath();
                    ctx.arc(0, 0, radius + 5.5, 0, Math.PI * 2);
                    ctx.strokeStyle = n.kind === 'job' ? 'rgba(16, 185, 129, 0.48)' : 'rgba(99, 102, 241, 0.48)';
                    ctx.lineWidth = 1.1;
                    ctx.stroke();
                    ctx.restore();
                }

                // Core glowing node
                ctx.beginPath();
                ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
                if (n.kind === 'job') {
                    ctx.fillStyle = 'rgba(16, 185, 129, 0.82)';
                    ctx.shadowColor = 'rgba(16, 185, 129, 0.75)';
                } else {
                    ctx.fillStyle = 'rgba(59, 130, 246, 0.82)';
                    ctx.shadowColor = 'rgba(99, 102, 241, 0.70)';
                }
                ctx.shadowBlur = 14;
                ctx.fill();
                ctx.shadowBlur = 0;
            }
        }

        return {
            resize: resize,
            step: step,
            disconnect: function() { if (ro) ro.disconnect(); }
        };
    }

    let mainNet = null;
    let sideNet = null;
    let rafId = null;
    let retryTimer = null;
    let attempts = 0;

    function initNets() {
        if (!mainNet) {
            const mc = ensureMainCanvas();
            if (mc) {
                mainNet = createConstellation(mc, { count: 52, speed: 0.40, linkDist: 190, matchAlpha: 0.72, peerAlpha: 0.45 });
            }
        }
        if (!sideNet) {
            const sc = ensureSidebarLayer();
            if (sc) {
                sideNet = createConstellation(sc, { count: 22, speed: 0.34, linkDist: 130, matchAlpha: 0.70, peerAlpha: 0.44 });
            }
        }
        if ((!mainNet || !sideNet) && attempts < 25) {
            attempts += 1;
            retryTimer = parentWin.setTimeout(initNets, 150);
        }
    }

    initNets();

    function animate(time) {
        if (mainNet) mainNet.step(time);
        if (sideNet) sideNet.step(time);
        if (!reducedMotion) {
            rafId = parentWin.requestAnimationFrame(animate);
        }
    }
    rafId = parentWin.requestAnimationFrame(animate);

    function onResize() {
        if (mainNet) mainNet.resize();
        if (sideNet) sideNet.resize();
    }

    // 5. KPI Metric Roll-Up Animation
    function animateMetrics() {
        if (reducedMotion) return;
        const els = doc.querySelectorAll('[data-testid="stMetricValue"] div, [data-testid="stMetricValue"]');
        els.forEach(function(el) {
            if (el.children.length > 0) return;
            const raw = (el.textContent || '').trim();
            if (!raw || el.dataset.rolledVal === raw) return;
            const m = raw.match(/^([0-9]+(?:\.[0-9]+)?)(%?)$/);
            if (!m) return;
            const target = parseFloat(m[1]);
            const suffix = m[2] || '';
            const decimals = (m[1].split('.')[1] || '').length;
            el.dataset.rolledVal = raw;
            const start = performance.now();
            const dur = 680;
            function tick(now) {
                const t = Math.min((now - start) / dur, 1);
                const ease = 1 - Math.pow(1 - t, 3);
                const cur = target * ease;
                const formatted = cur.toFixed(decimals) + suffix;
                el.dataset.rolledVal = formatted;
                el.textContent = formatted;
                if (t < 1) {
                    parentWin.requestAnimationFrame(tick);
                } else {
                    el.dataset.rolledVal = raw;
                    el.textContent = raw;
                }
            }
            parentWin.requestAnimationFrame(tick);
        });
    }
    animateMetrics();
    let metricObs = null;
    if (typeof parentWin.MutationObserver === 'function') {
        metricObs = new parentWin.MutationObserver(function() { animateMetrics(); });
        metricObs.observe(doc.body, { childList: true, subtree: true });
    }

    function onMouseMove(e) {
        pointer.x = e.clientX;
        pointer.y = e.clientY;
        pointer.active = true;
    }
    function onMouseLeave() {
        pointer.active = false;
    }

    doc.addEventListener('scroll', onScroll, { capture: true, passive: true });
    parentWin.addEventListener('scroll', onScroll, { passive: true });
    parentWin.addEventListener('resize', onResize, { passive: true });
    doc.addEventListener('mousemove', onMouseMove, { passive: true });
    doc.addEventListener('mouseleave', onMouseLeave, { passive: true });

    parentWin.__auroraCleanup = function() {
        if (rafId) parentWin.cancelAnimationFrame(rafId);
        if (retryTimer) parentWin.clearTimeout(retryTimer);
        if (mainNet) mainNet.disconnect();
        if (sideNet) sideNet.disconnect();
        if (metricObs) metricObs.disconnect();
        doc.removeEventListener('scroll', onScroll, { capture: true });
        parentWin.removeEventListener('scroll', onScroll);
        parentWin.removeEventListener('resize', onResize);
        doc.removeEventListener('mousemove', onMouseMove);
        doc.removeEventListener('mouseleave', onMouseLeave);
    };
})();
</script>
""", height=0)

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

def get_company_logo_svg(company_name: str) -> str:
    """Return a crisp inline SVG brand mark/monogram for company role card watermarks."""
    name = (company_name or "").strip().lower()
    if "google" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><path d="M108 61.2c0-3.5-.3-6.9-.9-10.2H60v19.3h27c-1.2 6.2-4.7 11.5-10 15v12.5h16.2c9.5-8.7 14.8-21.6 14.8-36.6z" fill="#4285F4"/><path d="M60 110c13.5 0 24.8-4.5 33.1-12.1L76.9 85.4c-4.5 3-10.2 4.8-16.9 4.8-13 0-24-8.8-27.9-20.6H15.4v12.9C23.8 99.1 40.7 110 60 110z" fill="#34A853"/><path d="M32.1 69.6c-1-3-1.6-6.2-1.6-9.6s.6-6.6 1.6-9.6V37.5H15.4C11.9 44.3 10 51.9 10 60s1.9 15.7 5.4 22.5l16.7-12.9z" fill="#FBBC05"/><path d="M60 29.8c7.3 0 13.9 2.5 19.1 7.5l14.3-14.3C84.8 14.9 73.5 10 60 10 40.7 10 23.8 20.9 15.4 37.5l16.7 12.9C36 38.6 47 29.8 60 29.8z" fill="#EA4335"/></svg>'
    if "microsoft" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><rect x="14" y="14" width="42" height="42" rx="4" fill="#F25022"/><rect x="64" y="14" width="42" height="42" rx="4" fill="#7FBA00"/><rect x="14" y="64" width="42" height="42" rx="4" fill="#00A4EF"/><rect x="64" y="64" width="42" height="42" rx="4" fill="#FFB900"/></svg>'
    if "amazon" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><text x="60" y="68" text-anchor="middle" font-family="Arial,sans-serif" font-weight="900" font-size="68" fill="#1E293B">a</text><path d="M22 82 C44 98 76 98 98 80" fill="none" stroke="#FF9900" stroke-width="7" stroke-linecap="round"/><path d="M88 76 L101 79 L95 92" fill="none" stroke="#FF9900" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    if "tcs" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><circle cx="60" cy="60" r="46" fill="none" stroke="#0076CE" stroke-width="6"/><path d="M26 70 Q60 22 94 70" fill="none" stroke="#E40046" stroke-width="6" stroke-linecap="round"/><text x="60" y="72" text-anchor="middle" font-family="Arial,sans-serif" font-weight="900" font-size="30" fill="#0076CE">TCS</text></svg>'
    if "infosys" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><polygon points="60,12 106,38 106,82 60,108 14,82 14,38" fill="none" stroke="#007CC3" stroke-width="7"/><text x="60" y="70" text-anchor="middle" font-family="Arial,sans-serif" font-weight="800" font-size="30" fill="#007CC3">Infy</text></svg>'
    if "accenture" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><path d="M28 24 L94 60 L28 96" fill="none" stroke="#A100FF" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    if "deloitte" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><text x="50" y="82" text-anchor="middle" font-family="Arial,sans-serif" font-weight="900" font-size="72" fill="#0F172A">D</text><circle cx="88" cy="76" r="11" fill="#86BC25"/></svg>'
    if "goldman" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><rect x="16" y="16" width="88" height="88" rx="14" fill="#5D87B8"/><text x="60" y="74" text-anchor="middle" font-family="Georgia,serif" font-weight="700" font-size="42" fill="#FFFFFF">GS</text></svg>'
    if "morgan" in name or "chase" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><polygon points="40,14 80,14 106,40 106,80 80,106 40,106 14,80 14,40" fill="none" stroke="#117ACA" stroke-width="12"/><rect x="44" y="44" width="32" height="32" rx="4" fill="#117ACA"/></svg>'
    if "zomato" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><rect x="14" y="14" width="92" height="92" rx="22" fill="#E23744"/><text x="60" y="80" text-anchor="middle" font-family="Arial,sans-serif" font-weight="900" font-size="64" fill="#FFFFFF">z</text></svg>'
    if "swiggy" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><path d="M60 12 C84 12 98 28 98 50 C98 74 60 108 60 108 C60 108 22 74 22 50 C22 28 36 12 60 12 Z" fill="#FC8019"/><text x="60" y="64" text-anchor="middle" font-family="Arial,sans-serif" font-weight="900" font-size="38" fill="#FFFFFF">S</text></svg>'
    if "cisco" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><g stroke="#049FD9" stroke-width="7" stroke-linecap="round"><line x1="20" y1="68" x2="20" y2="52"/><line x1="36" y1="74" x2="36" y2="40"/><line x1="52" y1="80" x2="52" y2="28"/><line x1="68" y1="80" x2="68" y2="28"/><line x1="84" y1="74" x2="84" y2="40"/><line x1="100" y1="68" x2="100" y2="52"/></g></svg>'
    if "wipro" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><circle cx="60" cy="60" r="38" fill="none" stroke="#3B82F6" stroke-width="6" stroke-dasharray="10 8"/><circle cx="60" cy="22" r="7" fill="#E11D48"/><circle cx="98" cy="60" r="7" fill="#10B981"/><circle cx="60" cy="98" r="7" fill="#F59E0B"/><circle cx="22" cy="60" r="7" fill="#8B5CF6"/><text x="60" y="67" text-anchor="middle" font-family="Arial,sans-serif" font-weight="800" font-size="22" fill="#1E3A8A">wipro</text></svg>'
    if "razorpay" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><polygon points="68,14 34,64 58,64 46,106 88,52 62,52" fill="#3395FF"/></svg>'
    if "cred" in name:
        return '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><path d="M24 20 H96 V66 C96 88 60 106 60 106 C60 106 24 88 24 66 Z" fill="none" stroke="#1E293B" stroke-width="8"/><text x="60" y="66" text-anchor="middle" font-family="Arial,sans-serif" font-weight="900" font-size="24" fill="#1E293B">CRED</text></svg>'
    words = [w[0].upper() for w in (company_name or "CO").split() if w]
    initials = "".join(words[:2]) or "CO"
    return f'<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><circle cx="60" cy="60" r="46" fill="none" stroke="#3B82F6" stroke-width="7"/><text x="60" y="73" text-anchor="middle" font-family="Arial,sans-serif" font-weight="900" font-size="38" fill="#1E3A8A">{initials}</text></svg>'

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

# Sidebar Live AI Engine & Preset Snapshot Card
_jobs_count = len(get_all_jobs())
_profiles_count = len(get_recent_profiles(100))
_champ_auc = predictor.bundle["metrics"]["roc_auc"]
st.sidebar.markdown(
    f"""
    <div class="sidebar-status-card">
        <div style="font-weight:700; color:#1E3A8A; margin-bottom:8px;">🧠 AI Engine &amp; DB Telemetry</div>
        <div class="sidebar-stat-row"><span>Champion Model</span><b>{predictor.bundle['model_name']}</b></div>
        <div class="sidebar-stat-row"><span>Validation ROC-AUC</span><b>{_champ_auc:.3f}</b></div>
        <div class="sidebar-stat-row"><span>Active Job Drives</span><b>{_jobs_count} Roles</b></div>
        <div class="sidebar-stat-row"><span>Saved Assessments</span><b>{_profiles_count} Profiles</b></div>
    </div>
    <div class="sidebar-status-card">
        <div style="font-weight:700; color:#047857; margin-bottom:8px;">⚡ Active Preset Snapshot</div>
        <div class="sidebar-stat-row"><span>Candidate</span><b>{default_name}</b></div>
        <div class="sidebar-stat-row"><span>Target CGPA</span><b>{default_cgpa:.1f} / 10</b></div>
        <div class="sidebar-stat-row"><span>Coding Score</span><b>{int(default_coding)} / 100</b></div>
        <div class="sidebar-stat-row"><span>Backlogs</span><b>{default_backlogs}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
        
        # 4-Pill Quick Profile Snapshot
        st.markdown(
            f"""
            <div class="kpi-mini-grid">
                <div class="kpi-mini-pill"><div class="kpi-mini-label">CGPA</div><div class="kpi-mini-val">{cgpa:.2f}</div></div>
                <div class="kpi-mini-pill"><div class="kpi-mini-label">Coding</div><div class="kpi-mini-val">{int(coding_score)}/100</div></div>
                <div class="kpi-mini-pill"><div class="kpi-mini-label">Internships</div><div class="kpi-mini-val">{internships}</div></div>
                <div class="kpi-mini-pill"><div class="kpi-mini-label">Backlogs</div><div class="kpi-mini-val" style="color:{'#047857' if backlogs == 0 else '#B91C1C'};">{backlogs}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
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
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Status Box with Risk-State Ambient Glow
        if prob >= 0.75:
            st.markdown(
                f"<div class='risk-banner risk-glow-high'><b>Status: {status}</b><br><br>"
                "The candidate meets high placement probability benchmarks across academics and technical evaluation.</div>",
                unsafe_allow_html=True,
            )
        elif prob >= 0.45:
            st.markdown(
                f"<div class='risk-banner risk-glow-moderate'><b>Status: {status}</b><br><br>"
                "The candidate has competitive baseline metrics but needs strengthening in key evaluated parameters.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='risk-banner risk-glow-low'><b>Status: {status}</b><br><br>"
                "High risk of campus placement rejection. Follow targeted guidance below to clear eligibility thresholds.</div>",
                unsafe_allow_html=True,
            )
            
        # Key Recommendations (Staggered Reveal)
        st.markdown("##### 📌 AI-Powered Personalized Recommendations:")
        for idx, rec in enumerate(result["recommendations"]):
            st.markdown(
                f"<div class='rec-item' style='animation-delay: {idx * 0.09:.2f}s;'>{rec}</div>",
                unsafe_allow_html=True,
            )

        # Profile Benchmark Radar Chart (Candidate vs Tier-1 Placed Benchmark)
        radar_cats = ["CGPA", "Coding Score", "Projects", "Internships", "Soft Skills"]
        cand_vals = [
            round((cgpa / 10.0) * 100, 1),
            round(coding_score, 1),
            round(min(projects / 5.0, 1.0) * 100, 1),
            round(min(internships / 3.0, 1.0) * 100, 1),
            round((soft_skills / 10.0) * 100, 1),
        ]
        bench_vals = [82.0, 80.0, 65.0, 67.0, 80.0]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=cand_vals + [cand_vals[0]],
            theta=radar_cats + [radar_cats[0]],
            fill="toself",
            name=name or "Candidate",
            line=dict(color="#3B82F6", width=2.5),
            fillcolor="rgba(59, 130, 246, 0.22)"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=bench_vals + [bench_vals[0]],
            theta=radar_cats + [radar_cats[0]],
            fill="toself",
            name="Placed Benchmark",
            line=dict(color="#10B981", width=2, dash="dash"),
            fillcolor="rgba(16, 185, 129, 0.12)"
        ))
        fig_radar.update_layout(
            title=dict(text="Candidate vs. Placed Benchmark (%)", font=dict(size=15, color="#1E293B")),
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)), bgcolor="rgba(255,255,255,0.45)"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
            height=290,
            margin=dict(l=35, r=35, t=45, b=45),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
            
        # Save to DB option
        if submit_btn:
            skills_str = ", ".join(skills_selected)
            saved_id = save_student_profile(
                name, f"{name.lower().replace(' ', '')}@college.edu", branch,
                cgpa, ssc, hsc, internships, projects, int(coding_score),
                int(soft_skills), backlogs, skills_str, round(prob, 4), status
            )
            if prob >= 0.75:
                st.balloons()
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
        eligible_jobs = [j for j in recommended_jobs if j["is_eligible"]]
        top_match = max(j["match_percentage"] for j in recommended_jobs)
        top_ctc = max(j["salary_lpa"] for j in recommended_jobs)
        avg_match = sum(j["match_percentage"] for j in recommended_jobs) / len(recommended_jobs)

        jm1, jm2, jm3, jm4 = st.columns(4)
        jm1.metric("Eligible Roles", f"{len(eligible_jobs)}/{len(recommended_jobs)}")
        jm2.metric("Highest Match", f"{top_match:.1f}%")
        jm3.metric("Top CTC Offered", f"₹ {top_ctc:.1f} LPA")
        jm4.metric("Avg Match Score", f"{avg_match:.1f}%")

        st.markdown(f"### 🎯 Found {len(recommended_jobs)} Recommended Roles")
        
        for idx, job in enumerate(recommended_jobs):
            with st.container(border=True):
                logo_svg = get_company_logo_svg(job["company_name"])
                st.markdown(f"<div class='role-bg-watermark' aria-hidden='true'>{logo_svg}</div>", unsafe_allow_html=True)
                # Card Header
                h_col1, h_col2 = st.columns([2.5, 1])
                with h_col1:
                    st.markdown(
                        f"<h3 class='job-card-anim' style='animation-delay: {idx * 0.08:.2f}s; margin-top: 0;'>"
                        f"{job['role_title']} &nbsp;•&nbsp; <span style='color:#2563EB;'>{job['company_name']}</span></h3>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Domain:** {job['department']} &nbsp;|&nbsp; **Location:** {job['location']} &nbsp;|&nbsp; **CTC:** `₹ {job['salary_lpa']} LPA`")
                with h_col2:
                    score = job['match_percentage']
                    badge_class = "badge-eligible" if job['is_eligible'] else "badge-ineligible"
                    st.markdown(f"<div style='text-align:right;'><span class='{badge_class}'>{job['eligibility_status']}</span></div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='match-bar-track'><div class='match-bar-fill' style='width: {score}%; animation-delay: {idx * 0.08 + 0.15:.2f}s;'></div></div>"
                        f"<div style='text-align:right; font-weight:700; color:#1E3A8A;'>{score}% Profile Match</div>",
                        unsafe_allow_html=True,
                    )
                
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

# -------------------------------------------------------------
# TAB 3: COMPANY & JOB OPENINGS DATABASE (DB)
# -------------------------------------------------------------
with tabs[2]:
    st.markdown("<div class='main-title'>Company & Job Openings Database</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Manage and explore real-time company recruitment drives stored in the SQLite database.</div>", unsafe_allow_html=True)
    
    df_all_jobs = get_all_jobs()

    db1, db2, db3, db4 = st.columns(4)
    db1.metric("Total Active Drives", len(df_all_jobs))
    db2.metric("Highest CTC", f"₹ {df_all_jobs['salary_lpa'].max():.1f} LPA")
    db3.metric("Average CTC", f"₹ {df_all_jobs['salary_lpa'].mean():.1f} LPA")
    db4.metric("Hiring Domains", df_all_jobs["department"].nunique())

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

    if not filtered_db_jobs.empty:
        fig_db = px.scatter(
            filtered_db_jobs,
            x="min_cgpa",
            y="salary_lpa",
            color="department",
            size="salary_lpa",
            hover_name="company_name",
            hover_data=["role_title", "location", "max_backlogs"],
            title="Company Compensation (LPA) vs. Minimum CGPA Cutoff",
            labels={"min_cgpa": "Minimum CGPA Cutoff", "salary_lpa": "CTC Package (LPA)", "department": "Domain"}
        )
        fig_db.update_layout(height=320, margin=dict(t=45, b=25, l=25, r=25), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.45)")
        st.plotly_chart(fig_db, use_container_width=True)
    
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
            with st.status("🚀 Running AI Batch Placement Engine...", expanded=True) as status_box:
                st.write("1. Validating & scaling student feature matrix...")
                probs = []
                preds = []
                statuses = []
                st.write(f"2. Executing ML inference across {len(batch_df)} student profiles...")
                for _, row in batch_df.iterrows():
                    res = predictor.predict(row.to_dict())
                    probs.append(round(res["placement_probability"] * 100, 2))
                    preds.append(res["prediction"])
                    statuses.append(res["status"])
                st.write("3. Aggregating batch placement readiness & risk distribution...")
                batch_df["Placement_Probability_%"] = probs
                batch_df["Predicted_Outcome"] = ["Placed" if p == 1 else "Unplaced" for p in preds]
                batch_df["Risk_Category"] = statuses
                status_box.update(label=f"✅ Batch Evaluation Complete ({len(batch_df)} Students)", state="complete", expanded=False)
                
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

import io
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos

class PlacementReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        # Top Banner Header
        self.set_fill_color(30, 58, 138)  # Deep Navy Blue
        self.rect(0, 0, 210, 22, 'F')
        
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 12)
        self.set_xy(10, 4)
        self.cell(190, 6, 'AI-POWERED PLACEMENT READINESS & RECOMMENDATION REPORT', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font('helvetica', '', 8)
        self.cell(190, 5, 'Campus Career Development & Automated Predictive Assessment System', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(6)

    def footer(self):
        self.set_y(-12)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 8, f'Page {self.page_no()}/{{nb}} | Confidential Student Assessment Report | Generated on {datetime.now().strftime("%d-%b-%Y %H:%M")}', align='C')


def clean_text(text: str) -> str:
    """Sanitize text to safe Latin-1 ASCII for core PDF fonts."""
    if not text:
        return ""
    replacements = {
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2022": "*", "\u2026": "...",
        "\u20b9": "INR ", "\u2713": "[v]", "\u2714": "[v]", "\u2716": "[x]",
        "\U0001f393": "", "\U0001f680": "", "\U0001f3af": "", "\U0001f4bc": "",
        "\U0001f3e2": "", "\U0001f4ca": "", "\U0001f465": "", "\U0001f31f": "",
        "\U0001f916": "", "\u26a0\ufe0f": "[!]", "\u26a0": "[!]", "\U0001f4dd": ""
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Filter to printable latin1
    return text.encode('latin-1', 'replace').decode('latin-1')


def generate_student_pdf(student_name: str, branch: str, profile_data: dict, skills_selected: list, prediction_result: dict, recommended_jobs: list) -> bytes:
    """
    Generates a student placement readiness & job recommendation PDF report.
    Returns:
        bytes: PDF binary content
    """
    pdf = PlacementReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # -------------------------------------------------------------
    # 1. CANDIDATE PROFILE HEADER CARD
    # -------------------------------------------------------------
    pdf.set_fill_color(248, 250, 252) # Soft light gray
    pdf.set_draw_color(226, 232, 240)
    pdf.rect(10, 26, 190, 24, 'DF')
    
    pdf.set_xy(14, 28)
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(110, 7, clean_text(student_name), new_x=XPos.RIGHT, new_y=YPos.TOP)
    
    # Date / Time
    pdf.set_font('helvetica', '', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(72, 7, f'Date: {datetime.now().strftime("%d %b %Y, %H:%M")}', align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_x(14)
    pdf.set_font('helvetica', '', 9)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(110, 6, f'Branch: {clean_text(branch)}', new_x=XPos.RIGHT, new_y=YPos.TOP)
    
    # Target Roles / Evaluation Type
    pdf.cell(72, 6, 'Assessment: Comprehensive Placement Evaluation', align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_y(54)
    
    # -------------------------------------------------------------
    # 2. PLACEMENT READINESS SCORE BANNER
    # -------------------------------------------------------------
    prob = prediction_result.get("placement_probability", 0.0)
    prob_percent = round(prob * 100, 1)
    status = prediction_result.get("status", "Evaluated")
    
    if prob >= 0.75:
        banner_bg = (220, 252, 231)  # Soft emerald
        banner_border = (16, 185, 129)
        banner_text = (22, 101, 52)
        score_badge = "HIGH PLACEMENT READINESS (TIER-1 CANDIDATE)"
    elif prob >= 0.45:
        banner_bg = (254, 243, 199)  # Soft amber
        banner_border = (245, 158, 11)
        banner_text = (180, 83, 9)
        score_badge = "MODERATE READINESS (NEEDS TARGETED POLISH)"
    else:
        banner_bg = (254, 226, 226)  # Soft red
        banner_border = (239, 68, 68)
        banner_text = (153, 27, 27)
        score_badge = "AT-RISK / REMEDIAL PREPARATION RECOMMENDED"
        
    pdf.set_fill_color(*banner_bg)
    pdf.set_draw_color(*banner_border)
    pdf.rect(10, 53, 190, 22, 'DF')
    
    pdf.set_xy(14, 55)
    pdf.set_font('helvetica', 'B', 16)
    pdf.set_text_color(*banner_text)
    pdf.cell(85, 9, f"Readiness Score: {prob_percent}%", new_x=XPos.RIGHT, new_y=YPos.TOP)
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(97, 9, score_badge, align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_x(14)
    pdf.set_font('helvetica', '', 8.5)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(182, 5, f"Status: {clean_text(status)} | Evaluated across 9 ML academic, coding, and achievement indicators.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_y(79)
    
    # -------------------------------------------------------------
    # 3. METRIC BREAKDOWN TABLE
    # -------------------------------------------------------------
    pdf.set_font('helvetica', 'B', 10.5)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "1. CANDIDATE PROFILE & COMPETENCY METRICS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    
    metrics_data = [
        [
            ("CGPA", f"{profile_data.get('cgpa', 0.0):.2f} / 10.0"),
            ("10th Standard", f"{profile_data.get('ssc_percentage', 0.0):.1f}%"),
            ("12th Standard", f"{profile_data.get('hsc_percentage', 0.0):.1f}%"),
            ("Current Backlogs", str(profile_data.get('backlogs', 0)))
        ],
        [
            ("Coding / DSA Score", f"{profile_data.get('coding_score', 0):.0f} / 100"),
            ("Soft Skills Rating", f"{profile_data.get('soft_skills_score', 0):.1f} / 10.0"),
            ("Major Projects", str(profile_data.get('projects', 0))),
            ("Internships", str(profile_data.get('internships', 0)))
        ],
        [
            ("Certifications", str(profile_data.get('certifications_count', 0))),
            ("GitHub Repositories", str(profile_data.get('github_repos', 0))),
            ("Hackathon Wins", str(profile_data.get('hackathons_count', 0))),
            ("Skill Count", str(len(skills_selected)))
        ]
    ]
    
    pdf.set_font('helvetica', '', 8.5)
    col_width = 47.5
    for row in metrics_data:
        # Header / Label row
        pdf.set_fill_color(241, 245, 249)
        pdf.set_draw_color(203, 213, 225)
        pdf.set_text_color(71, 85, 105)
        pdf.set_font('helvetica', 'B', 7.5)
        for label, _ in row:
            pdf.cell(col_width, 5, f" {label}", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(5)
        
        # Value row
        pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(15, 23, 42)
        pdf.set_font('helvetica', 'B', 8.5)
        for _, val in row:
            pdf.cell(col_width, 6, f" {val}", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(6)
        
    pdf.ln(2)
    
    # Skills Summary
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(32, 5, "Candidate Skills:", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font('helvetica', '', 8)
    pdf.set_text_color(15, 23, 42)
    skills_text = ", ".join(skills_selected) if skills_selected else "None listed"
    pdf.multi_cell(158, 5, clean_text(skills_text))
    pdf.ln(3)
    
    # -------------------------------------------------------------
    # 4. PERSONALIZED AI RECOMMENDATIONS
    # -------------------------------------------------------------
    pdf.set_font('helvetica', 'B', 10.5)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "2. PERSONALIZED AI RECOMMENDATIONS & ACTION PLAN", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    
    recommendations = prediction_result.get("recommendations", [])
    if not recommendations:
        recommendations = ["Continue maintaining consistent performance across algorithms and core subjects."]
        
    pdf.set_font('helvetica', '', 8.5)
    pdf.set_text_color(30, 41, 59)
    for rec in recommendations:
        pdf.set_x(12)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(5, 5, ">", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_text_color(30, 41, 59)
        pdf.multi_cell(173, 5, clean_text(rec))
        pdf.ln(1)
        
    pdf.ln(3)
    
    # -------------------------------------------------------------
    # 5. RECOMMENDED JOBS & SKILL GAP ANALYSIS
    # -------------------------------------------------------------
    # Check if page break is needed
    if pdf.get_y() > 195:
        pdf.add_page()
        
    pdf.set_font('helvetica', 'B', 10.5)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "3. TOP RECOMMENDED CAMPUS JOB ROLES & SKILL GAP ANALYSIS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    
    if not recommended_jobs:
        pdf.set_font('helvetica', 'I', 8.5)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No direct matching job openings found for current skill filter.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        # Display top jobs (up to 5)
        top_jobs = recommended_jobs[:5]
        
        # Table Header
        pdf.set_fill_color(30, 58, 138)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('helvetica', 'B', 8)
        
        pdf.cell(48, 6, " Company & Role", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(32, 6, " Domain", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(26, 6, " CTC Package", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(28, 6, " Match Score", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(56, 6, " Eligibility Status", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.set_font('helvetica', '', 7.5)
        for j_idx, job in enumerate(top_jobs):
            # Check page overflow
            if pdf.get_y() > 255:
                pdf.add_page()
                pdf.set_fill_color(30, 58, 138)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font('helvetica', 'B', 8)
                pdf.cell(48, 6, " Company & Role", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
                pdf.cell(32, 6, " Domain", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
                pdf.cell(26, 6, " CTC Package", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
                pdf.cell(28, 6, " Match Score", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
                pdf.cell(56, 6, " Eligibility Status", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font('helvetica', '', 7.5)
                
            bg_col = 255 if j_idx % 2 == 0 else 248
            pdf.set_fill_color(bg_col, bg_col, bg_col)
            pdf.set_draw_color(226, 232, 240)
            pdf.set_text_color(15, 23, 42)
            
            comp_role = f"{job['company_name']} - {job['role_title']}"
            if len(comp_role) > 30:
                comp_role = comp_role[:28] + ".."
            domain_txt = job.get('department', 'Tech')
            if len(domain_txt) > 20:
                domain_txt = domain_txt[:18] + ".."
            ctc_txt = f"INR {job['salary_lpa']} LPA"
            match_txt = f"{job['match_percentage']}% Match"
            elig_txt = job.get('eligibility_status', 'Eligible')
            
            pdf.cell(48, 6, f" {clean_text(comp_role)}", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.cell(32, 6, f" {clean_text(domain_txt)}", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.cell(26, 6, f" {ctc_txt}", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.cell(28, 6, f" {match_txt}", border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
            
            if job.get('is_eligible', True):
                pdf.set_text_color(22, 101, 52)
                pdf.set_font('helvetica', 'B', 7.5)
            else:
                pdf.set_text_color(153, 27, 27)
                pdf.set_font('helvetica', 'B', 7.5)
                
            pdf.cell(56, 6, f" {clean_text(elig_txt)}", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('helvetica', '', 7.5)
            pdf.set_text_color(15, 23, 42)
            
            # Sub-row for matched vs missing skills
            matched_s = ", ".join(job.get('matched_required_skills', [])) or "None direct"
            missing_s = ", ".join(job.get('missing_required_skills', [])) or "None (Full match!)"
            
            pdf.set_fill_color(248, 250, 252)
            pdf.set_font('helvetica', '', 7)
            pdf.set_text_color(71, 85, 105)
            sub_text = f"  * Matched: {matched_s}  |  * Skill Gaps to Bridge: {missing_s}"
            pdf.cell(190, 5, clean_text(sub_text[:130]), border='LRB', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
    # Output to bytes
    pdf_bytes = bytes(pdf.output())
    return pdf_bytes


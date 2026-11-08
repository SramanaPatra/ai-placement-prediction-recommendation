import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "placement_system.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Jobs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_name TEXT NOT NULL,
        role_title TEXT NOT NULL,
        department TEXT NOT NULL,
        min_cgpa REAL NOT NULL,
        max_backlogs INTEGER DEFAULT 0,
        required_skills TEXT NOT NULL,
        preferred_skills TEXT NOT NULL,
        salary_lpa REAL NOT NULL,
        location TEXT NOT NULL,
        job_type TEXT DEFAULT 'Full-Time',
        experience_level TEXT DEFAULT 'Entry Level',
        description TEXT
    )
    """)
    
    # 2. Student Submissions / History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS student_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        email TEXT,
        branch TEXT,
        cgpa REAL NOT NULL,
        ssc_marks REAL,
        hsc_marks REAL,
        internships_count INTEGER DEFAULT 0,
        projects_count INTEGER DEFAULT 0,
        coding_rating INTEGER DEFAULT 0,
        soft_skills_rating INTEGER DEFAULT 0,
        history_of_backlogs INTEGER DEFAULT 0,
        skills TEXT NOT NULL,
        placement_probability REAL,
        prediction_status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 3. Placement History / Benchmark Dataset Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS historical_placements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cgpa REAL,
        ssc_percentage REAL,
        hsc_percentage REAL,
        internships INTEGER,
        projects INTEGER,
        coding_score INTEGER,
        soft_skills_score INTEGER,
        certifications INTEGER,
        backlogs INTEGER,
        placed INTEGER
    )
    """)
    
    conn.commit()
    conn.close()

def insert_initial_jobs():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if jobs already exist
    cursor.execute("SELECT COUNT(*) as count FROM jobs")
    if cursor.fetchone()["count"] > 0:
        conn.close()
        return

    sample_jobs = [
        (
            "Google", "Associate Software Engineer", "Software Engineering", 8.0, 0,
            "Python, Data Structures, Algorithms, System Design, SQL",
            "C++, Cloud, Distributed Systems, Git",
            24.5, "Bengaluru / Hyderabad", "Full-Time", "Entry Level",
            "Develop scalable core web infrastructure and cloud-native services using clean architecture and algorithmic efficiency."
        ),
        (
            "Microsoft", "Software Engineer - Cloud & AI", "Cloud / Backend", 7.8, 0,
            "C#, Python, Data Structures, Algorithms, Azure",
            "Microservices, Docker, Kubernetes, SQL",
            21.0, "Hyderabad / Noida", "Full-Time", "Entry Level",
            "Build next-generation intelligent cloud services and APIs on Azure supporting millions of concurrent transactions."
        ),
        (
            "Amazon", "SDE-1", "Software Development", 7.5, 0,
            "Java, Object Oriented Programming, Data Structures, Algorithms, Problem Solving",
            "AWS, Spring Boot, NoSQL, Linux",
            20.0, "Bengaluru / Chennai", "Full-Time", "Entry Level",
            "Design highly available distributed services, e-commerce checkout systems, and real-time inventory management."
        ),
        (
            "TCS Innovation Labs", "Data Scientist / ML Engineer", "Artificial Intelligence", 7.0, 1,
            "Python, Machine Learning, Deep Learning, Pandas, Scikit-Learn, SQL",
            "TensorFlow, PyTorch, NLP, Computer Vision, MLOps",
            9.5, "Pune / Bengaluru", "Full-Time", "Entry Level",
            "Formulate predictive models, NLP analytics pipelines, and computer vision classification systems for enterprise customers."
        ),
        (
            "Infosys", "Specialist Programmer", "Application Development", 6.8, 1,
            "Java, Python, SQL, REST APIs, Git",
            "React, Spring Boot, Docker, Problem Solving",
            9.5, "Bengaluru / Mysuru", "Full-Time", "Entry Level",
            "Develop enterprise-scale microservices, web portals, and API integrations with robust test coverage."
        ),
        (
            "Accenture", "Associate Software Engineer", "IT Services", 6.0, 2,
            "C++, Java, SQL, Database Management, Problem Solving",
            "HTML, CSS, JavaScript, Git",
            4.5, "Hyderabad / Mumbai / Gurugram", "Full-Time", "Entry Level",
            "Support software lifecycle, client testing suites, database queries, and modular frontend/backend modules."
        ),
        (
            "Deloitte", "Cybersecurity Analyst", "Security & Risk", 6.5, 1,
            "Networking, Cybersecurity, Linux, Cryptography, Python",
            "Ethical Hacking, Wireshark, SIEM, Cloud Security",
            8.0, "Hyderabad / Bengaluru", "Full-Time", "Entry Level",
            "Assess vulnerability vectors, analyze incident reports, conduct penetration tests, and enforce secure IAM postures."
        ),
        (
            "Goldman Sachs", "Analyst - Quantitative & Tech", "FinTech", 8.2, 0,
            "Python, C++, Financial Mathematics, Data Structures, Algorithms, SQL",
            "Time Series, Pandas, Distributed Computing, Linux",
            26.0, "Bengaluru", "Full-Time", "Entry Level",
            "Implement high-frequency trading models, portfolio risk simulations, and secure financial pipeline architecture."
        ),
        (
            "J.P. Morgan Chase", "Software Engineer - FinTech", "Banking Technology", 7.5, 0,
            "Java, Spring Boot, SQL, Microservices, REST APIs",
            "React, Kafka, Docker, Kubernetes, AWS",
            16.5, "Mumbai / Bengaluru", "Full-Time", "Entry Level",
            "Build resilient payment gateway integrations, fraud anomaly detection systems, and scalable banking microservices."
        ),
        (
            "Zomato", "Full Stack Developer", "Product Engineering", 7.0, 1,
            "JavaScript, TypeScript, React, Node.js, MongoDB",
            "Next.js, Redis, Docker, System Design, Tailwind CSS",
            14.0, "Gurugram / Remote", "Full-Time", "Entry Level",
            "Create fast, snappy consumer applications, live order delivery tracking interfaces, and vendor management dashboards."
        ),
        (
            "Swiggy", "Backend Engineer", "Platform Engineering", 7.2, 0,
            "Golang, Python, Redis, Kafka, PostgreSQL, Microservices",
            "Docker, Kubernetes, AWS, System Design",
            15.5, "Bengaluru", "Full-Time", "Entry Level",
            "Optimize high-throughput order dispatch algorithms, geospatial routing, and real-time streaming pipelines."
        ),
        (
            "Cisco", "Network & Software Engineer", "Networking & Cloud", 7.0, 1,
            "C, C++, Computer Networks, Linux, Python",
            "TCP/IP, Socket Programming, Docker, Git",
            15.0, "Bengaluru", "Full-Time", "Entry Level",
            "Develop next-generation SDN controllers, secure routing switches, and high-performance packet processing engines."
        ),
        (
            "Wipro", "Project Engineer", "IT Infrastructure", 6.0, 2,
            "Java, Python, Database Management, Problem Solving, Communication Skills",
            "SQL, Cloud Basics, HTML/CSS",
            4.0, "Chennai / Bengaluru / Pune", "Full-Time", "Entry Level",
            "Contribute to multi-tenant client infrastructure, automated deployment checklists, and application support."
        ),
        (
            "Razorpay", "Frontend Engineer", "Payment Experience", 7.0, 1,
            "JavaScript, React, HTML5, CSS3, Redux, Web Performance",
            "TypeScript, Next.js, Webpack, Testing Library",
            16.0, "Bengaluru", "Full-Time", "Entry Level",
            "Craft seamless, accessible payment checkouts, merchant analytics portals, and intuitive financial workflows."
        ),
        (
            "Cred", "Data Analyst", "Analytics & Growth", 7.0, 1,
            "SQL, Python, Pandas, Tableau, Power BI, Statistics",
            "A/B Testing, Machine Learning, Excel, Data Visualization",
            12.0, "Bengaluru", "Full-Time", "Entry Level",
            "Transform transactional metrics into high-impact growth experiments, funnel retention graphs, and risk heatmaps."
        )
    ]
    
    cursor.executemany("""
    INSERT INTO jobs (
        company_name, role_title, department, min_cgpa, max_backlogs,
        required_skills, preferred_skills, salary_lpa, location, job_type, experience_level, description
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_jobs)
    
    conn.commit()
    conn.close()

def get_all_jobs():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM jobs", conn)
    conn.close()
    return df

def save_student_profile(name, email, branch, cgpa, ssc, hsc, internships, projects, coding, soft, backlogs, skills, prob, status):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO student_profiles (
        full_name, email, branch, cgpa, ssc_marks, hsc_marks,
        internships_count, projects_count, coding_rating, soft_skills_rating,
        history_of_backlogs, skills, placement_probability, prediction_status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, email, branch, cgpa, ssc, hsc, internships, projects, coding, soft, backlogs, skills, prob, status))
    conn.commit()
    profile_id = cursor.lastrowid
    conn.close()
    return profile_id

def get_recent_profiles(limit=10):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM student_profiles ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    return df

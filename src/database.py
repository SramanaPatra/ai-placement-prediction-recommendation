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

    # 4. Change 4: Student Applications / Shortlist Table
    # Tracks which job roles a student has saved/applied to from Tab 2.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS student_applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT NOT NULL,
        job_id INTEGER NOT NULL,
        company_name TEXT NOT NULL,
        role_title TEXT NOT NULL,
        department TEXT NOT NULL,
        salary_lpa REAL NOT NULL,
        match_percentage REAL DEFAULT 0.0,
        eligibility_status TEXT DEFAULT 'Unknown',
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
    )
    """)

    # Change 3: Migrate existing department names to canonical set.
    # This runs as idempotent UPDATEs so re-running init_db() is safe.
    _migrate_department_names(cursor)

    conn.commit()
    conn.close()


def _migrate_department_names(cursor):
    """Change 3: Standardize department column values to 6 canonical categories.

    Canonical set (matches the Tab 2 domain dropdown):
      - Software Engineering
      - Artificial Intelligence
      - Cloud / Backend
      - FinTech
      - Security & Risk
      - IT Services

    This UPDATE is intentionally idempotent so it is safe to run on
    every startup even if the rows are already migrated.
    """
    migrations = [
        # Software Engineering bucket
        ("Software Engineering",
         "department IN ('Software Development','Application Development',"
         "'Product Engineering','Platform Engineering')"),
        # Cloud / Backend bucket
        ("Cloud / Backend",
         "department IN ('Networking & Cloud')"),
        # FinTech bucket
        ("FinTech",
         "department IN ('Banking Technology')"),
        # IT Services bucket (catch-all for support/analytics/payments)
        ("IT Services",
         "department IN ('IT Infrastructure','Analytics & Growth','Payment Experience')"),
    ]
    for canonical, condition in migrations:
        cursor.execute(
            f"UPDATE jobs SET department = ? WHERE {condition}",
            (canonical,)
        )


def insert_initial_jobs():
    conn = get_connection()
    cursor = conn.cursor()

    # Check if jobs already exist
    cursor.execute("SELECT COUNT(*) as count FROM jobs")
    if cursor.fetchone()["count"] > 0:
        conn.close()
        return

    # Change 3: All department values use the 6 canonical categories from the start.
    # 15 original drives + 12 new drives (DevOps, Data Engineering, Mobile, Embedded).
    sample_jobs = [
        # ---- Original 15 drives (departments now canonical) ----
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
            "Amazon", "SDE-1", "Software Engineering", 7.5, 0,
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
            "Infosys", "Specialist Programmer", "Software Engineering", 6.8, 1,
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
            "J.P. Morgan Chase", "Software Engineer - FinTech", "FinTech", 7.5, 0,
            "Java, Spring Boot, SQL, Microservices, REST APIs",
            "React, Kafka, Docker, Kubernetes, AWS",
            16.5, "Mumbai / Bengaluru", "Full-Time", "Entry Level",
            "Build resilient payment gateway integrations, fraud anomaly detection systems, and scalable banking microservices."
        ),
        (
            "Zomato", "Full Stack Developer", "Software Engineering", 7.0, 1,
            "JavaScript, TypeScript, React, Node.js, MongoDB",
            "Next.js, Redis, Docker, System Design, Tailwind CSS",
            14.0, "Gurugram / Remote", "Full-Time", "Entry Level",
            "Create fast, snappy consumer applications, live order delivery tracking interfaces, and vendor management dashboards."
        ),
        (
            "Swiggy", "Backend Engineer", "Software Engineering", 7.2, 0,
            "Golang, Python, Redis, Kafka, PostgreSQL, Microservices",
            "Docker, Kubernetes, AWS, System Design",
            15.5, "Bengaluru", "Full-Time", "Entry Level",
            "Optimize high-throughput order dispatch algorithms, geospatial routing, and real-time streaming pipelines."
        ),
        (
            "Cisco", "Network & Software Engineer", "Cloud / Backend", 7.0, 1,
            "C, C++, Computer Networks, Linux, Python",
            "TCP/IP, Socket Programming, Docker, Git",
            15.0, "Bengaluru", "Full-Time", "Entry Level",
            "Develop next-generation SDN controllers, secure routing switches, and high-performance packet processing engines."
        ),
        (
            "Wipro", "Project Engineer", "IT Services", 6.0, 2,
            "Java, Python, Database Management, Problem Solving, Communication Skills",
            "SQL, Cloud Basics, HTML/CSS",
            4.0, "Chennai / Bengaluru / Pune", "Full-Time", "Entry Level",
            "Contribute to multi-tenant client infrastructure, automated deployment checklists, and application support."
        ),
        (
            "Razorpay", "Frontend Engineer", "IT Services", 7.0, 1,
            "JavaScript, React, HTML5, CSS3, Redux, Web Performance",
            "TypeScript, Next.js, Webpack, Testing Library",
            16.0, "Bengaluru", "Full-Time", "Entry Level",
            "Craft seamless, accessible payment checkouts, merchant analytics portals, and intuitive financial workflows."
        ),
        (
            "Cred", "Data Analyst", "IT Services", 7.0, 1,
            "SQL, Python, Pandas, Tableau, Power BI, Statistics",
            "A/B Testing, Machine Learning, Excel, Data Visualization",
            12.0, "Bengaluru", "Full-Time", "Entry Level",
            "Transform transactional metrics into high-impact growth experiments, funnel retention graphs, and risk heatmaps."
        ),
        # ---- Change 3: 12 New Drives (DevOps / Data Eng / Mobile / Embedded) ----
        (
            "GitLab", "DevOps Engineer", "Cloud / Backend", 7.0, 1,
            "Linux, Docker, Kubernetes, CI/CD, Git, Python",
            "Terraform, Ansible, Prometheus, Grafana, AWS",
            18.0, "Remote / Bengaluru", "Full-Time", "Entry Level",
            "Design and maintain CI/CD pipelines, container orchestration clusters, and GitOps-driven delivery workflows for globally distributed engineering teams."
        ),
        (
            "HashiCorp", "Site Reliability Engineer", "Cloud / Backend", 7.5, 0,
            "Linux, Kubernetes, Terraform, Go, Python, System Design",
            "Vault, Consul, Nomad, AWS, Prometheus",
            22.0, "Remote / Hyderabad", "Full-Time", "Entry Level",
            "Build reliability-first infrastructure-as-code tooling, automate incident response runbooks, and ensure five-nines SLA across cloud platforms."
        ),
        (
            "Databricks", "Data Engineer", "Artificial Intelligence", 7.2, 0,
            "Python, Apache Spark, SQL, Data Pipelines, Delta Lake",
            "Kafka, Airflow, Scala, AWS, Databricks Runtime",
            20.0, "Bengaluru / Remote", "Full-Time", "Entry Level",
            "Architect lakehouse ingestion pipelines, real-time streaming transformations, and petabyte-scale data processing workflows on the Databricks platform."
        ),
        (
            "Cloudera", "Big Data Engineer", "Artificial Intelligence", 7.0, 1,
            "Hadoop, Spark, Hive, Python, SQL, Data Warehousing",
            "Kafka, HBase, Impala, Scala, Linux",
            16.0, "Pune / Bengaluru", "Full-Time", "Entry Level",
            "Build fault-tolerant data ingestion, batch processing, and analytics pipelines on Hadoop and Spark ecosystems for Fortune 500 enterprises."
        ),
        (
            "Flipkart", "Android Developer", "Software Engineering", 7.0, 1,
            "Java, Kotlin, Android SDK, REST APIs, Git",
            "MVVM, Jetpack Compose, Room DB, Retrofit, Firebase",
            18.0, "Bengaluru", "Full-Time", "Entry Level",
            "Develop high-performance Android commerce features for India's largest e-retail platform serving 400 million+ customers."
        ),
        (
            "PhonePe", "iOS Developer", "Software Engineering", 7.0, 0,
            "Swift, Objective-C, iOS SDK, REST APIs, Xcode",
            "SwiftUI, CoreData, Firebase, Combine, Unit Testing",
            19.0, "Bengaluru", "Full-Time", "Entry Level",
            "Build secure, low-latency iOS payment flows, UPI integrations, and financial widgets for 500M+ PhonePe users."
        ),
        (
            "Texas Instruments", "Embedded Systems Engineer", "IT Services", 7.5, 0,
            "C, C++, Embedded C, Microcontrollers, RTOS, Circuit Design",
            "ARM Cortex, I2C, SPI, UART, CAN Bus, Python",
            12.0, "Bengaluru / Dallas (India R&D)", "Full-Time", "Entry Level",
            "Design firmware for mixed-signal microcontrollers, power-management ICs, and real-time sensor interfaces used in industrial and automotive electronics."
        ),
        (
            "Bosch", "Firmware / IoT Engineer", "IT Services", 7.0, 1,
            "C, Embedded C, RTOS, IoT Protocols, Linux, Python",
            "MQTT, BLE, Zephyr RTOS, AWS IoT, Circuit Debugging",
            11.0, "Bengaluru / Coimbatore", "Full-Time", "Entry Level",
            "Develop connected-device firmware, OTA update stacks, and cloud-bridge software for Bosch smart home and industrial IoT platforms."
        ),
        (
            "ThoughtWorks", "Full Stack Developer", "Software Engineering", 7.0, 1,
            "Java, Python, JavaScript, React, REST APIs, Git, Agile",
            "Spring Boot, Node.js, Docker, TDD, CI/CD",
            14.0, "Bengaluru / Pune / Hyderabad", "Full-Time", "Entry Level",
            "Deliver iterative full-stack solutions across diverse client domains using extreme programming, pair coding, and continuous delivery practices."
        ),
        (
            "Atlassian", "Backend Engineer", "Software Engineering", 7.5, 0,
            "Java, Python, Microservices, REST APIs, SQL, System Design",
            "Kafka, AWS, Docker, Kubernetes, Spring Boot",
            22.0, "Bengaluru / Remote", "Full-Time", "Entry Level",
            "Build scalable backend services powering Jira, Confluence, and Bitbucket SaaS platforms used by 10M+ developers worldwide."
        ),
        (
            "Freshworks", "SDE-1", "Software Engineering", 6.8, 1,
            "Java, Python, SQL, REST APIs, Data Structures, Algorithms",
            "React, Node.js, Redis, Docker, Git",
            12.0, "Chennai / Bengaluru", "Full-Time", "Entry Level",
            "Build delightful CRM, helpdesk, and customer-engagement product features across Freshworks SaaS suite serving 60K+ global businesses."
        ),
        (
            "Zepto", "Mobile Engineer", "Software Engineering", 7.0, 1,
            "React Native, JavaScript, TypeScript, REST APIs, Git",
            "Android, iOS, Redux, Firebase, Performance Optimization",
            17.0, "Mumbai / Bengaluru", "Full-Time", "Entry Level",
            "Engineer high-speed, 10-minute grocery delivery mobile experiences with sub-second navigation, real-time tracking, and seamless payment flows."
        ),
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


def add_job_drive(
    company_name,
    role_title,
    department,
    min_cgpa,
    max_backlogs,
    required_skills,
    preferred_skills,
    salary_lpa,
    location,
    description,
):
    """Insert a new job drive if (company_name, role_title) does not already exist."""
    clean_company = company_name.strip()
    clean_role = role_title.strip()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM jobs WHERE LOWER(TRIM(company_name)) = LOWER(?) AND LOWER(TRIM(role_title)) = LOWER(?)",
        (clean_company, clean_role),
    )
    if cursor.fetchone() is not None:
        conn.close()
        return False, None

    cursor.execute(
        """
        INSERT INTO jobs (
            company_name, role_title, department, min_cgpa, max_backlogs,
            required_skills, preferred_skills, salary_lpa, location, description
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            clean_company,
            clean_role,
            department,
            min_cgpa,
            max_backlogs,
            required_skills.strip(),
            (preferred_skills or "").strip(),
            salary_lpa,
            location.strip(),
            (description or "").strip(),
        ),
    )
    conn.commit()
    job_id = cursor.lastrowid
    conn.close()
    return True, job_id


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
    """Return the N most-recently saved student_profiles rows."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM student_profiles ORDER BY id DESC LIMIT ?", conn, params=(limit,)
    )
    conn.close()
    return df


# Change 5: Additional helpers for the Assessment History tab.

def get_all_profiles():
    """Return all saved student_profiles rows, newest first."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM student_profiles ORDER BY id DESC", conn
    )
    conn.close()
    return df


def get_profiles_paginated(page: int = 0, page_size: int = 20):
    """Return a page of student_profiles rows for paginated display."""
    offset = page * page_size
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM student_profiles ORDER BY id DESC LIMIT ? OFFSET ?",
        conn,
        params=(page_size, offset),
    )
    total_count = pd.read_sql_query(
        "SELECT COUNT(*) AS cnt FROM student_profiles", conn
    )["cnt"].iloc[0]
    conn.close()
    return df, int(total_count)


def delete_profile(profile_id: int) -> bool:
    """Delete a student_profiles row by id. Returns True if a row was deleted."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM student_profiles WHERE id = ?", (profile_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# Change 4: Application / Shortlist helpers.

def save_application(
    student_name: str,
    job_id: int,
    company_name: str,
    role_title: str,
    department: str,
    salary_lpa: float,
    match_percentage: float,
    eligibility_status: str,
) -> int:
    """Save a job shortlist entry for a student.

    Returns the new application id, or -1 if the (student_name, job_id)
    pair already exists (prevent duplicate saves).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM student_applications WHERE student_name = ? AND job_id = ?",
        (student_name.strip(), job_id),
    )
    if cursor.fetchone() is not None:
        conn.close()
        return -1  # Already saved

    cursor.execute(
        """
        INSERT INTO student_applications (
            student_name, job_id, company_name, role_title, department,
            salary_lpa, match_percentage, eligibility_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            student_name.strip(),
            job_id,
            company_name,
            role_title,
            department,
            salary_lpa,
            match_percentage,
            eligibility_status,
        ),
    )
    conn.commit()
    app_id = cursor.lastrowid
    conn.close()
    return app_id


def get_applications(student_name: str = None) -> pd.DataFrame:
    """Return shortlisted applications.

    If student_name is provided, filters to that student only.
    Returns columns: id, student_name, job_id, company_name, role_title,
    department, salary_lpa, match_percentage, eligibility_status, applied_at.
    """
    conn = get_connection()
    if student_name:
        df = pd.read_sql_query(
            "SELECT * FROM student_applications WHERE student_name = ? ORDER BY applied_at DESC",
            conn,
            params=(student_name.strip(),),
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM student_applications ORDER BY applied_at DESC", conn
        )
    conn.close()
    return df


def delete_application(app_id: int) -> bool:
    """Remove a shortlisted application by id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM student_applications WHERE id = ?", (app_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted

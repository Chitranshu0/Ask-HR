#!/usr/bin/env python3
"""Synthetic HR knowledge base generator for Feku Tech Solutions Pvt. Ltd.

Creates CSV datasets and markdown HR documents for RAG ingestion.
"""
from __future__ import annotations
import random
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from faker import Faker

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"

random.seed(42)
F = Faker("en_IN")
F.seed_instance(42)

COMPANY = {
    "name": "Feku Tech Solutions Pvt. Ltd.",
    "headquarters": "Bhubaneswar, Odisha, India",
    "founded": 2021,
    "ceo": "Chitranshu Sanket",
}

DEPARTMENTS = [
    "Leadership",
    "Human Resources",
    "Finance",
    "AI & Machine Learning",
    "Generative AI",
    "Data Engineering",
    "Backend Engineering",
    "MERN Development",
    "Cloud & DevOps",
    "QA",
    "Sales & Marketing",
    "Customer Success",
]

ROLES = {
    "Leadership": ["CEO", "CTO", "HR Director"],
    "Human Resources": ["HR Executive", "Talent Acquisition Specialist"],
    "Finance": ["Finance Manager", "Accountant"],
    "AI & Machine Learning": [
        "AI Team Lead",
        "AI Engineer",
        "AI Engineer",
        "Machine Learning Engineer",
        "Data Scientist",
        "MLOps Engineer",
    ],
    "Generative AI": [
        "GenAI Lead",
        "GenAI Engineer",
        "GenAI Engineer",
        "Prompt Engineer",
    ],
    "Data Engineering": ["Data Engineering Lead", "Data Engineer", "Data Engineer"],
    "Backend Engineering": ["Backend Lead", "Python Backend Developer", "Java Backend Developer"],
    "MERN Development": ["MERN Lead", "MERN Stack Developer", "Frontend React Developer"],
    "Cloud & DevOps": ["DevOps Lead", "DevOps Engineer", "Cloud Engineer"],
    "QA": ["QA Lead", "QA Engineer"],
    "Sales & Marketing": ["Sales Manager", "Business Development Executive", "Marketing Executive"],
    "Customer Success": ["Customer Success Manager", "Customer Success Executive"],
}

SKILL_POOLS = {
    "AI/ML": [
        "Python",
        "TensorFlow",
        "PyTorch",
        "LangChain",
        "LangGraph",
        "Hugging Face",
        "OpenAI API",
    ],
    "Data Engineering": ["Spark", "Airflow", "Kafka", "SQL", "PostgreSQL"],
    "Backend": ["FastAPI", "Django", "Spring Boot", "Java", "REST APIs"],
    "Frontend": ["React", "Next.js", "TypeScript"],
    "Cloud/DevOps": ["AWS", "Azure", "Docker", "Kubernetes", "Terraform", "CI/CD"],
}

PROJECTS = [
    {
        "project_id": "P001",
        "name": "AskHR",
        "department": "AI & Generative, Backend",
        "description": "Agentic AI HR Assistant using LangGraph, RAG, FastAPI, PostgreSQL, and React.",
        "status": "Active",
    },
    {
        "project_id": "P002",
        "name": "MedAssist AI",
        "department": "AI & ML",
        "description": "Healthcare AI platform for medical report analysis and patient assistance.",
        "status": "Active",
    },
    {
        "project_id": "P003",
        "name": "CloudOps360",
        "department": "Cloud & DevOps",
        "description": "Cloud monitoring and infrastructure automation platform.",
        "status": "Active",
    },
    {
        "project_id": "P004",
        "name": "RetailIQ",
        "department": "Data Engineering, MERN",
        "description": "Retail analytics dashboard for inventory forecasting and sales intelligence.",
        "status": "Active",
    },
]

BENEFITS = [
    {"benefit": "Health Insurance", "details": "Group health coverage for employees and dependents."},
    {"benefit": "Provident Fund", "details": "Provident fund contributions as per Indian law."},
    {"benefit": "Performance Bonus", "details": "Annual performance-linked bonus."},
    {"benefit": "Learning Allowance", "details": "Budget for professional development and courses."},
]

HOLIDAYS = [
    {"date": "2026-01-26", "name": "Republic Day"},
    {"date": "2026-03-25", "name": "Holi"},
    {"date": "2026-04-14", "name": "Dr. B.R. Ambedkar Jayanti"},
    {"date": "2026-05-01", "name": "Labor Day"},
    {"date": "2026-08-15", "name": "Independence Day"},
    {"date": "2026-10-02", "name": "Gandhi Jayanti"},
    {"date": "2026-11-04", "name": "Diwali"},
    {"date": "2026-12-25", "name": "Christmas Day"},
]


def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)


def gen_employee_id(n: int) -> str:
    return f"E{n:04d}"


def salary_for_role(role: str) -> str:
    if "Lead" in role or "Director" in role or role == "CEO":
        return "30L-45L"
    if "Manager" in role:
        return "12L-20L"
    if "Engineer" in role or "Developer" in role or "Data Scientist" in role:
        return "8L-18L"
    if "Executive" in role or "Accountant" in role or "QA" in role:
        return "4L-8L"
    return "4L-12L"


def random_skills_for(role: str) -> list[str]:
    pool = []
    if any(k in role for k in ["AI", "Machine", "Data Scientist", "MLOps", "GenAI", "Prompt"]):
        pool += SKILL_POOLS["AI/ML"] + SKILL_POOLS["Data Engineering"]
    if any(k in role for k in ["Backend", "Developer", "Java", "Python"]):
        pool += SKILL_POOLS["Backend"]
    if any(k in role for k in ["MERN", "Frontend", "React"]):
        pool += SKILL_POOLS["Frontend"]
    if any(k in role for k in ["DevOps", "Cloud"]):
        pool += SKILL_POOLS["Cloud/DevOps"]
    pool = list(dict.fromkeys(pool))
    n = random.randint(2, min(6, max(2, len(pool))))
    return random.sample(pool, n) if pool else random.sample(sum(SKILL_POOLS.values(), []), 3)


def generate_employees() -> pd.DataFrame:
    rows = []
    idx = 1
    ceo = {
        "employee_id": gen_employee_id(idx),
        "full_name": COMPANY["ceo"],
        "gender": "Male",
        "age": 38,
        "email": "chitranshu.sanket@fekutech.com",
        "phone_number": F.phone_number(),
        "address": F.address().replace('\n', ', '),
        "department": "Leadership",
        "designation": "CEO",
        "reporting_manager": "",
        "joining_date": "2021-01-15",
        "employment_type": "Full-time",
        "work_location": "Bhubaneswar",
        "salary_band": "45L-60L",
        "skills": "Leadership, Strategy",
        "emergency_contact": F.name(),
    }
    rows.append(ceo)
    idx += 1

    for dept in ROLES:
        for role in ROLES[dept]:
            if dept == "Leadership" and role == "CEO":
                continue
            if len(rows) >= 33:
                break
            name = F.name()
            gender = random.choice(["Male", "Female", "Other"])
            age = random.randint(24, 50)
            email = f"{name.lower().replace(' ', '.')}@fekutech.com"
            phone = F.phone_number()
            address = F.address().replace('\n', ', ')
            manager = "Chitranshu Sanket" if dept == "Leadership" else ""
            if "Lead" in role or "Manager" in role or "Director" in role:
                manager = "Chitranshu Sanket"
            row = {
                "employee_id": gen_employee_id(idx),
                "full_name": name,
                "gender": gender,
                "age": age,
                "email": email,
                "phone_number": phone,
                "address": address,
                "department": dept,
                "designation": role,
                "reporting_manager": manager,
                "joining_date": (datetime.now() - timedelta(days=random.randint(30, 1500))).date().isoformat(),
                "employment_type": random.choices(["Full-time", "Contract"], [0.9, 0.1])[0],
                "work_location": random.choice(["Bhubaneswar", "Remote"]),
                "salary_band": salary_for_role(role),
                "skills": ", ".join(random_skills_for(role)),
                "emergency_contact": F.name(),
            }
            rows.append(row)
            idx += 1
        if len(rows) >= 33:
            break

    while len(rows) < 33:
        name = F.name()
        role = random.choice([r for roles in ROLES.values() for r in roles])
        dept = next((d for d, rs in ROLES.items() if role in rs), "Engineering")
        rows.append(
            {
                "employee_id": gen_employee_id(len(rows) + 1),
                "full_name": name,
                "gender": random.choice(["Male", "Female"]),
                "age": random.randint(23, 45),
                "email": f"{name.lower().replace(' ', '.')}@fekutech.com",
                "phone_number": F.phone_number(),
                "address": F.address().replace('\n', ', '),
                "department": dept,
                "designation": role,
                "reporting_manager": "Chitranshu Sanket",
                "joining_date": (datetime.now() - timedelta(days=random.randint(30, 1200))).date().isoformat(),
                "employment_type": "Full-time",
                "work_location": random.choice(["Bhubaneswar", "Remote"]),
                "salary_band": salary_for_role(role),
                "skills": ", ".join(random_skills_for(role)),
                "emergency_contact": F.name(),
            }
        )

    df = pd.DataFrame(rows)

    for dept in ROLES:
        leads = df[(df.department == dept) & df.designation.str.contains("Lead|Manager|Director")]
        if not leads.empty:
            lead_name = leads.iloc[0]["full_name"]
        else:
            lead_name = COMPANY["ceo"]
        df.loc[df.department == dept, "reporting_manager"] = df.loc[
            (df.department == dept) & df.designation.str.contains("Lead|Manager|Director"), "full_name"
        ].ffill().bfill().iloc[0] if not df.loc[(df.department == dept) & df.designation.str.contains("Lead|Manager|Director")].empty else lead_name

    df.loc[df.designation == "CEO", "reporting_manager"] = ""

    return df


def generate_departments() -> pd.DataFrame:
    rows = []
    for d in DEPARTMENTS:
        rows.append({"department": d, "headquarters": COMPANY["headquarters"]})
    return pd.DataFrame(rows)


def generate_projects(employees_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in PROJECTS:
        candidates = employees_df[employees_df.department.str.contains(p["department"].split(",")[0].strip(), na=False)]
        if candidates.empty:
            manager = COMPANY["ceo"]
        else:
            manager = candidates.sample(1).iloc[0]["full_name"]
        rows.append({
            "project_id": p["project_id"],
            "name": p["name"],
            "department": p["department"],
            "description": p["description"],
            "status": p["status"],
            "project_manager": manager,
        })
    return pd.DataFrame(rows)


def generate_project_assignments(employees_df: pd.DataFrame, projects_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, emp in employees_df.iterrows():
        assigned = random.choices(projects_df.project_id.tolist(), k=random.choice([0, 1, 1, 2]))
        for pid in set(assigned):
            rows.append({
                "project_id": pid,
                "employee_id": emp.employee_id,
                "role": emp.designation,
                "allocation_pct": random.choice([25, 50, 75, 100]),
            })
    return pd.DataFrame(rows)


def generate_leave_balances(employees_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, emp in employees_df.iterrows():
        rows.append({
            "employee_id": emp.employee_id,
            "casual_leave_balance": random.randint(2, 10),
            "sick_leave_balance": random.randint(2, 10),
            "earned_leave_balance": random.randint(6, 24),
        })
    return pd.DataFrame(rows)


def generate_leave_requests(employees_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    statuses = ["Approved", "Rejected", "Pending"]
    leave_types = ["Casual Leave", "Sick Leave", "Earned Leave"]
    for _ in range(120):
        emp = employees_df.sample(1).iloc[0]
        start = datetime.now() - timedelta(days=random.randint(1, 400))
        days = random.randint(1, 5)
        status = random.choices(statuses, [0.7, 0.1, 0.2])[0]
        rows.append({
            "request_id": f"LR{random.randint(1000,9999)}",
            "employee_id": emp.employee_id,
            "leave_type": random.choice(leave_types),
            "start_date": start.date().isoformat(),
            "end_date": (start + timedelta(days=days)).date().isoformat(),
            "days": days,
            "status": status,
            "applied_on": (start - timedelta(days=random.randint(1, 10))).date().isoformat(),
        })
    return pd.DataFrame(rows)


def generate_attendance(employees_df: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    end = datetime.now().date()
    start = end - pd.DateOffset(months=months)
    dates = pd.bdate_range(start=start, end=end)
    rows = []
    for _, emp in employees_df.iterrows():
        for d in dates:
            status = random.choices(["Present", "Absent", "WFH"], [0.88, 0.05, 0.07])[0]
            if status == "Present" or status == "WFH":
                check_in = datetime.combine(d, datetime.min.time()) + timedelta(hours=9, minutes=random.randint(0, 50))
                check_out = check_in + timedelta(hours=8 + random.choice([0, 0, 1]), minutes=random.randint(0, 60))
                work_hours = round((check_out - check_in).seconds / 3600, 2)
                rows.append({
                    "employee_id": emp.employee_id,
                    "date": d.date().isoformat(),
                    "check_in": check_in.time().isoformat(timespec='minutes'),
                    "check_out": check_out.time().isoformat(timespec='minutes'),
                    "work_hours": work_hours,
                    "attendance_status": status,
                })
            else:
                rows.append({
                    "employee_id": emp.employee_id,
                    "date": d.date().isoformat(),
                    "check_in": "",
                    "check_out": "",
                    "work_hours": 0.0,
                    "attendance_status": "Absent",
                })
    return pd.DataFrame(rows)


def generate_assets(employees_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, emp in employees_df.iterrows():
        rows.append({"employee_id": emp.employee_id, "asset_type": "Laptop", "asset_tag": f"LAP-{random.randint(1000,9999)}"})
        rows.append({"employee_id": emp.employee_id, "asset_type": "VPN Access", "asset_tag": f"VPN-{random.randint(100,999)}"})
        if random.random() > 0.3:
            rows.append({"employee_id": emp.employee_id, "asset_type": "Monitor", "asset_tag": f"MON-{random.randint(1000,9999)}"})
        rows.append({"employee_id": emp.employee_id, "asset_type": "Software License", "asset_tag": random.choice(["MS365", "JetBrains", "Adobe"])})
    return pd.DataFrame(rows)


def generate_performance_reviews(employees_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, emp in employees_df.iterrows():
        for year in [2024, 2025]:
            reviewer = employees_df.sample(1).iloc[0]["full_name"]
            rating = random.choice([3, 4, 4, 5, 2])
            strengths = ", ".join(random.sample(emp.skills.split(", "), min(2, len(emp.skills.split(", "))))) if emp.skills else ""
            rows.append({
                "employee_id": emp.employee_id,
                "reviewer": reviewer,
                "review_period": f"{year}-01-01 to {year}-12-31",
                "rating": rating,
                "strengths": strengths,
                "improvement_areas": "Communication, Time management" if rating < 4 else "System design, Mentoring",
            })
    return pd.DataFrame(rows)


def generate_training_records(employees_df: pd.DataFrame) -> pd.DataFrame:
    trainings = [
        "LangGraph Workshop",
        "Advanced PyTorch",
        "Kubernetes for Developers",
        "Data Pipeline Design with Airflow",
        "Effective Code Reviews",
    ]
    rows = []
    for _ in range(80):
        emp = employees_df.sample(1).iloc[0]
        rows.append({
            "employee_id": emp.employee_id,
            "training": random.choice(trainings),
            "provider": random.choice(["Coursera", "Udemy", "Internal"]),
            "completed_on": (datetime.now() - timedelta(days=random.randint(1, 800))).date().isoformat(),
        })
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, name: str):
    path = DATA_DIR / name
    df.to_csv(path, index=False)


def write_markdown(filename: str, content: str):
    path = DOCS_DIR / filename
    path.write_text(content, encoding="utf-8")


def create_hr_documents():
    docs = {
        "employee_handbook.md": f"# Employee Handbook - {COMPANY['name']}\n\nWelcome to {COMPANY['name']}.\n",
        "leave_policy.md": "# Leave Policy\n\nCasual, Sick, and Earned Leave policies with carryover rules.",
        "attendance_policy.md": "# Attendance Policy\n\nCheck-in/out, WFH rules, and punctuality.",
        "work_from_home_policy.md": "# Work From Home Policy\n\nHybrid work and expectations.",
        "reimbursement_policy.md": "# Reimbursement Policy\n\nApproved expense types, submission process.",
        "code_of_conduct.md": "# Code of Conduct\n\nProfessional behaviour and ethics.",
        "travel_policy.md": "# Travel Policy\n\nBusiness travel booking and approvals.",
        "benefits_policy.md": "# Benefits Policy\n\nHealth insurance, PF, bonuses.",
        "promotion_policy.md": "# Promotion Policy\n\nCriteria for promotion and review cycles.",
        "performance_review_policy.md": "# Performance Review Policy\n\nAnnual review cycles and rating scale.",
        "onboarding_guide.md": "# Onboarding Guide\n\nFirst-day checklist and access provisioning.",
        "faqs.md": "# FAQs\n\nCommon HR questions and answers.",
        "company_announcements.md": "# Company Announcements\n\nRecent major announcements and updates.",
    }
    for fname, content in docs.items():
        write_markdown(fname, content)


def main():
    ensure_dirs()
    employees = generate_employees()
    departments = generate_departments()
    projects = generate_projects(employees)
    project_assignments = generate_project_assignments(employees, projects)
    leave_balances = generate_leave_balances(employees)
    leave_requests = generate_leave_requests(employees)
    attendance = generate_attendance(employees)
    assets = generate_assets(employees)
    performance_reviews = generate_performance_reviews(employees)
    training = generate_training_records(employees)

    write_csv(employees, "employees.csv")
    write_csv(departments, "departments.csv")
    write_csv(leave_balances, "leave_balances.csv")
    write_csv(leave_requests, "leave_requests.csv")
    write_csv(attendance, "attendance.csv")
    write_csv(projects, "projects.csv")
    write_csv(project_assignments, "project_assignments.csv")
    skills_rows = []
    for _, r in employees.iterrows():
        for s in [x.strip() for x in r.skills.split(",") if x.strip()]:
            skills_rows.append({"employee_id": r.employee_id, "skill": s})
    write_csv(pd.DataFrame(skills_rows), "skills.csv")
    write_csv(assets, "assets.csv")
    write_csv(pd.DataFrame(BENEFITS), "benefits.csv")
    write_csv(pd.DataFrame(HOLIDAYS), "holidays.csv")
    write_csv(performance_reviews, "performance_reviews.csv")
    write_csv(training, "training_records.csv")

    create_hr_documents()

    print("Generated HR knowledge base under:", DATA_DIR, DOCS_DIR)


if __name__ == "__main__":
    main()

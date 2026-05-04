"""
Database module for DR Screening Dashboard.
Uses SQLite for patient records and screening results.
"""

import sqlite3
import json
import os
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dr_screening.db")


def get_db():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            contact TEXT,
            diabetes_duration INTEGER DEFAULT 0,
            medical_history TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            image_path TEXT NOT NULL,
            severity INTEGER NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT DEFAULT '[]',
            doctor_notes TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE SET NULL
        )
    """)

    conn.commit()
    conn.close()


# ──────────── Patient CRUD ────────────

def add_patient(name, age, gender, contact="", diabetes_duration=0, medical_history=""):
    """Add a new patient. Returns the patient ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO patients (name, age, gender, contact, diabetes_duration, medical_history)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (name, age, gender, contact, diabetes_duration, medical_history)
    )
    conn.commit()
    patient_id = cursor.lastrowid
    conn.close()
    return patient_id


def get_all_patients(search=""):
    """Get all patients, optionally filtered by search term."""
    conn = get_db()
    if search:
        patients = conn.execute(
            "SELECT * FROM patients WHERE name LIKE ? OR contact LIKE ? ORDER BY created_at DESC",
            (f"%{search}%", f"%{search}%")
        ).fetchall()
    else:
        patients = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(p) for p in patients]


def get_patient(patient_id):
    """Get a single patient by ID."""
    conn = get_db()
    patient = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    conn.close()
    return dict(patient) if patient else None


def update_patient(patient_id, name, age, gender, contact="", diabetes_duration=0, medical_history=""):
    """Update an existing patient."""
    conn = get_db()
    conn.execute(
        """UPDATE patients SET name=?, age=?, gender=?, contact=?, 
           diabetes_duration=?, medical_history=? WHERE id=?""",
        (name, age, gender, contact, diabetes_duration, medical_history, patient_id)
    )
    conn.commit()
    conn.close()


def delete_patient(patient_id):
    """Delete a patient by ID."""
    conn = get_db()
    conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    conn.commit()
    conn.close()


# ──────────── Screening CRUD ────────────

def add_screening(patient_id, image_path, severity, label, confidence, probabilities, doctor_notes=""):
    """Add a screening result. Returns the screening ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO screenings (patient_id, image_path, severity, label, confidence, probabilities, doctor_notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (patient_id, image_path, severity, label, confidence, json.dumps(probabilities), doctor_notes)
    )
    conn.commit()
    screening_id = cursor.lastrowid
    conn.close()
    return screening_id


def get_screening(screening_id):
    """Get a single screening by ID with patient info."""
    conn = get_db()
    screening = conn.execute(
        """SELECT s.*, p.name as patient_name, p.age as patient_age, 
                  p.gender as patient_gender, p.contact as patient_contact,
                  p.diabetes_duration, p.medical_history
           FROM screenings s 
           LEFT JOIN patients p ON s.patient_id = p.id 
           WHERE s.id = ?""",
        (screening_id,)
    ).fetchone()
    conn.close()
    if screening:
        result = dict(screening)
        result["probabilities"] = json.loads(result["probabilities"])
        return result
    return None


def get_patient_screenings(patient_id):
    """Get all screenings for a patient."""
    conn = get_db()
    screenings = conn.execute(
        "SELECT * FROM screenings WHERE patient_id = ? ORDER BY created_at DESC",
        (patient_id,)
    ).fetchall()
    conn.close()
    results = []
    for s in screenings:
        d = dict(s)
        d["probabilities"] = json.loads(d["probabilities"])
        results.append(d)
    return results


def get_all_screenings(limit=None):
    """Get all screenings with patient info."""
    conn = get_db()
    query = """
        SELECT s.*, p.name as patient_name 
        FROM screenings s 
        LEFT JOIN patients p ON s.patient_id = p.id 
        ORDER BY s.created_at DESC
    """
    if limit:
        query += f" LIMIT {limit}"
    screenings = conn.execute(query).fetchall()
    conn.close()
    results = []
    for s in screenings:
        d = dict(s)
        d["probabilities"] = json.loads(d["probabilities"])
        results.append(d)
    return results


# ──────────── Analytics ────────────

def get_analytics():
    """Get analytics data for the dashboard."""
    conn = get_db()

    total_screenings = conn.execute("SELECT COUNT(*) FROM screenings").fetchone()[0]
    total_patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]

    severity_dist = conn.execute(
        "SELECT label, COUNT(*) as count FROM screenings GROUP BY label"
    ).fetchall()
    severity_data = {row["label"]: row["count"] for row in severity_dist}

    dr_cases = conn.execute(
        "SELECT COUNT(*) FROM screenings WHERE severity > 0"
    ).fetchone()[0]
    dr_rate = round((dr_cases / total_screenings * 100), 1) if total_screenings > 0 else 0

    avg_confidence = conn.execute(
        "SELECT AVG(confidence) FROM screenings"
    ).fetchone()[0]
    avg_confidence = round(avg_confidence * 100, 1) if avg_confidence else 0

    daily_screenings = conn.execute(
        """SELECT DATE(created_at) as date, COUNT(*) as count 
           FROM screenings 
           WHERE created_at >= datetime('now', '-30 days')
           GROUP BY DATE(created_at)
           ORDER BY date"""
    ).fetchall()
    daily_data = [{"date": row["date"], "count": row["count"]} for row in daily_screenings]

    conn.close()

    return {
        "total_screenings": total_screenings,
        "total_patients": total_patients,
        "dr_rate": dr_rate,
        "avg_confidence": avg_confidence,
        "severity_distribution": severity_data,
        "daily_screenings": daily_data,
    }


# Initialize on import
init_db()

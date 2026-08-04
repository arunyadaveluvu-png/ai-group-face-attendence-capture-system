import sqlite3
import pickle
import os
from typing import List, Dict, Tuple, Optional, Any

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(DB_DIR, "attendance_system.db")

def get_connection(db_path: str = DB_NAME) -> sqlite3.Connection:
    """Establish and return a connection to the SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DB_NAME) -> None:
    """Initialize database tables and seed default faculty account."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Create students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            register_no TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    
    # Create faculty table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faculty (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            name TEXT NOT NULL
        )
    """)
    
    # Create attendance_sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot_name TEXT NOT NULL,
            faculty_name TEXT NOT NULL,
            date_str TEXT NOT NULL,
            time_str TEXT NOT NULL,
            total_students INTEGER NOT NULL,
            present_count INTEGER NOT NULL,
            absent_count INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create attendance_logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            register_no TEXT NOT NULL,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES attendance_sessions (id) ON DELETE CASCADE
        )
    """)

    # Seed default faculty account if not exists
    cursor.execute("SELECT * FROM faculty WHERE username = ?", ("faculty",))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO faculty (username, password, name) VALUES (?, ?, ?)",
            ("faculty", "password123", "Faculty Admin")
        )
    
    conn.commit()
    conn.close()

def save_attendance_session(
    slot_name: str,
    faculty_name: str,
    date_str: str,
    time_str: str,
    total_students: int,
    present_count: int,
    absent_count: int,
    records: List[Dict[str, str]],
    db_path: str = DB_NAME
) -> int:
    """
    Save an attendance session and its individual student records into database.
    Returns the generated session_id.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO attendance_sessions (slot_name, faculty_name, date_str, time_str, total_students, present_count, absent_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (slot_name.strip(), faculty_name.strip(), date_str.strip(), time_str.strip(), total_students, present_count, absent_count))
        
        session_id = cursor.lastrowid
        
        for r in records:
            cursor.execute("""
                INSERT INTO attendance_logs (session_id, register_no, name, department, status)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, r.get("Register No", r.get("register_no", "")).strip(),
                  r.get("Name", r.get("name", "")).strip(),
                  r.get("Department", r.get("department", "")).strip(),
                  r.get("Status", r.get("status", "")).strip()))
        
        conn.commit()
        return session_id
    finally:
        conn.close()

def get_all_attendance_sessions(db_path: str = DB_NAME) -> List[Dict[str, Any]]:
    """Retrieve all historical attendance sessions sorted by date and time descending."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, slot_name, faculty_name, date_str, time_str, total_students, present_count, absent_count, created_at
        FROM attendance_sessions
        ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    sessions = []
    for r in rows:
        sessions.append({
            "id": r["id"],
            "slot_name": r["slot_name"],
            "faculty_name": r["faculty_name"],
            "date_str": r["date_str"],
            "time_str": r["time_str"],
            "total_students": r["total_students"],
            "present_count": r["present_count"],
            "absent_count": r["absent_count"],
            "created_at": r["created_at"]
        })
    return sessions

def get_attendance_session_details(session_id: int, db_path: str = DB_NAME) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    """Retrieve session info and individual student logs for a given session ID."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM attendance_sessions WHERE id = ?", (session_id,))
    s_row = cursor.fetchone()
    
    if not s_row:
        conn.close()
        return None, []
        
    session_info = {
        "id": s_row["id"],
        "slot_name": s_row["slot_name"],
        "faculty_name": s_row["faculty_name"],
        "date_str": s_row["date_str"],
        "time_str": s_row["time_str"],
        "total_students": s_row["total_students"],
        "present_count": s_row["present_count"],
        "absent_count": s_row["absent_count"],
        "created_at": s_row["created_at"]
    }
    
    cursor.execute("SELECT register_no, name, department, status FROM attendance_logs WHERE session_id = ?", (session_id,))
    l_rows = cursor.fetchall()
    conn.close()
    
    records = [{
        "Register No": r["register_no"],
        "Name": r["name"],
        "Department": r["department"],
        "Status": r["status"]
    } for r in l_rows]
    
    return session_info, records

def delete_attendance_session(session_id: int, db_path: str = DB_NAME) -> bool:
    """Delete a saved attendance session and its logs."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM attendance_logs WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM attendance_sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def add_student(register_no: str, name: str, department: str, embedding: Any, db_path: str = DB_NAME) -> bool:
    """
    Insert a new student into the database.
    Embedding vector is serialized using pickle.
    """
    serialized_embedding = pickle.dumps(embedding)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO students (register_no, name, department, embedding) VALUES (?, ?, ?, ?)",
            (register_no.strip(), name.strip(), department.strip(), serialized_embedding)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def add_faculty(username: str, password: str, name: str, db_path: str = DB_NAME) -> bool:
    """Register a new faculty account."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO faculty (username, password, name) VALUES (?, ?, ?)",
            (username.strip(), password.strip(), name.strip())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def faculty_exists(username: str, db_path: str = DB_NAME) -> bool:
    """Check if a faculty username exists."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM faculty WHERE username = ?", (username.strip(),))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def student_exists(register_no: str, db_path: str = DB_NAME) -> bool:
    """Check if a student register number already exists."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM students WHERE register_no = ?", (register_no.strip(),))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def get_all_students(db_path: str = DB_NAME) -> List[Dict[str, Any]]:
    """
    Retrieve all registered students with deserialized facial embeddings.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT register_no, name, department, embedding FROM students")
    rows = cursor.fetchall()
    conn.close()
    
    students = []
    for row in rows:
        students.append({
            "register_no": row["register_no"],
            "name": row["name"],
            "department": row["department"],
            "embedding": pickle.loads(row["embedding"])
        })
    return students

def get_all_faculty(db_path: str = DB_NAME) -> List[Dict[str, str]]:
    """Retrieve all faculty members."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT username, name FROM faculty")
    rows = cursor.fetchall()
    conn.close()
    return [{"username": row["username"], "name": row["name"]} for row in rows]

def verify_faculty(username: str, password: str, db_path: str = DB_NAME) -> Optional[Dict[str, Any]]:
    """Verify faculty login credentials."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT username, name FROM faculty WHERE username = ? AND password = ?",
        (username.strip(), password.strip())
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {"username": row["username"], "name": row["name"]}
    return None

def update_student(register_no: str, name: str, department: str, db_path: str = DB_NAME) -> bool:
    """Update name and department of an existing student."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE students SET name = ?, department = ? WHERE register_no = ?",
            (name.strip(), department.strip(), register_no.strip())
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def delete_student(register_no: str, db_path: str = DB_NAME) -> bool:
    """Delete a student profile from database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM students WHERE register_no = ?", (register_no.strip(),))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    print("Database updated and initialized successfully.")


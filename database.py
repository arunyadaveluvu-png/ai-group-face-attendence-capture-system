import sqlite3
import pickle
import os
from typing import List, Dict, Tuple, Optional, Any

DB_NAME = "attendance_system.db"

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
    
    # Seed default faculty account if not exists
    cursor.execute("SELECT * FROM faculty WHERE username = ?", ("faculty",))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO faculty (username, password, name) VALUES (?, ?, ?)",
            ("faculty", "password123", "Faculty Admin")
        )
    
    conn.commit()
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


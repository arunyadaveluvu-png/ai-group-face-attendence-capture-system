import sqlite3
import pickle
import base64
import requests
import sys
import os

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(DB_DIR, "attendance_system.db")

def migrate(url: str, key: str):
    url = url.strip().rstrip('/')
    key = key.strip()
    
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    
    # 1. Migrate Students
    students = conn.execute("SELECT register_no, name, department, embedding FROM students").fetchall()
    print(f"Found {len(students)} local student(s) to migrate...")
    
    s_success = 0
    for s in students:
        reg = s["register_no"]
        name = s["name"]
        dept = s["department"]
        emb_bytes = s["embedding"]
        b64_emb = base64.b64encode(emb_bytes).decode('utf-8')
        
        payload = {
            "register_no": reg,
            "name": name,
            "department": dept,
            "embedding": b64_emb
        }
        res = requests.post(f"{url}/rest/v1/students", json=payload, headers=headers)
        if res.status_code in (200, 201):
            print(f"  [+] Migrated student: {name} ({reg})")
            s_success += 1
        elif res.status_code == 409:
            print(f"  [=] Student already in Supabase: {name} ({reg})")
        else:
            print(f"  [-] Failed student {reg}: {res.status_code} - {res.text}")

    # 2. Migrate Faculty
    faculty = conn.execute("SELECT username, password, name FROM faculty").fetchall()
    print(f"\nFound {len(faculty)} local faculty account(s) to migrate...")
    
    f_success = 0
    for f in faculty:
        u = f["username"]
        p = f["password"]
        n = f["name"]
        
        payload = {"username": u, "password": p, "name": n}
        res = requests.post(f"{url}/rest/v1/faculty", json=payload, headers=headers)
        if res.status_code in (200, 201):
            print(f"  [+] Migrated faculty: {n} ({u})")
            f_success += 1
        elif res.status_code == 409:
            print(f"  [=] Faculty already in Supabase: {n} ({u})")
        else:
            print(f"  [-] Failed faculty {u}: {res.status_code} - {res.text}")

    print(f"\n✅ Migration Complete! {s_success} student(s) and {f_success} faculty account(s) uploaded to Supabase.")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        sup_url = sys.argv[1]
        sup_key = sys.argv[2]
        migrate(sup_url, sup_key)
    else:
        print("Usage: python migrate_to_supabase.py <SUPABASE_URL> <SUPABASE_KEY>")

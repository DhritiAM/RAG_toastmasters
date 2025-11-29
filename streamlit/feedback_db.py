import sqlite3
from datetime import datetime

DB_PATH = "feedback.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        answer TEXT,
        rating INTEGER,
        comments TEXT,
        timestamp TEXT
    )
    """)
    
    conn.commit()
    conn.close()

def save_feedback(query, answer, rating, comments):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
    INSERT INTO feedback (query, answer, rating, comments, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (query, answer, rating, comments, datetime.now()))
    
    conn.commit()
    conn.close()

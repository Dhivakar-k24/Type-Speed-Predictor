import sqlite3
from datetime import datetime

DB_PATH = "typing_sessions.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wpm INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            space_count INTEGER NOT NULL,
            char_count INTEGER NOT NULL,
            duration_seconds REAL NOT NULL,
            top_words TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.commit()
    conn.close()

def save_session(wpm, word_count, space_count, char_count, duration_seconds, top_words=""):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sessions (wpm, word_count, space_count, char_count, duration_seconds, top_words)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (wpm, word_count, space_count, char_count, duration_seconds, top_words))
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id

def get_all_sessions(limit=20):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_overall_stats():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*) as total_sessions,
            ROUND(AVG(wpm), 1) as avg_wpm,
            MAX(wpm) as best_wpm,
            MIN(wpm) as lowest_wpm,
            SUM(word_count) as total_words_typed
        FROM sessions
    """)
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else {}

def delete_session(session_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0

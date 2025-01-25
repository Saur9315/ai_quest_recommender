import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    """Establish a connection to the SQLite database."""
    try:
        if DATABASE_URL:
            return sqlite3.connect(DATABASE_URL)
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        raise


def create_results_table():
    """Create the results table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            efficiency REAL,
            true_answer_amount INTEGER,
            total_questions INTEGER,
            difficulty_encoded INTEGER,
            time_spent_task REAL,
            avg_spent_time REAL,
            prediction INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def update_results(eff, true_answer_amount, total_questions, difficulty_encoded, time_spent_task, avg_spent_time,
                   knowledge_pred, username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
            INSERT INTO results (
                username, efficiency, true_answer_amount, total_questions, 
                difficulty_encoded, time_spent_task, avg_spent_time, prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                efficiency = excluded.efficiency,
                true_answer_amount = excluded.true_answer_amount,
                total_questions = excluded.total_questions,
                difficulty_encoded = excluded.difficulty_encoded,
                time_spent_task = excluded.time_spent_task,
                avg_spent_time = excluded.avg_spent_time,
                prediction = excluded.prediction,
                timestamp = CURRENT_TIMESTAMP
        """, (username, eff, true_answer_amount, total_questions,
              difficulty_encoded, time_spent_task, avg_spent_time, knowledge_pred))

    conn.commit()
    conn.close()


def get_leaderboard_results(limit=10):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT username, efficiency, prediction FROM results
        ORDER BY efficiency DESC LIMIT ?
    """, (limit,))
    results = cursor.fetchall()
    conn.close()
    return results

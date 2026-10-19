"""
Database Management Module (SQLite)
Author: Jayprakash Maurya (T.Y. B.Sc. IT - Vidyalankar School of Information Technology)
Project: Detecting Water Leaks Using Acoustic Signals

This module manages the local SQLite database for logging detection history.
Table: predictions(id, filename, prediction, confidence, timestamp)
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.db")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Create and return a database connection."""
    target_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialize SQLite table for storing acoustic leak prediction records.
    Schema:
    - id: Auto-increment primary key
    - filename: Name of the analyzed audio file
    - prediction: 'NORMAL CONDITION' or 'POSSIBLE WATER LEAK DETECTED'
    - confidence: Confidence percentage (0.0 to 100.0)
    - timestamp: Date and time of analysis
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_result(
    filename: str,
    prediction: str,
    confidence: float,
    db_path: Optional[str] = None,
) -> int:
    """
    Save a prediction record into SQLite.

    Parameters:
    - filename: Name of the audio file analyzed
    - prediction: Result string ('NORMAL CONDITION' or 'POSSIBLE WATER LEAK DETECTED')
    - confidence: Confidence score as percentage (e.g. 94.2)
    - db_path: Optional custom path to db file

    Returns:
    - Inserted row ID.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO predictions (filename, prediction, confidence, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (filename, prediction, round(float(confidence), 2), now_str),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_history(db_path: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
    """
    Retrieve past prediction records ordered by latest first.

    Returns:
    - Pandas DataFrame with columns: ['id', 'filename', 'prediction', 'confidence', 'timestamp']
    """
    init_db(db_path)
    conn = get_connection(db_path)
    query = """
        SELECT id, filename, prediction, confidence, timestamp
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def clear_history(db_path: Optional[str] = None) -> bool:
    """Clear all records from the predictions table."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return True


if __name__ == "__main__":
    init_db()
    test_id = save_result("demo_sample.wav", "POSSIBLE WATER LEAK DETECTED", 92.5)
    print(f"Sample inserted with ID: {test_id}")
    history = get_history()
    print("Recent Predictions History:")
    print(history)

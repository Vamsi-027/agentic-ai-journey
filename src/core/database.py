import os
import sqlite3
import json
from typing import Optional

# Resolve the absolute path of the workspace root to ensure data is written to the correct folder
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DB_PATH = os.path.join(ROOT_DIR, "data", "prompt_experiments.db")

def init_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initializes the SQLite database, creating standard schemas. Handles upgrades from legacy tables."""
    # Ensure the parent data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if table exists and has 'run_id'
    cursor.execute("PRAGMA table_info(experiments)")
    columns = [col[1] for col in cursor.fetchall()]

    if columns and "run_id" not in columns:
        print("⚠️ Legacy database schema detected. Re-creating the table with the new schema...")
        cursor.execute("DROP TABLE experiments")
        columns = []

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            timestamp TEXT,
            prompt_template TEXT,
            test_inputs_json TEXT,
            variant_name TEXT,
            model TEXT,
            temperature REAL,
            system_prompt TEXT,
            max_tokens INTEGER,
            status TEXT,
            latency REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            stop_reason TEXT,
            truncated INTEGER,
            response TEXT,
            error TEXT,
            accuracy_score INTEGER DEFAULT NULL,
            clarity_score INTEGER DEFAULT NULL,
            prompt_hash TEXT,
            latency_ms INTEGER,
            cost_usd REAL,
            tags TEXT
        )
    """)
    conn.commit()

    # Run ALTER TABLE schema migrations if columns are missing
    if columns:
        required_migrations = [
            ("model", "TEXT"),
            ("temperature", "REAL"),
            ("prompt_hash", "TEXT"),
            ("latency_ms", "INTEGER"),
            ("cost_usd", "REAL"),
            ("tags", "TEXT")
        ]
        cursor = conn.cursor()
        for col_name, col_type in required_migrations:
            if col_name not in columns:
                try:
                    cursor.execute(f"ALTER TABLE experiments ADD COLUMN {col_name} {col_type}")
                    conn.commit()
                except sqlite3.OperationalError as e:
                    print(f"⚠️ Column migration for '{col_name}' skipped or failed: {e}")

    return conn

def log_result_to_db(
    conn: sqlite3.Connection, 
    run_id: str, 
    timestamp: str, 
    prompt_template: str, 
    test_input: dict, 
    variant: dict, 
    res: dict,
    tags: Optional[str] = None
):
    """Logs the results of a single prompt execution run into the SQLite database."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO experiments (
            run_id, timestamp, prompt_template, test_inputs_json, variant_name, 
            model, temperature, system_prompt, max_tokens, status, 
            latency, input_tokens, output_tokens, total_tokens, stop_reason,
            truncated, response, error, prompt_hash, latency_ms, cost_usd, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            run_id,
            timestamp,
            prompt_template,
            json.dumps(test_input),
            variant["name"],
            str(variant["model"]),
            variant["temperature"],
            variant["system_prompt"],
            variant.get("max_tokens", 1000),
            res["status"],
            res["latency"],
            res.get("input_tokens", 0),
            res.get("output_tokens", 0),
            res.get("total_tokens", 0),
            res.get("stop_reason", ""),
            res.get("truncated", 0),
            res["response"],
            res["error"],
            res.get("prompt_hash", ""),
            res.get("latency_ms", 0),
            res.get("cost_usd", 0.0),
            tags or res.get("tags", "")
        ),
    )
    conn.commit()

def get_results(experiment_id=None, run_id=None, db_path: str = DEFAULT_DB_PATH):
    """Reads experiment results from the database, optionally filtering by experiment_id or run_id, returning a pandas DataFrame."""
    import pandas as pd
    query = "SELECT * FROM experiments WHERE 1=1"
    params = []
    if experiment_id is not None:
        query += " AND id = ?"
        params.append(experiment_id)
    if run_id is not None:
        query += " AND run_id = ?"
        params.append(run_id)
        
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn, params=params or None)
    conn.close()
    return df

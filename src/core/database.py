import os
import sqlite3
import json

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
            clarity_score INTEGER DEFAULT NULL
        )
    """)
    conn.commit()
    return conn

def log_result_to_db(
    conn: sqlite3.Connection, 
    run_id: str, 
    timestamp: str, 
    prompt_template: str, 
    test_input: dict, 
    variant: dict, 
    res: dict
):
    """Logs the results of a single prompt execution run into the SQLite database."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO experiments (
            run_id, timestamp, prompt_template, test_inputs_json, variant_name, 
            model, temperature, system_prompt, max_tokens, status, 
            latency, input_tokens, output_tokens, total_tokens, stop_reason,
            truncated, response, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            run_id,
            timestamp,
            prompt_template,
            json.dumps(test_input),
            variant["name"],
            variant["model"],
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
        ),
    )
    conn.commit()

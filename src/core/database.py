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
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_runs (
            run_id TEXT PRIMARY KEY,
            task TEXT,
            model TEXT,
            started_at TEXT,
            finished_at TEXT,
            total_cost_usd REAL,
            total_tokens INTEGER,
            outcome TEXT,
            error_msg TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_steps (
            step_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            step_num INTEGER,
            thought TEXT,
            action TEXT,
            action_input TEXT,
            observation TEXT,
            latency_ms INTEGER,
            tokens_used INTEGER,
            FOREIGN KEY(run_id) REFERENCES agent_runs(run_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_benchmarks (
            benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model TEXT,
            embedding_dimension INTEGER,
            num_queries INTEGER,
            top_1_accuracy REAL,
            top_3_accuracy REAL,
            mrr REAL,
            avg_latency_ms REAL,
            total_cost_usd REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_benchmark_queries (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            benchmark_id INTEGER,
            query TEXT,
            expected_chunk_id TEXT,
            actual_rank_1_id TEXT,
            actual_rank_2_id TEXT,
            actual_rank_3_id TEXT,
            rank_found INTEGER,
            reciprocal_rank REAL,
            score_diff_rank_1 REAL,
            FOREIGN KEY(benchmark_id) REFERENCES retrieval_benchmarks(benchmark_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rag_indexing_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            directory_path TEXT,
            extensions TEXT,
            file_count INTEGER,
            chunk_count INTEGER,
            embedding_cost_usd REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rag_benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition TEXT,
            task_id TEXT,
            run_id TEXT,
            steps INTEGER,
            cost_usd REAL,
            retrieval_calls INTEGER,
            outcome TEXT,
            duration_s REAL
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

def log_rag_indexing(
    db_path: str,
    directory_path: str,
    extensions: list[str],
    file_count: int,
    chunk_count: int,
    embedding_cost_usd: float
):
    """Logs the results of a RAG indexing run into the SQLite database."""
    import datetime
    timestamp = datetime.datetime.now().isoformat()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO rag_indexing_logs (
            timestamp, directory_path, extensions, file_count, chunk_count, embedding_cost_usd
        ) VALUES (?, ?, ?, ?, ?, ?)
    """,
        (
            timestamp,
            directory_path,
            json.dumps(extensions),
            file_count,
            chunk_count,
            embedding_cost_usd
        ),
    )
    conn.commit()
    conn.close()

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


class AgentTracer:
    """Context manager for tracing agent execution runs and individual steps.
    Never raises exceptions on database writes to ensure agent execution is never interrupted.
    """
    def __init__(self, task: str, model: str, db_path: Optional[str] = None):
        import uuid
        self.task = task
        self.model = model
        self.db_path = db_path or DEFAULT_DB_PATH
        self.run_id = str(uuid.uuid4())
        self.started_at = None
        self.finished_at = None
        self.total_cost_usd = 0.0
        self.total_tokens = 0
        self.outcome = "failure"
        self.error_msg = None
        self.step_num = 0

    def __enter__(self):
        from datetime import datetime, timezone
        self.started_at = datetime.now(timezone.utc).isoformat()
        try:
            # Ensure tables are initialized
            init_db(self.db_path)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO agent_runs (run_id, task, model, started_at, total_cost_usd, total_tokens, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (self.run_id, self.task, self.model, self.started_at, 0.0, 0, "running")
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ AgentTracer DB Error on enter: {e}")
        return self

    def log_step(self, thought: str, action: str, action_input: str, observation: str, latency_ms: int, tokens_used: int):
        self.step_num += 1
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO agent_steps (run_id, step_num, thought, action, action_input, observation, latency_ms, tokens_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    self.run_id,
                    self.step_num,
                    thought,
                    action,
                    action_input,
                    observation,
                    latency_ms,
                    tokens_used
                )
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ AgentTracer DB Error on log_step: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        from datetime import datetime, timezone
        self.finished_at = datetime.now(timezone.utc).isoformat()
        if exc_type is not None:
            self.outcome = "failure"
            self.error_msg = f"{exc_type.__name__}: {str(exc_val)}"
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE agent_runs
                SET finished_at = ?, total_cost_usd = ?, total_tokens = ?, outcome = ?, error_msg = ?
                WHERE run_id = ?
            """,
                (
                    self.finished_at,
                    self.total_cost_usd,
                    self.total_tokens,
                    self.outcome,
                    self.error_msg,
                    self.run_id
                )
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ AgentTracer DB Error on exit: {e}")


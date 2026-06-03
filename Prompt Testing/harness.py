import os
import uuid
import sqlite3
import json
import time
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found. Please set it in your .env file.")

# Define prompt template
PROMPT_TEMPLATE = """Explain the concept of {concept} to a {target_audience}."""

# Define variants to test (including the three new experiments)
VARIANTS = [
    {
        "name": "Sonnet-Factual",
        "model": "claude-sonnet-4-6",
        "temperature": 0.0,
        "system_prompt": "You are a precise, academic scientist. Explain concepts technically and factually.",
        "max_tokens": 4000,
    },
    {
        "name": "Sonnet-Creative",
        "model": "claude-sonnet-4-6",
        "temperature": 0.7,
        "system_prompt": "You are an imaginative storyteller. Explain concepts using vivid analogies and simple metaphors.",
        "max_tokens": 4000,
        "repeats": 5,  # Experiment 3: Run this variant 5 times to test repeatability
    },
    {
        "name": "Haiku-Concise",
        "model": "claude-haiku-4-5-20251001",
        "temperature": 0.2,
        "system_prompt": "You are a concise assistant. Provide explanations in under 100 words.",
        "max_tokens": 1000,
    },
    {
        "name": "Haiku-Factual",
        "model": "claude-haiku-4-5-20251001",
        "temperature": 0.0,
        "system_prompt": "You are a precise, academic scientist. Explain concepts technically and factually.",
        "max_tokens": 1000,  # Experiment 1: Haiku with Sonnet-Factual system prompt
    },
    {
        "name": "Haiku-PlainProse",
        "model": "claude-haiku-4-5-20251001",
        "temperature": 0.2,
        "system_prompt": "Respond in plain prose only. No markdown, no headers, no bullets. Under 100 words.",
        "max_tokens": 200,  # Experiment 2: Markdown Ban with a small token budget
    },
]

# Define test inputs
TEST_INPUTS = [
    {"concept": "Quantum Entanglement", "target_audience": "10-year-old"},
    {"concept": "Neural Networks", "target_audience": "software engineer"},
    {"concept": "Inflation", "target_audience": "high school student"},
]


def init_db(db_path="prompt_experiments.db"):
    """Initializes the SQLite database. Upgrades legacy table if needed."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if table exists and has 'run_id'
    cursor.execute("PRAGMA table_info(experiments)")
    columns = [col[1] for col in cursor.fetchall()]

    if columns and "run_id" not in columns:
        print(
            "⚠️ Legacy database schema detected. Re-creating the table with the new schema..."
        )
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
    conn, run_id, timestamp, prompt_template, test_input, variant, res
):
    """Inserts a single run's results into the database."""
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


def run_single_experiment(client, variant, test_input, repeat_idx):
    """Runs a single prompt experiment iteration with rate-limit retries."""
    start_time = time.time()
    max_tok = variant.get("max_tokens", 1000)

    try:
        user_prompt = PROMPT_TEMPLATE.format(**test_input)
    except KeyError as e:
        return {
            "status": "Error",
            "error": f"Missing template parameter: {e}",
            "latency": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "stop_reason": "error",
            "truncated": 0,
            "response": "",
            "test_input": test_input,
            "repeat_idx": repeat_idx,
            **variant,
        }

    # Retry logic parameters
    max_retries = 5
    backoff = 2.0

    for attempt in range(max_retries):
        try:
            # Call Anthropic API
            response = client.messages.create(
                model=variant["model"],
                max_tokens=max_tok,
                temperature=variant["temperature"],
                system=variant["system_prompt"],
                messages=[{"role": "user", "content": user_prompt}],
            )

            latency = time.time() - start_time
            response_text = response.content[0].text

            # Token tracking
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            # Stop reason and truncation flag
            stop_reason = getattr(response, "stop_reason", "")
            truncated = 1 if stop_reason == "max_tokens" else 0

            return {
                "status": "Success",
                "error": "",
                "latency": round(latency, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "stop_reason": stop_reason,
                "truncated": truncated,
                "response": response_text,
                "test_input": test_input,
                "repeat_idx": repeat_idx,
                **variant,
            }

        except Exception as e:
            # Detect Rate Limit (HTTP 429 or class name)
            is_rate_limit = False
            if hasattr(e, "status_code") and e.status_code == 429:
                is_rate_limit = True
            elif "rate_limit" in str(e).lower() or "429" in str(e):
                is_rate_limit = True

            if is_rate_limit and attempt < max_retries - 1:
                # Backoff with jitter
                sleep_time = backoff + random.uniform(0.5, 1.5)
                time.sleep(sleep_time)
                backoff *= 2
                continue

            # Non-rate-limit error or we ran out of retries
            latency = time.time() - start_time
            return {
                "status": "Error",
                "error": str(e),
                "latency": round(latency, 2),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "stop_reason": "error",
                "truncated": 0,
                "response": "",
                "test_input": test_input,
                "repeat_idx": repeat_idx,
                **variant,
            }


def print_summary_table(results):
    """Prints a beautiful summary table of all experiments to the console."""
    print("\n" + "=" * 123)
    print("📊 EXPERIMENT RUN SUMMARY REPORT")
    print("=" * 123)

    col_widths = {
        "input": 20,
        "variant": 16,
        "model": 23,
        "status": 7,
        "latency": 7,
        "in_tok": 7,
        "out_tok": 7,
        "tot_tok": 7,
        "trunc": 6,
    }

    header_fmt = (
        f"{{:<{col_widths['input']}}} | "
        f"{{:<{col_widths['variant']}}} | "
        f"{{:<{col_widths['model']}}} | "
        f"{{:<{col_widths['status']}}} | "
        f"{{:>{col_widths['latency']}}} | "
        f"{{:>{col_widths['in_tok']}}} | "
        f"{{:>{col_widths['out_tok']}}} | "
        f"{{:>{col_widths['tot_tok']}}} | "
        f"{{:<{col_widths['trunc']}}}"
    )

    # Print headers
    print(
        header_fmt.format(
            "Input Preview",
            "Variant",
            "Model",
            "Status",
            "Latency",
            "In Tok",
            "Out Tok",
            "Tot Tok",
            "Trunc",
        )
    )
    print("-" * 123)

    total_latency = 0
    total_in_tokens = 0
    total_out_tokens = 0
    success_count = 0
    truncation_count = 0

    for r in results:
        # Format input preview with repetition index if it was repeated
        input_preview = ", ".join(f"{k}={v}" for k, v in r["test_input"].items())
        if r.get("repeat_idx", 1) > 1 or r.get("repeats", 1) > 1:
            input_preview = f"({r['repeat_idx']}) {input_preview}"

        if len(input_preview) > col_widths["input"]:
            input_preview = input_preview[: col_widths["input"] - 3] + "..."

        status = r["status"]
        latency = f"{r['latency']}s"
        in_tok = str(r.get("input_tokens", 0))
        out_tok = str(r.get("output_tokens", 0))
        tot_tok = str(r.get("total_tokens", 0))
        trunc = "YES ⚠️" if r.get("truncated", 0) else "No"

        # Trim model name if too long
        model_name = r["model"]
        if len(model_name) > col_widths["model"]:
            model_name = model_name[: col_widths["model"] - 3] + "..."

        print(
            header_fmt.format(
                input_preview,
                r["name"],
                model_name,
                status,
                latency,
                in_tok,
                out_tok,
                tot_tok,
                trunc,
            )
        )

        if status == "Success":
            success_count += 1
            total_latency += r["latency"]
            total_in_tokens += r.get("input_tokens", 0)
            total_out_tokens += r.get("output_tokens", 0)
            if r.get("truncated", 0):
                truncation_count += 1

    print("-" * 123)
    if success_count > 0:
        avg_latency = round(total_latency / success_count, 2)
        total_tokens = total_in_tokens + total_out_tokens
        print(
            f"✅ Success Rate: {success_count}/{len(results)} | "
            f"Avg Latency: {avg_latency}s | "
            f"Total Input Tok: {total_in_tokens} | "
            f"Total Output Tok: {total_out_tokens} | "
            f"Total Tok: {total_tokens} | "
            f"Truncated Runs: {truncation_count}"
        )
    else:
        print("❌ All runs failed.")
    print("=" * 123 + "\n")


def main():
    client = Anthropic(api_key=api_key)

    # Initialize SQLite Database
    db_path = "prompt_experiments.db"
    conn = init_db(db_path)

    # Generate unique run ID
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Generate list of all runs, expanding variants that request repeats
    all_runs = []
    for test_input in TEST_INPUTS:
        for variant in VARIANTS:
            repeats = variant.get("repeats", 1)
            for i in range(repeats):
                all_runs.append((variant, test_input, i + 1))

    print("\n" + "=" * 80)
    print("🚀 STARTING PROMPT EXPERIMENTATION HARNESS")
    print(f"Run ID: {run_id}")
    print("=" * 80)
    print(f"Prompt Template: {PROMPT_TEMPLATE.strip()}")
    print(f"Loaded {len(VARIANTS)} variants and {len(TEST_INPUTS)} test inputs.")
    print(f"Total iterations to perform: {len(all_runs)}")
    print("=" * 80 + "\n")

    results = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Use 8 workers to reduce rate-limit pressure
    max_workers = min(8, len(all_runs))
    print(f"Executing runs in parallel with {max_workers} worker threads...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_experiment, client, variant, test_input, repeat_idx
            ): (variant, test_input, repeat_idx)
            for variant, test_input, repeat_idx in all_runs
        }

        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            variant, test_input, repeat_idx = futures[future]
            try:
                res = future.result()
                results.append(res)

                # Log result to SQLite
                log_result_to_db(
                    conn, run_id, timestamp, PROMPT_TEMPLATE, test_input, variant, res
                )

                # Console progress updates
                status_symbol = "✅" if res["status"] == "Success" else "❌"
                trunc_suffix = " ⚠️ [TRUNCATED]" if res.get("truncated", 0) else ""
                repeat_suffix = (
                    f" (run {repeat_idx})" if variant.get("repeats", 1) > 1 else ""
                )

                print(
                    f"[{completed_count}/{len(all_runs)}] {status_symbol} "
                    f"Variant: {res['name']}{repeat_suffix} | "
                    f"Input: {list(test_input.values())} | "
                    f"Latency: {res['latency']}s | "
                    f"Tokens: In={res.get('input_tokens', 0)}, Out={res.get('output_tokens', 0)}"
                    f"{trunc_suffix}"
                )

                if res["status"] == "Error":
                    print(f"   ⚠️ Error: {res['error']}")
            except Exception as exc:
                print(
                    f"[{completed_count}/{len(all_runs)}] 💥 Generated an exception: {exc}"
                )

    # Close SQLite connection
    conn.close()

    # Print the summary report
    print_summary_table(results)

    print(
        f"🎉 All experiment records have been logged to the SQLite database: {db_path}\n"
    )


if __name__ == "__main__":
    main()

import os
import uuid
import json
import time
import random
import yaml
import asyncio
import hashlib
from datetime import datetime
from src.core.llm import get_llm_client, LLMProvider, ClaudeModel, OpenAIModel
from src.core.database import init_db, log_result_to_db

# ==============================================================================
# CONFIGURATION: Choose provider (LLMProvider.CLAUDE or LLMProvider.OPENAI)
# ==============================================================================
PROVIDER = LLMProvider.CLAUDE
CONCURRENCY = 5
# ==============================================================================

# Resolve workspace root and load decoupled prompts from prompts/concept_explanations.yaml
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPTS_PATH = os.path.join(ROOT_DIR, "prompts", "concept_explanations.yaml")

with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    prompts_config = yaml.safe_load(f)

# Load prompt template from config
PROMPT_TEMPLATE = prompts_config["templates"]["explanation"]

# Resolve provider name safely (supporting both Enum and raw string fallback)
provider_name = PROVIDER.value if isinstance(PROVIDER, LLMProvider) else str(PROVIDER)
provider_clean = provider_name.lower().strip()

# Define variants using prompts loaded from YAML configuration, split by provider
if provider_clean == LLMProvider.OPENAI.value:
    VARIANTS = [
        {
            "name": "GPT4o-Factual",
            "model": OpenAIModel.GPT_4O,
            "temperature": 0.0,
            "system_prompt": prompts_config["system_prompts"]["Sonnet-Factual"],
            "max_tokens": 4000,
        },
        {
            "name": "GPT4o-Creative",
            "model": OpenAIModel.GPT_4O,
            "temperature": 0.7,
            "system_prompt": prompts_config["system_prompts"]["Sonnet-Creative"],
            "max_tokens": 4000,
            "repeats": 5,  # Run this variant 5 times to test repeatability
        },
        {
            "name": "GPT4oMini-Concise",
            "model": OpenAIModel.GPT_4O_MINI,
            "temperature": 0.2,
            "system_prompt": prompts_config["system_prompts"]["Haiku-Concise"],
            "max_tokens": 1000,
        },
        {
            "name": "GPT4oMini-Factual",
            "model": OpenAIModel.GPT_4O_MINI,
            "temperature": 0.0,
            "system_prompt": prompts_config["system_prompts"]["Haiku-Factual"],
            "max_tokens": 1000,
        },
        {
            "name": "GPT4oMini-PlainProse",
            "model": OpenAIModel.GPT_4O_MINI,
            "temperature": 0.2,
            "system_prompt": prompts_config["system_prompts"]["Haiku-PlainProse"],
            "max_tokens": 200,
        },
    ]
else:  # default to claude
    VARIANTS = [
        {
            "name": "Sonnet-Factual",
            "model": ClaudeModel.CLAUDE_SONNET_4_6,
            "temperature": 0.0,
            "system_prompt": prompts_config["system_prompts"]["Sonnet-Factual"],
            "max_tokens": 4000,
        },
        {
            "name": "Sonnet-Creative",
            "model": ClaudeModel.CLAUDE_SONNET_4_6,
            "temperature": 0.7,
            "system_prompt": prompts_config["system_prompts"]["Sonnet-Creative"],
            "max_tokens": 4000,
            "repeats": 5,  # Run this variant 5 times to test repeatability
        },
        {
            "name": "Haiku-Concise",
            "model": ClaudeModel.CLAUDE_HAIKU_4_5,
            "temperature": 0.2,
            "system_prompt": prompts_config["system_prompts"]["Haiku-Concise"],
            "max_tokens": 1000,
        },
        {
            "name": "Haiku-Factual",
            "model": ClaudeModel.CLAUDE_HAIKU_4_5,
            "temperature": 0.0,
            "system_prompt": prompts_config["system_prompts"]["Haiku-Factual"],
            "max_tokens": 1000,
        },
        {
            "name": "Haiku-PlainProse",
            "model": ClaudeModel.CLAUDE_HAIKU_4_5,
            "temperature": 0.2,
            "system_prompt": prompts_config["system_prompts"]["Haiku-PlainProse"],
            "max_tokens": 200,
        },
    ]

# Define test inputs
TEST_INPUTS = [
    {"concept": "Quantum Entanglement", "target_audience": "10-year-old"},
    {"concept": "Neural Networks", "target_audience": "software engineer"},
    {"concept": "Inflation", "target_audience": "high school student"},
]


async def run_single_experiment(client, variant, test_input, repeat_idx):
    """Runs a single prompt experiment iteration asynchronously using unified LLM layer."""
    start_time = time.time()
    max_tok = variant.get("max_tokens", 1000)

    try:
        user_prompt = PROMPT_TEMPLATE.format(**test_input)
        prompt_hash = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
    except KeyError as e:
        return {
            "status": "Error",
            "error": f"Missing template parameter: {e}",
            "latency": 0,
            "latency_ms": 0,
            "prompt_hash": "",
            "cost_usd": 0.0,
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

    try:
        # Call the unified async generate method
        response = await client.generate(
            prompt=user_prompt,
            system_prompt=variant["system_prompt"],
            model=variant["model"],
            temperature=variant["temperature"],
            max_tokens=max_tok
        )

        latency = time.time() - start_time
        latency_ms = int(latency * 1000)
        stop_reason = response.stop_reason or "end_turn"
        truncated = 1 if stop_reason.lower() in ("max_tokens", "length") else 0

        return {
            "status": "Success",
            "error": "",
            "latency": round(latency, 2),
            "latency_ms": latency_ms,
            "prompt_hash": prompt_hash,
            "cost_usd": response.cost_usd,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.input_tokens + response.output_tokens,
            "stop_reason": stop_reason,
            "truncated": truncated,
            "response": response.text,
            "test_input": test_input,
            "repeat_idx": repeat_idx,
            **variant,
            "model": response.model,
        }

    except Exception as e:
        latency = time.time() - start_time
        latency_ms = int(latency * 1000)
        return {
            "status": "Error",
            "error": str(e),
            "latency": round(latency, 2),
            "latency_ms": latency_ms,
            "prompt_hash": prompt_hash if 'prompt_hash' in locals() else "",
            "cost_usd": 0.0,
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
    """Prints a summary table of all experiments to the console."""
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


async def run_and_log(sem, completed_counter, total_runs, conn, run_id, timestamp, client, variant, test_input, repeat_idx, results):
    """Executes a single experiment run, records logs to SQLite DB, and prints incremental progress."""
    async with sem:
        res = await run_single_experiment(client, variant, test_input, repeat_idx)
    results.append(res)

    # Log result to SQLite with tags
    tags = f"{provider_clean},variant_{variant['name']}"
    log_result_to_db(
        conn, run_id, timestamp, PROMPT_TEMPLATE, test_input, variant, res, tags
    )

    completed_counter[0] += 1

    # Console progress updates
    status_symbol = "✅" if res["status"] == "Success" else "❌"
    trunc_suffix = " ⚠️ [TRUNCATED]" if res.get("truncated", 0) else ""
    repeat_suffix = f" (run {repeat_idx})" if variant.get("repeats", 1) > 1 else ""

    print(
        f"[{completed_counter[0]}/{total_runs}] {status_symbol} "
        f"Variant: {res['name']}{repeat_suffix} | "
        f"Input: {list(test_input.values())} | "
        f"Latency: {res['latency']}s | "
        f"Tokens: In={res.get('input_tokens', 0)}, Out={res.get('output_tokens', 0)}"
        f"{trunc_suffix}"
    )

    if res["status"] == "Error":
        print(f"   ⚠️ Error: {res['error']}")


async def main():
    # Use centralized factory client wrapper based on selected PROVIDER
    client = get_llm_client(provider=PROVIDER)

    # Initialize SQLite Database at the centralized location (data/prompt_experiments.db)
    conn = init_db()

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
    print(f"Provider: {PROVIDER.upper()} | Run ID: {run_id}")
    print("=" * 80)
    print(f"Prompt Template: {PROMPT_TEMPLATE.strip()}")
    print(f"Loaded {len(VARIANTS)} variants and {len(TEST_INPUTS)} test inputs.")
    print(f"Total iterations to perform: {len(all_runs)}")
    print("=" * 80 + "\n")

    results = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Use CONCURRENCY to reduce rate-limit pressure
    max_workers = min(CONCURRENCY, len(all_runs))
    print(f"Executing runs concurrently via async gather (limit={max_workers} concurrent tasks)...\n")

    sem = asyncio.Semaphore(max_workers)
    completed_counter = [0]
    tasks = [
        run_and_log(sem, completed_counter, len(all_runs), conn, run_id, timestamp, client, variant, test_input, repeat_idx, results)
        for variant, test_input, repeat_idx in all_runs
    ]
    await asyncio.gather(*tasks)

    # Close SQLite connection
    conn.close()

    # Print the summary report
    print_summary_table(results)

    # Print centralized DB reference
    db_path = os.path.join(ROOT_DIR, "data", "prompt_experiments.db")
    print(f"🎉 All experiment records have been logged to the SQLite database: {db_path}\n")


if __name__ == "__main__":
    # asyncio.run() is the correct entry point for sync callers.
    # Never call async methods directly from sync code — use this pattern only
    # at the outermost boundary (CLI scripts, __main__ blocks).
    asyncio.run(main())

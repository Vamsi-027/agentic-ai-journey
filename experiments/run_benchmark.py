import os
import sys
import time
import asyncio
import sqlite3
import numpy as np
from pathlib import Path
from src.core.database import init_db
from src.core.llm.openai import OpenAIClient
from src.core.retrieval.chunker import chunk_python_file
from src.core.retrieval.vector_store import VectorStore

BENCHMARK_TASKS = [
    {
        "query": "encoding detection",
        "expected_file": "src/core/llm/tools.py",
        "expected_name": "read_file_content_internal"
    },
    {
        "query": "path safety",
        "expected_file": "src/core/llm/tools.py",
        "expected_name": "validate_path"
    },
    {
        "query": "call openai chat completions api",
        "expected_file": "src/core/llm/openai.py",
        "expected_name": "chat"
    },
    {
        "query": "call anthropic claude messages api",
        "expected_file": "src/core/llm/claude.py",
        "expected_name": "chat"
    },
    {
        "query": "reflexion self-critique generation",
        "expected_file": "src/core/agent/react.py",
        "expected_name": "generate_reflection"
    },
    {
        "query": "run pytest subprocess",
        "expected_file": "src/core/llm/tools.py",
        "expected_name": "run_tests"
    },
    {
        "query": "calculate chat token cost",
        "expected_file": "src/core/llm/tracker.py",
        "expected_name": "calculate_cost"
    },
    {
        "query": "vector store load from disk",
        "expected_file": "src/core/retrieval/vector_store.py",
        "expected_name": "load"
    },
    {
        "query": "cosine similarity matrix math",
        "expected_file": "src/core/retrieval/similarity.py",
        "expected_name": "cosine_similarity_matrix"
    },
    {
        "query": "python ast node chunker",
        "expected_file": "src/core/retrieval/chunker.py",
        "expected_name": "chunk_python_file"
    }
]

MODELS = [
    ("text-embedding-3-small", 1536),
    ("text-embedding-3-large", 3072)
]

async def fetch_embeddings_with_usage(client: OpenAIClient, texts: list[str], model: str):
    """Fetch embeddings using the client completions client directly, capturing token usage and calculating cost."""
    response = await client.client.embeddings.create(
        input=texts,
        model=model
    )
    # Sort embeddings by their original list index
    data = sorted(response.data, key=lambda x: x.index)
    embeddings = [item.embedding for item in data]
    total_tokens = response.usage.total_tokens
    
    # Pricing:
    # text-embedding-3-small: $0.00002 / 1,000 tokens
    # text-embedding-3-large: $0.00013 / 1,000 tokens
    if "large" in model:
        cost = (total_tokens / 1000.0) * 0.00013
    else:
        cost = (total_tokens / 1000.0) * 0.00002
        
    return embeddings, total_tokens, cost

async def run_benchmark_for_model(client: OpenAIClient, model_name: str, dimension: int, chunks: list, workspace_root: Path):
    print(f"\n🚀 Running retrieval benchmark for model: '{model_name}' (dim={dimension})...")
    
    # 1. Fetch corpus embeddings
    chunk_texts = [c.text for c in chunks]
    print(f"  Embedding {len(chunk_texts)} corpus chunks...")
    corpus_embeds, _, _ = await fetch_embeddings_with_usage(client, chunk_texts, model_name)
    
    # 2. Populate Vector Store
    store = VectorStore()
    for idx, (chunk, emb) in enumerate(zip(chunks, corpus_embeds)):
        metadata = {
            "file_path": str(Path(chunk.file_path).relative_to(workspace_root)),
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "type": chunk.type,
            "name": chunk.name
        }
        store.add(id=f"chunk_{idx}", text=chunk.text, embedding=np.array(emb, dtype=np.float32), metadata=metadata)
        
    # 3. Fetch Query Embeddings
    query_texts = [task["query"] for task in BENCHMARK_TASKS]
    print(f"  Embedding {len(query_texts)} search queries...")
    query_embeds, total_query_tokens, total_query_cost = await fetch_embeddings_with_usage(client, query_texts, model_name)
    
    # 4. Search and Evaluate
    query_results = []
    latencies = []
    
    for task, q_emb in zip(BENCHMARK_TASKS, query_embeds):
        q_vec = np.array(q_emb, dtype=np.float32)
        
        # Measure latency
        start_time = time.perf_counter()
        results = store.search(q_vec, top_k=10)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        latencies.append(latency_ms)
        
        # Determine rank of the expected chunk
        expected_file = task["expected_file"]
        expected_name = task["expected_name"]
        
        rank_found = -1
        reciprocal_rank = 0.0
        score_diff_rank_1 = 0.0
        
        # Extract rank details
        actual_rank_1_id = results[0].metadata.get("name") if len(results) > 0 else "None"
        actual_rank_2_id = results[1].metadata.get("name") if len(results) > 1 else "None"
        actual_rank_3_id = results[2].metadata.get("name") if len(results) > 2 else "None"
        
        for rank, res in enumerate(results, 1):
            meta = res.metadata
            if meta["file_path"] == expected_file and meta["name"] == expected_name:
                rank_found = rank
                reciprocal_rank = 1.0 / rank
                break
                
        # If expected is found, compute score diff with Rank 1 (if expected is not Rank 1)
        if rank_found > 1:
            score_diff_rank_1 = results[0].score - results[rank_found - 1].score
            
        query_results.append({
            "query": task["query"],
            "expected_chunk_id": f"{expected_file}::{expected_name}",
            "actual_rank_1_id": actual_rank_1_id,
            "actual_rank_2_id": actual_rank_2_id,
            "actual_rank_3_id": actual_rank_3_id,
            "rank_found": rank_found,
            "reciprocal_rank": reciprocal_rank,
            "score_diff_rank_1": score_diff_rank_1
        })
        
    # Calculate Overall Metrics
    num_queries = len(BENCHMARK_TASKS)
    top_1_hits = sum(1 for qr in query_results if qr["rank_found"] == 1)
    top_3_hits = sum(1 for qr in query_results if 1 <= qr["rank_found"] <= 3)
    
    top_1_accuracy = top_1_hits / num_queries
    top_3_accuracy = top_3_hits / num_queries
    mrr = sum(qr["reciprocal_rank"] for qr in query_results) / num_queries
    avg_latency = sum(latencies) / num_queries
    avg_cost_per_query = total_query_cost / num_queries
    
    print(f"  Done. Top-1: {top_1_accuracy:.2%}, Top-3: {top_3_accuracy:.2%}, MRR: {mrr:.4f}, Latency: {avg_latency:.2f}ms, Cost/Query: ${avg_cost_per_query:.6f}")
    
    return {
        "model": model_name,
        "embedding_dimension": dimension,
        "num_queries": num_queries,
        "top_1_accuracy": top_1_accuracy,
        "top_3_accuracy": top_3_accuracy,
        "mrr": mrr,
        "avg_latency_ms": avg_latency,
        "total_cost_usd": total_query_cost,
        "query_results": query_results
    }

def log_to_sqlite(db_path: Path, run_data: dict):
    """Logs the benchmark run metadata and individual query details to the database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    import datetime
    timestamp = datetime.datetime.now().isoformat()
    
    cursor.execute("""
        INSERT INTO retrieval_benchmarks (
            timestamp, model, embedding_dimension, num_queries,
            top_1_accuracy, top_3_accuracy, mrr, avg_latency_ms, total_cost_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        run_data["model"],
        run_data["embedding_dimension"],
        run_data["num_queries"],
        run_data["top_1_accuracy"],
        run_data["top_3_accuracy"],
        run_data["mrr"],
        run_data["avg_latency_ms"],
        run_data["total_cost_usd"]
    ))
    benchmark_id = cursor.lastrowid
    
    for qr in run_data["query_results"]:
        cursor.execute("""
            INSERT INTO retrieval_benchmark_queries (
                benchmark_id, query, expected_chunk_id,
                actual_rank_1_id, actual_rank_2_id, actual_rank_3_id,
                rank_found, reciprocal_rank, score_diff_rank_1
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            benchmark_id,
            qr["query"],
            qr["expected_chunk_id"],
            qr["actual_rank_1_id"],
            qr["actual_rank_2_id"],
            qr["actual_rank_3_id"],
            qr["rank_found"],
            qr["reciprocal_rank"],
            qr["score_diff_rank_1"]
        ))
        
    conn.commit()
    conn.close()

async def run():
    workspace_root = Path(os.getcwd())
    src_core_dir = workspace_root / "src" / "core"
    db_path = workspace_root / "data" / "prompt_experiments.db"
    
    # Initialize DB (creates the retrieval benchmark tables if they don't exist)
    init_db(str(db_path))
    
    # Collect corpus chunks
    print("Collecting codebase Python files recursively...")
    py_files = sorted(list(src_core_dir.rglob("*.py")))
    chunks = []
    for f in py_files:
        try:
            chunks.extend(chunk_python_file(str(f)))
        except Exception:
            pass
            
    print(f"Total codebase chunks collected: {len(chunks)}")
    
    client = OpenAIClient()
    
    # Run benchmark for both models
    results = []
    for model, dim in MODELS:
        res = await run_benchmark_for_model(client, model, dim, chunks, workspace_root)
        log_to_sqlite(db_path, res)
        results.append(res)
        
    # Print Comparative Summary
    print("\n" + "=" * 100)
    print("COMPARATIVE BENCHMARK RETRIEVAL SUMMARY")
    print("=" * 100)
    
    print(f"{'Embedding Model':<25} | {'Dims':<6} | {'Top-1 Acc':<10} | {'Top-3 Acc':<10} | {'MRR':<8} | {'Avg Latency':<12} | {'Cost/Query':<12}")
    print("-" * 100)
    for res in results:
        print(
            f"{res['model']:<25} | {res['embedding_dimension']:<6} | "
            f"{res['top_1_accuracy']:<10.2%} | {res['top_3_accuracy']:<10.2%} | "
            f"{res['mrr']:<8.4f} | {res['avg_latency_ms']:<10.2f} ms | "
            f"${res['total_cost_usd']/res['num_queries']:<11.6f}"
        )
        
    # Print Detailed Query Rank Performance
    print("\n" + "=" * 100)
    print("DETAILED RANK SEARCH PERFORMANCE BY QUERY")
    print("=" * 100)
    print(f"{'Query':<35} | {'Expected Chunk':<40} | {'Small Rank':<10} | {'Large Rank':<10}")
    print("-" * 105)
    
    small_q = results[0]["query_results"]
    large_q = results[1]["query_results"]
    for sq, lq in zip(small_q, large_q):
        s_rank = "Rank " + str(sq["rank_found"]) if sq["rank_found"] != -1 else "Not Found"
        l_rank = "Rank " + str(lq["rank_found"]) if lq["rank_found"] != -1 else "Not Found"
        # format expected chunk to look concise
        short_exp = sq["expected_chunk_id"].split("src/core/")[-1]
        print(f"{sq['query']:<35} | {short_exp:<40} | {s_rank:<10} | {l_rank:<10}")

if __name__ == "__main__":
    asyncio.run(run())

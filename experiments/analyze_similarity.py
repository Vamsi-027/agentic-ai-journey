import asyncio
import numpy as np
from src.core.llm.openai import OpenAIClient
from src.core.retrieval.similarity import cosine_similarity_matrix, cosine_similarity

# 10 Function docstrings from our coding agent codebase
DOCSTRINGS = {
    "write_file": (
        "Writes content to a file at path atomically using a temporary file."
    ),
    "read_file": (
        "Reads and returns the content of the file at path with automatic encoding detection."
    ),
    "run_python": (
        "Executes arbitrary Python code in a safe subprocess with a timeout, "
        "returning stdout, stderr, and the exit code. Runs inside settings.WORKSPACE_ROOT. "
        "Truncates output to 3000 characters from the end."
    ),
    "run_tests": (
        "Run pytest on a specific path to verify tests. Returns pytest output, "
        "capturing test failures and stack traces, truncated to 3,000 characters from the end."
    ),
    "search_web": (
        "Search the web for the given query using Tavily or DuckDuckGo. Returns at "
        "most 5 results, each with a title, URL, and a snippet truncated to 300 characters."
    ),
    "list_directory": (
        "List directory contents recursively as a file tree with sizes. Can filter "
        "files by an optional glob pattern (e.g. *.py)."
    ),
    "search_in_files": (
        "Search recursively for an exact string query in files, returning matching "
        "lines with file names, line numbers, and snippets."
    ),
    "edit_file": (
        "Perform a find-and-replace edit on a file. Replaces a unique block of text "
        "exactly once. Returns an error if the block is missing or not unique."
    ),
    "evaluate_success": (
        "Evaluates task completion using rule-based checks and an LLM-as-judge call. "
        "Returns (success, reason)."
    ),
    "generate_reflection": (
        "Calls the LLM to reflect on a failed step-by-step trace and generate a self-critique."
    )
}

async def analyze():
    client = OpenAIClient()
    
    names = list(DOCSTRINGS.keys())
    texts = list(DOCSTRINGS.values())
    
    print("Fetching OpenAI embeddings for the 10 docstrings...")
    # Fetch embeddings (shape: 10 x 1536)
    embeds_list = await client.get_embeddings(texts, model="text-embedding-3-small")
    matrix = np.array(embeds_list)
    print(f"Embeddings shape: {matrix.shape}")
    
    # 1. Compute Cosine Similarity Matrix
    cos_matrix = cosine_similarity_matrix(matrix, matrix)
    
    # 2. Compute Dot Product Matrix
    dot_matrix = np.dot(matrix, matrix.T)
    
    # 3. Print Cosine Similarity Matrix formatted nicely
    print("\n" + "=" * 100)
    print("COSINE SIMILARITY MATRIX")
    print("=" * 100)
    header = f"{'Function':<20} | " + " | ".join(f"{name[:10]:<10}" for name in names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(names):
        row_str = f"{name:<20} | " + " | ".join(f"{cos_matrix[i, j]:.4f}" for j in range(len(names)))
        print(row_str)
        
    print("\n" + "=" * 100)
    print("DOT PRODUCT MATRIX")
    print("=" * 100)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(names):
        row_str = f"{name:<20} | " + " | ".join(f"{dot_matrix[i, j]:.4f}" for j in range(len(names)))
        print(row_str)
        
    # Check if cosine similarity lands near dot product
    diff = np.abs(cos_matrix - dot_matrix)
    max_diff = np.max(diff)
    print(f"\nMax absolute difference between Cosine Similarity and Dot Product: {max_diff:.1e}")
    
    # Print analysis of specific interest pairs
    # Pair 1: read_file and write_file
    idx_read = names.index("read_file")
    idx_write = names.index("write_file")
    print(f"\nSimilarity between read_file and write_file: {cos_matrix[idx_read, idx_write]:.4f}")
    
    # Find the top 3 most similar pairs (excluding self-similarity)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((cos_matrix[i, j], names[i], names[j]))
    
    pairs.sort(reverse=True)
    print("\nTop 5 Most Similar Pairs:")
    for score, p1, p2 in pairs[:5]:
        print(f"  {p1} <--> {p2}: {score:.4f}")
        
    # Find the least similar pairs
    print("\nLeast Similar Pairs:")
    for score, p1, p2 in pairs[-3:]:
        print(f"  {p1} <--> {p2}: {score:.4f}")

if __name__ == "__main__":
    asyncio.run(analyze())

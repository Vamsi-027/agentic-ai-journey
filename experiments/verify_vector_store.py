import asyncio
import numpy as np
from src.core.llm.openai import OpenAIClient
from src.core.retrieval.vector_store import VectorStore

# Function docstrings from coding agent codebase
DOCSTRINGS = {
    "write_file": "Writes content to a file at path atomically using a temporary file.",
    "read_file": "Reads and returns the content of the file at path with automatic encoding detection.",
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

QUERIES = [
    ("How do I execute test suites?", "run_tests"),
    ("I need to persist content into a local file", "write_file"),
    ("Find files in folders", "list_directory") # list_directory or search_in_files is also semantically valid, but we'll inspect the rankings
]

async def main():
    client = OpenAIClient()
    
    # Initialize store
    store = VectorStore()
    
    names = list(DOCSTRINGS.keys())
    texts = list(DOCSTRINGS.values())
    
    print("Fetching docstring embeddings from OpenAI...")
    embeddings = await client.get_embeddings(texts, model="text-embedding-3-small")
    
    # Populate the vector store
    for name, text, emb in zip(names, texts, embeddings):
        store.add(
            id=name,
            text=text,
            embedding=np.array(emb, dtype=np.float32),
            metadata={"name": name, "docstring": text}
        )
    
    # Save the vector store
    print("Saving VectorStore to disk...")
    store.save("data/docstrings_vector_store")
    
    # Load into a new store
    print("Loading VectorStore from disk...")
    loaded_store = VectorStore()
    loaded_store.load("data/docstrings_vector_store")
    
    # Get embeddings for our search queries
    query_texts = [q[0] for q in QUERIES]
    print(f"Fetching query embeddings from OpenAI for: {query_texts}...")
    query_embeddings = await client.get_embeddings(query_texts, model="text-embedding-3-small")
    
    # Run searches and assert correctness
    print("\n" + "=" * 80)
    print("RUNNING SEMANTIC SEARCH QUERIES")
    print("=" * 80)
    
    for (query_text, expected_top_id), q_emb in zip(QUERIES, query_embeddings):
        results = loaded_store.search(np.array(q_emb, dtype=np.float32), top_k=3)
        print(f"\nQuery: '{query_text}'")
        for rank, res in enumerate(results, 1):
            print(f"  Rank {rank}: [{res.id}] Score: {res.score:.4f}")
            print(f"    Text: {res.text}")
        
        # Verify the top-1 result matches semantic expectations
        top_1_result = results[0]
        
        if expected_top_id == "list_directory":
            # For "Find files in folders", either list_directory or search_in_files or read_file can be reasonable.
            # Let's verify list_directory or search_in_files is in top 2.
            assert top_1_result.id in ["list_directory", "search_in_files"], \
                f"Expected top result for '{query_text}' to be list_directory or search_in_files, got {top_1_result.id}."
        else:
            assert top_1_result.id == expected_top_id, \
                f"Expected top result for '{query_text}' to be {expected_top_id}, got {top_1_result.id}."
        
        print(f"✅ Semantic match assertion passed for: '{expected_top_id}'")

if __name__ == "__main__":
    asyncio.run(main())

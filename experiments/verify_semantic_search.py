import os
import asyncio
import numpy as np
from pathlib import Path
from src.core.llm.openai import OpenAIClient
from src.core.retrieval.chunker import chunk_python_file
from src.core.retrieval.vector_store import VectorStore

QUERIES = [
    "path validation",
    "encoding detection",
    "stop sequence handling",
    "trace store and db schema",
    "reflexion self-critique generation"
]

async def main():
    client = OpenAIClient()
    store = VectorStore()
    
    workspace_root = Path(os.getcwd())
    src_core_dir = workspace_root / "src" / "core"
    
    print(f"Recursively gathering Python files from: {src_core_dir}...")
    py_files = sorted(list(src_core_dir.rglob("*.py")))
    
    all_chunks = []
    for filepath in py_files:
        rel_path = filepath.relative_to(workspace_root)
        print(f"  Chunking {rel_path}...")
        try:
            chunks = chunk_python_file(str(filepath))
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ⚠️ Error chunking {rel_path}: {e}")
            
    print(f"Total chunks extracted: {len(all_chunks)}")
    
    # Extract texts for embeddings
    texts = [chunk.text for chunk in all_chunks]
    
    # Fetch embeddings in batch
    print(f"Fetching embeddings for {len(texts)} chunks from OpenAI...")
    embeddings = await client.get_embeddings(texts, model="text-embedding-3-small")
    
    # Load into the VectorStore
    for idx, (chunk, emb) in enumerate(zip(all_chunks, embeddings)):
        metadata = {
            "file_path": str(Path(chunk.file_path).relative_to(workspace_root)),
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "type": chunk.type,
            "name": chunk.name
        }
        # Using a unique ID for each chunk
        chunk_id = f"chunk_{idx}_{chunk.type}_{chunk.name}"
        store.add(id=chunk_id, text=chunk.text, embedding=np.array(emb, dtype=np.float32), metadata=metadata)
        
    print("VectorStore successfully loaded with all codebase chunks.")
    
    # Save the store to disk for persistency check
    store.save("data/codebase_vector_store")
    
    # Get embeddings for search queries
    print(f"Fetching query embeddings for: {QUERIES}...")
    query_embeddings = await client.get_embeddings(QUERIES, model="text-embedding-3-small")
    
    print("\n" + "=" * 100)
    print("RUNNING SEMANTIC CODE SEARCH RESULTS")
    print("=" * 100)
    
    for query, q_emb in zip(QUERIES, query_embeddings):
        results = store.search(np.array(q_emb, dtype=np.float32), top_k=3)
        print(f"\nQuery: '{query}'")
        print("-" * len(query))
        for rank, res in enumerate(results, 1):
            meta = res.metadata
            print(f"  Rank {rank}: [{res.id}] Score: {res.score:.4f}")
            print(f"    File: {meta['file_path']} (Lines {meta['start_line']}-{meta['end_line']})")
            print(f"    Type: {meta['type']} | Name: {meta['name']}")
            # Truncate text output to look neat
            preview_lines = res.text.strip().splitlines()
            preview = "\n".join(preview_lines[:4])
            if len(preview_lines) > 4:
                preview += "\n    ..."
            print("    Source Code Preview:")
            print("\n".join(f"      {line}" for line in preview.splitlines()))
            print()

if __name__ == "__main__":
    asyncio.run(main())

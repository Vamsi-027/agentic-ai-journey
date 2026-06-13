import asyncio
import numpy as np
from src.core.llm.openai import OpenAIClient
from src.core.retrieval.vector_store import VectorStore
from src.core.retrieval.faiss_store import FaissVectorStore

QUERIES = [
    "path validation",
    "encoding detection",
    "stop sequence handling",
    "trace store and db schema",
    "reflexion self-critique generation"
]

async def main():
    client = OpenAIClient()
    
    # Load Numpy Vector Store
    print("Loading NumPy VectorStore from disk...")
    np_store = VectorStore()
    np_store.load("data/codebase_vector_store")
    
    # Initialize FAISS Store
    print("Populating FAISS VectorStore with identical corpus...")
    faiss_store = FaissVectorStore()
    for name, text, emb, meta in zip(np_store.ids, np_store.texts, np_store.embeddings, np_store.metadatas):
        faiss_store.add(id=name, text=text, embedding=emb, metadata=meta)
        
    print(f"Total indexed items: {len(np_store.ids)}")
    
    # Get query embeddings
    print("Fetching query embeddings from OpenAI...")
    query_embeddings = await client.get_embeddings(QUERIES, model="text-embedding-3-small")
    
    print("\n" + "=" * 100)
    print("COMPARING SEARCH RESULTS: NUMPY VS. FAISS FLAT")
    print("=" * 100)
    
    mismatch_count = 0
    for query, q_emb in zip(QUERIES, query_embeddings):
        q_vec = np.array(q_emb, dtype=np.float32)
        
        # Search Numpy Store
        np_results = np_store.search(q_vec, top_k=3)
        # Search FAISS Store
        faiss_results = faiss_store.search(q_vec, top_k=3)
        
        print(f"\nQuery: '{query}'")
        print(f"{'Rank':<4} | {'NumPy Result':<40} | {'FAISS Result':<40} | {'Scores (NP / FAISS)':<25}")
        print("-" * 120)
        
        for rank in range(3):
            np_res = np_results[rank]
            faiss_res = faiss_results[rank]
            
            scores_str = f"{np_res.score:.5f} / {faiss_res.score:.5f}"
            print(f"{rank+1:<4} | {np_res.id[:40]:<40} | {faiss_res.id[:40]:<40} | {scores_str:<25}")
            
            # Assertions
            try:
                assert np_res.id == faiss_res.id, f"ID mismatch at Rank {rank+1}: {np_res.id} vs {faiss_res.id}"
                # Allow minor precision tolerance (e.g. 1e-3) due to Float32 precision limits in FAISS distance and non-perfect normalization
                assert np.isclose(np_res.score, faiss_res.score, atol=1e-3), \
                    f"Score mismatch at Rank {rank+1}: {np_res.score} vs {faiss_res.score}"
            except AssertionError as e:
                print(f"  ❌ Mismatch Detected: {e}")
                mismatch_count += 1
                
    if mismatch_count == 0:
        print("\n✅ Verification Success: Both VectorStore implementations returned IDENTICAL top-k items and scores!")
    else:
        print(f"\n❌ Verification Failed: Detected {mismatch_count} mismatches.")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())

import os
import asyncio
from pathlib import Path
from src.core.retrieval.vector_store import VectorStore
from src.core.retrieval.rag_pipeline import RAGPipeline
from src.core.llm.openai import OpenAIClient
from src.core.llm.tools import retrieve_context

async def verify():
    workspace_root = Path(os.getcwd())
    src_core_dir = workspace_root / "src" / "core"
    index_path = workspace_root / "data" / "codebase_index"
    
    print("🚀 Running E2E RAG Pipeline Verification Script...")
    
    # 1. Initialize Vector Store, OpenAIClient, and RAGPipeline
    store = VectorStore()
    client = OpenAIClient()
    pipeline = RAGPipeline(store, client)
    
    print(f"Indexing codebase directory '{src_core_dir}'...")
    # 2. Index the codebase using RAGPipeline
    total_chunks, total_cost = await pipeline.index_directory(
        path=str(src_core_dir),
        extensions=[".py"]
    )
    print(f"✅ Codebase indexed. Total chunks: {total_chunks}, total cost: ${total_cost:.5f}")
    
    # 3. Save Vector Store to disk
    print(f"Saving vector store index to '{index_path}'...")
    store.save(str(index_path))
    
    # Check if files exist
    assert Path(str(index_path) + ".npz").is_file(), "Index npz file was not saved!"
    assert Path(str(index_path) + ".json").is_file(), "Index json file was not saved!"
    print("✅ Index persisted successfully.")
    
    # 4. Perform retrieval query via the tool (will load cached index internally)
    queries = [
        "path validation",
        "encoding detection",
        "reflexion self-critique"
    ]
    
    print("\n" + "="*80)
    print("TESTING RETRIEVE_CONTEXT TOOL OUTPUTS")
    print("="*80)
    
    for idx, query in enumerate(queries, 1):
        print(f"\n🔍 Query [{idx}]: '{query}'")
        # retrieve_context defaults to k=3
        context = await retrieve_context(query)
        print("-" * 40)
        print(context)
        print("-" * 40)
        
        # Verify output structure matches template:
        # [1] filename (lines start-end, chunk_type)
        assert "[1]" in context, "Retrieval result should be numbered and start with [1]"
        assert "lines" in context, "Retrieval header should include line numbers keyword"
        
    print("\n✅ Verification complete. All assertions passed successfully!")

if __name__ == "__main__":
    asyncio.run(verify())

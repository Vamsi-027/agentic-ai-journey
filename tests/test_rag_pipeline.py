import pytest
import numpy as np
import tempfile
import sqlite3
from pathlib import Path
from src.core.retrieval.vector_store import VectorStore
from src.core.retrieval.rag_pipeline import RAGPipeline
from src.core.database import init_db

class MockLLMClient:
    def __init__(self):
        self.get_embeddings_calls = []
        self.get_embeddings_with_usage_calls = []

    async def get_embeddings(self, texts, model="text-embedding-3-small"):
        self.get_embeddings_calls.append((texts, model))
        # Return a list of mock 3D vectors
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def get_embeddings_with_usage(self, texts, model="text-embedding-3-small"):
        self.get_embeddings_with_usage_calls.append((texts, model))
        # Return (embeddings, total_tokens, cost)
        embeddings = [[1.0, 0.0, 0.0] for _ in texts]
        return embeddings, len(texts) * 10, len(texts) * 0.00002

@pytest.fixture
def mock_db():
    # Use temporary file database for tests
    with tempfile.NamedTemporaryFile(suffix=".db") as temp_db:
        db_path = temp_db.name
        init_db(db_path)
        yield db_path

@pytest.fixture
def vector_store():
    return VectorStore()

@pytest.fixture
def llm_client():
    return MockLLMClient()

@pytest.fixture
def pipeline(vector_store, llm_client, mock_db):
    return RAGPipeline(vector_store, llm_client, db_path=mock_db)

@pytest.mark.asyncio
async def test_index_file_python(pipeline, vector_store, llm_client):
    # Create a temporary python file
    content = """def hello():\n    print("Hello world")\n\nclass Dummy:\n    pass\n"""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_py:
        temp_py.write(content)
        temp_py_path = temp_py.name

    try:
        chunk_count, cost = await pipeline.index_file(temp_py_path)
        
        # Two AST chunks: hello() and Dummy, plus some module scope
        assert chunk_count > 0
        assert len(vector_store.embeddings) == chunk_count
        assert len(llm_client.get_embeddings_with_usage_calls) == 1
        
        # Verify metadata keys: {file, start_line, end_line, chunk_type}
        for metadata in vector_store.metadatas:
            assert "file" in metadata
            assert metadata["file"] == temp_py_path
            assert "start_line" in metadata
            assert "end_line" in metadata
            assert "chunk_type" in metadata
            assert metadata["chunk_type"] in ["function", "class", "module"]
    finally:
        Path(temp_py_path).unlink()

@pytest.mark.asyncio
async def test_index_file_text_fallback(pipeline, vector_store):
    # Create a temporary text file
    content = "This is a line.\nThis is another line.\n" + ("a" * 600)
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as temp_txt:
        temp_txt.write(content)
        temp_txt_path = temp_txt.name

    try:
        chunk_count, cost = await pipeline.index_file(temp_txt_path)
        
        # Non-py file defaults to character chunking
        assert chunk_count > 0
        for metadata in vector_store.metadatas:
            assert metadata["file"] == temp_txt_path
            assert metadata["chunk_type"] == "character"
    finally:
        Path(temp_txt_path).unlink()

@pytest.mark.asyncio
async def test_index_directory_and_sqlite_logging(pipeline, vector_store, mock_db):
    # Populate vector store first to test duplicate clearance
    dummy_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vector_store.add("old_chunk", "old text", dummy_vec, {})
    assert len(vector_store.embeddings) == 1

    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Python file
        py_file = temp_dir_path / "mod1.py"
        py_file.write_text("def test_func():\n    pass\n")

        # Text file
        txt_file = temp_dir_path / "readme.txt"
        txt_file.write_text("Instruction manual text here.")

        # Ignore hidden file/directory
        hidden_dir = temp_dir_path / ".git"
        hidden_dir.mkdir()
        (hidden_dir / "config.txt").write_text("git config")

        # Index python files only
        chunk_count, cost = await pipeline.index_directory(str(temp_dir_path), extensions=[".py"])
        
        # Verify old chunk was cleared
        assert "old_chunk" not in vector_store.ids
        
        # Mod1.py has chunks, readme.txt is ignored because it doesn't match extension
        assert chunk_count > 0
        assert len(vector_store.embeddings) == chunk_count
        
        # Verify SQLite logging
        conn = sqlite3.connect(mock_db)
        cursor = conn.cursor()
        cursor.execute("SELECT directory_path, file_count, chunk_count, embedding_cost_usd FROM rag_indexing_logs")
        logs = cursor.fetchall()
        conn.close()

        assert len(logs) == 1
        db_dir, db_file_count, db_chunk_count, db_cost = logs[0]
        assert Path(db_dir).resolve() == Path(temp_dir_path).resolve()
        assert db_file_count == 1
        assert db_chunk_count == chunk_count

@pytest.mark.asyncio
async def test_retrieve_and_formatting(pipeline, vector_store):
    # Empty retrieval check
    empty_results = await pipeline.retrieve("hello")
    assert empty_results == []
    
    empty_formatted = await pipeline.retrieve_formatted("hello")
    assert empty_formatted == "No matching context found."

    # Populated retrieval check
    dummy_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vector_store.add(
        id="chunk_1",
        text="def compute_sum(a, b):\n    return a + b",
        embedding=dummy_vec,
        metadata={"file": "math_utils.py", "start_line": 5, "end_line": 6, "chunk_type": "function"}
    )
    
    results = await pipeline.retrieve("sum", k=1)
    assert len(results) == 1
    assert results[0].id == "chunk_1"
    assert results[0].text == "def compute_sum(a, b):\n    return a + b"
    
    formatted = await pipeline.retrieve_formatted("sum", k=1)
    # Check formatting template matches expected structure
    expected_header = "[1] math_utils.py (lines 5–6, function)"
    assert expected_header in formatted
    assert "def compute_sum" in formatted

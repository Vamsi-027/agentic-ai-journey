import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.core.retrieval.vector_store import VectorStore, SearchResult

def test_vector_store_add_and_search():
    store = VectorStore()
    
    # Define test vectors
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v3 = np.array([0.707, 0.707, 0.0], dtype=np.float32) # cos similarities to v1 will be 1.0, 0.0, 0.707
    
    store.add("id1", "apple text", v1, {"category": "fruit"})
    store.add("id2", "banana text", v2, {"category": "fruit"})
    store.add("id3", "hybrid text", v3, {"category": "hybrid"})
    
    # Query matching v1 exactly
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=2)
    
    assert len(results) == 2
    assert results[0].id == "id1"
    assert results[0].text == "apple text"
    assert pytest.approx(results[0].score, abs=1e-3) == 1.0
    assert results[0].metadata == {"category": "fruit"}
    
    assert results[1].id == "id3"
    assert pytest.approx(results[1].score, abs=1e-3) == 0.707
    assert results[1].metadata == {"category": "hybrid"}

def test_vector_store_dimension_mismatch():
    store = VectorStore()
    store.add("id1", "dim 3", np.array([1.0, 0.0, 0.0]), {})
    
    with pytest.raises(ValueError) as excinfo:
        store.add("id2", "dim 2", np.array([1.0, 0.0]), {})
    assert "Dimension mismatch" in str(excinfo.value)

def test_vector_store_save_and_load(tmp_path):
    store = VectorStore()
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    store.add("id1", "first", v1, {"num": 1})
    store.add("id2", "second", v2, {"num": 2})
    
    persist_path = tmp_path / "test_store"
    store.save(str(persist_path))
    
    # Assert both npz and json files exist
    assert Path(str(persist_path) + ".npz").is_file()
    assert Path(str(persist_path) + ".json").is_file()
    
    # Load into a new store
    new_store = VectorStore()
    new_store.load(str(persist_path))
    
    # Verify content loaded correctly
    assert new_store.ids == ["id1", "id2"]
    assert new_store.texts == ["first", "second"]
    assert new_store.metadatas == [{"num": 1}, {"num": 2}]
    assert np.allclose(new_store.embeddings[0], v1)
    assert np.allclose(new_store.embeddings[1], v2)
    
    # Search loaded store
    results = new_store.search(v1, top_k=1)
    assert len(results) == 1
    assert results[0].id == "id1"
    assert results[0].score == 1.0

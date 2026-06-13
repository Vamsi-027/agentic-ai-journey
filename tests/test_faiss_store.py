import pytest
import numpy as np
from pathlib import Path
from src.core.retrieval.faiss_store import FaissVectorStore

def test_faiss_vector_store_add_and_search():
    store = FaissVectorStore()
    
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v3 = np.array([0.707, 0.707, 0.0], dtype=np.float32)
    
    store.add("id1", "first", v1, {"val": 1})
    store.add("id2", "second", v2, {"val": 2})
    store.add("id3", "third", v3, {"val": 3})
    
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=2)
    
    assert len(results) == 2
    assert results[0].id == "id1"
    assert pytest.approx(results[0].score, abs=1e-3) == 1.0
    
    assert results[1].id == "id3"
    assert pytest.approx(results[1].score, abs=1e-3) == 0.707

def test_faiss_vector_store_save_and_load(tmp_path):
    store = FaissVectorStore()
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    store.add("id1", "apple", v1, {"cat": "a"})
    store.add("id2", "banana", v2, {"cat": "b"})
    
    save_path = tmp_path / "faiss_test"
    store.save(str(save_path))
    
    assert Path(str(save_path) + ".index").is_file()
    assert Path(str(save_path) + ".json").is_file()
    
    # Load into new store
    new_store = FaissVectorStore()
    new_store.load(str(save_path))
    
    assert new_store.ids == ["id1", "id2"]
    assert new_store.texts == ["apple", "banana"]
    assert new_store.metadatas == [{"cat": "a"}, {"cat": "b"}]
    assert np.allclose(new_store.embeddings[0], v1)
    
    results = new_store.search(v1, top_k=1)
    assert len(results) == 1
    assert results[0].id == "id1"
    assert results[0].score == 1.0

import os
import json
import faiss
import numpy as np
from typing import Optional
from src.core.retrieval.vector_store import SearchResult

class FaissVectorStore:
    """A teaching-exercise vector store backed by FAISS IndexFlatL2."""

    def __init__(self):
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.metadatas: list[dict] = []
        self.embeddings: list[np.ndarray] = []
        self.index: Optional[faiss.IndexFlatL2] = None

    def add(self, id: str, text: str, embedding: np.ndarray, metadata: dict):
        """Stores a record, adding its embedding to the FAISS L2 Index."""
        emb = np.array(embedding, dtype=np.float32)
        
        # Ensure dimensions match
        if self.embeddings:
            existing_dim = self.embeddings[0].shape[0]
            if emb.shape[0] != existing_dim:
                raise ValueError(
                    f"Dimension mismatch. Existing vectors have dimension {existing_dim}, "
                    f"tried to insert vector of dimension {emb.shape[0]}."
                )
                
        self.ids.append(id)
        self.texts.append(text)
        self.metadatas.append(metadata)
        self.embeddings.append(emb)
        
        # Add to FAISS index
        dim = emb.shape[0]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
            
        # Reshape to 2D (1, dim) for FAISS
        self.index.add(emb.reshape(1, -1))

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        """Returns the top-k results by converting L2 distances to cosine similarity."""
        if not self.embeddings or self.index is None:
            return []
            
        q_emb = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # FAISS search returns distances (squared L2) and index offsets
        distances, indices = self.index.search(q_emb, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            idx = int(idx)
            
            # Since vectors are L2-normalized:
            # L2 distance squared = ||A - B||^2 = ||A||^2 + ||B||^2 - 2 * A . B
            # D^2 = 1 + 1 - 2 * cos(theta) = 2 - 2 * cos(theta)
            # Therefore, cos(theta) = 1.0 - (D^2 / 2.0)
            cos_sim = 1.0 - float(dist) / 2.0
            
            results.append(
                SearchResult(
                    id=self.ids[idx],
                    text=self.texts[idx],
                    score=cos_sim,
                    metadata=self.metadatas[idx]
                )
            )
        return results

    def save(self, path: str):
        """Persists the FAISS index and JSON metadata payload to disk."""
        if path.endswith('.index'):
            base_path = path[:-6]
        elif path.endswith('.json'):
            base_path = path[:-5]
        else:
            base_path = path
            
        index_path = base_path + ".index"
        json_path = base_path + ".json"
        
        if self.index is not None:
            faiss.write_index(self.index, index_path)
            
        # Save payload along with embeddings list for complete reconstruction
        embeddings_list = [emb.tolist() for emb in self.embeddings]
        payload = {
            "ids": self.ids,
            "texts": self.texts,
            "metadatas": self.metadatas,
            "embeddings": embeddings_list
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """Loads a persisted FAISS store from disk."""
        if path.endswith('.index'):
            base_path = path[:-6]
        elif path.endswith('.json'):
            base_path = path[:-5]
        else:
            base_path = path
            
        index_path = base_path + ".index"
        json_path = base_path + ".json"
        
        with open(json_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
            
        self.ids = payload["ids"]
        self.texts = payload["texts"]
        self.metadatas = payload["metadatas"]
        self.embeddings = [np.array(emb, dtype=np.float32) for emb in payload["embeddings"]]
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            # Re-create index from loaded raw embeddings if index file is missing
            self.index = None
            if self.embeddings:
                dim = self.embeddings[0].shape[0]
                self.index = faiss.IndexFlatL2(dim)
                stacked = np.vstack(self.embeddings).astype(np.float32)
                self.index.add(stacked)

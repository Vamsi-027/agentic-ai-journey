import json
import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.core.retrieval.similarity import cosine_similarity_matrix

@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict

class VectorStore:
    """An in-memory vector database built from scratch using numpy."""

    def __init__(self):
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.metadatas: list[dict] = []
        self.embeddings: list[np.ndarray] = []

    def clear(self):
        """Clears all records and embeddings from the store."""
        self.ids = []
        self.texts = []
        self.metadatas = []
        self.embeddings = []


    def add(self, id: str, text: str, embedding: np.ndarray, metadata: dict):
        """Stores a record including text content, vector embedding, and metadata."""
        # Convert embedding to numpy array if it isn't one already
        emb = np.array(embedding, dtype=np.float32)
        
        # If there are already embeddings, ensure shape/dimensionality compatibility
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

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        """Returns the top-k results by cosine similarity, sorted descending."""
        if not self.embeddings:
            return []
        
        q_emb = np.array(query_embedding, dtype=np.float32)
        corpus = np.vstack(self.embeddings)  # shape (M, D)
        
        # Calculate cosine similarity using our vectorized similarity module
        # Scores will have shape (M,) since q_emb is 1D
        scores = cosine_similarity_matrix(q_emb, corpus)
        
        # Get indices of top_k elements sorted in descending order
        # argsort sorts ascending, so we reverse it with [::-1]
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append(
                SearchResult(
                    id=self.ids[idx],
                    text=self.texts[idx],
                    score=float(scores[idx]),
                    metadata=self.metadatas[idx]
                )
            )
        return results

    def save(self, path: str):
        """Persists the store to disk using npz for embeddings and JSON for metadata."""
        if path.endswith('.npz'):
            base_path = path[:-4]
        elif path.endswith('.json'):
            base_path = path[:-5]
        else:
            base_path = path

        npz_path = base_path + ".npz"
        json_path = base_path + ".json"
        
        # Save embeddings
        embeddings_arr = np.array(self.embeddings, dtype=np.float32)
        np.savez(npz_path, embeddings=embeddings_arr)
        
        # Save payload metadata
        payload = {
            "ids": self.ids,
            "texts": self.texts,
            "metadatas": self.metadatas
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """Loads a persisted store from disk."""
        if path.endswith('.npz'):
            base_path = path[:-4]
        elif path.endswith('.json'):
            base_path = path[:-5]
        else:
            base_path = path

        npz_path = base_path + ".npz"
        json_path = base_path + ".json"
        
        # Load embeddings
        data = np.load(npz_path)
        embeddings_arr = data['embeddings']
        self.embeddings = [row for row in embeddings_arr]
        
        # Load payload metadata
        with open(json_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
            
        self.ids = payload["ids"]
        self.texts = payload["texts"]
        self.metadatas = payload["metadatas"]

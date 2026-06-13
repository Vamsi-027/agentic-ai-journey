import os
from pathlib import Path
import numpy as np
from src.core.retrieval.chunker import chunk_python_file, chunk_character, Chunk
from src.core.retrieval.vector_store import VectorStore, SearchResult
from src.core.database import DEFAULT_DB_PATH, log_rag_indexing


class RAGPipeline:
    """Orchestrates chunking, embedding generation, and vector storage."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client,
        embedding_model: str = "text-embedding-3-small",
        db_path: str = DEFAULT_DB_PATH,
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.db_path = db_path

    async def index_file(self, path: str, batch_size: int = 100) -> tuple[int, float]:
        """Chunks a file, embeds each chunk in batches, and adds to the vector store."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found at {path}")

        # Choose chunker strategy based on extension
        if p.suffix == ".py":
            chunks = chunk_python_file(str(p))
        else:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            chunks = chunk_character(content, str(p))

        if not chunks:
            return 0, 0.0

        total_cost = 0.0

        # Batch processing
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_texts = [c.text for c in batch]

            embeddings, tokens, cost = await self.llm_client.get_embeddings_with_usage(
                batch_texts, model=self.embedding_model
            )
            total_cost += cost

            for idx, (chunk, emb) in enumerate(zip(batch, embeddings)):
                # Generate unique ID for the chunk
                chunk_id = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}:{chunk.name}:{i + idx}"
                metadata = {
                    "file": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.type,
                }
                self.vector_store.add(
                    id=chunk_id,
                    text=chunk.text,
                    embedding=np.array(emb, dtype=np.float32),
                    metadata=metadata,
                )

        return len(chunks), total_cost

    async def index_directory(
        self, path: str, extensions: list[str], batch_size: int = 100
    ) -> tuple[int, float]:
        """Walks directory, clear current store, index all matching files, log chunk count & cost to SQLite."""
        # 1. Clear vector store at the start of indexing (full rebuild)
        self.vector_store.clear()

        dir_path = Path(path)
        if not dir_path.is_dir():
            raise ValueError(
                f"Directory path '{path}' does not exist or is not a directory."
            )

        total_chunks = 0
        total_cost = 0.0
        file_count = 0

        # Walk recursively
        for root, _, files in os.walk(dir_path):
            # Exclude standard hidden folders / venv folders
            parts = Path(root).parts
            if any(
                part.startswith(".")
                or part in ("__pycache__", ".venv", "venv", "node_modules", "data")
                for part in parts
            ):
                continue

            for file in files:
                file_p = Path(root) / file
                if any(file.endswith(ext) for ext in extensions):
                    try:
                        chunks, cost = await self.index_file(
                            str(file_p), batch_size=batch_size
                        )
                        total_chunks += chunks
                        total_cost += cost
                        file_count += 1
                    except Exception as e:
                        print(f"⚠️ Failed to index file {file_p}: {e}")

        # Log indexing run to SQLite
        log_rag_indexing(
            db_path=self.db_path,
            directory_path=str(dir_path.resolve()),
            extensions=extensions,
            file_count=file_count,
            chunk_count=total_chunks,
            embedding_cost_usd=total_cost,
        )

        return total_chunks, total_cost

    async def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        """Embeds query and queries the vector store for top-k results. Handles empty store safely."""
        if not self.vector_store.embeddings:
            return []

        embeddings = await self.llm_client.get_embeddings(
            [query], model=self.embedding_model
        )
        if not embeddings:
            return []

        q_emb = np.array(embeddings[0], dtype=np.float32)
        return self.vector_store.search(q_emb, top_k=k)

    async def retrieve_formatted(self, query: str, k: int = 5) -> str:
        """Retrieves top-k chunks and formats them into a clean markdown block with line ranges and files."""
        results = await self.retrieve(query, k=k)
        if not results:
            return "No matching context found."

        formatted_chunks = []
        for idx, res in enumerate(results, 1):
            file_path = res.metadata.get("file", "unknown")
            # Try to resolve relative path if possible for cleaner output
            try:
                from src.core.config import settings

                workspace_path = Path(settings.WORKSPACE_ROOT).resolve()
                file_path = str(Path(file_path).resolve().relative_to(workspace_path))
            except Exception:
                pass

            start_line = res.metadata.get("start_line", "?")
            end_line = res.metadata.get("end_line", "?")
            chunk_type = res.metadata.get("chunk_type", "unknown")

            header = (
                f"[{idx}] {file_path} (lines {start_line}–{end_line}, {chunk_type})"
            )
            formatted_chunks.append(f"{header}\n{res.text}")

        return "\n\n".join(formatted_chunks)

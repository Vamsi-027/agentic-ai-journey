# Vector Stores Comparison: NumPy vs. FAISS Flat vs. FAISS IVF

This document provides a technical comparison of the three primary vector search architectures that are commonly used in Retrieval-Augmented Generation (RAG) pipelines.

---

## 📐 1. NumPy-backed Custom Store (Exact Search)

A NumPy-backed store implements brute-force, exact nearest-neighbor search from scratch using vector/matrix operations. 

### 🧮 How it Works & Complexity
- **Distance Metric:** Cosine similarity computed using:
  $$\text{cos}(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$
- **Algorithm:** Brute-force linear scan. It computes the dot product of the query vector $Q$ against the stacked corpus matrix $C$ of shape $(N, D)$ in a single matrix multiplication step ($Q \cdot C^T$), followed by vector divisions of computed L2 norms.
- **Search Complexity:** $O(N \cdot D)$ where $N$ is the number of database chunks and $D$ is the embedding dimension (e.g., 1536).
- **Memory Overhead:** Minimal. Stores raw arrays directly in Python's memory space.

### 🌟 Pros
- **Zero External Dependencies:** Built entirely on base Python and NumPy, making it highly portable.
- **Exact Accuracy:** 100% recall (guarantees the mathematically closest vectors are retrieved).
- **Dynamic Updates:** Extremely easy to add, modify, or delete vectors in memory dynamically without rebuilding indices.

### ⚠️ Cons
- **Linear Scaling Bottleneck:** As $N$ grows into the hundreds of thousands or millions, brute-force search latency increases linearly ($O(N)$), leading to bottlenecks.
- **Lack of Multi-threading:** Executed inside the Python interpreter (subject to GIL constraints unless offloaded to native BLAS/LAPACK bindings underlying NumPy).

---

## ⚡ 2. FAISS IndexFlat (Flat Exact Search)

FAISS (Facebook AI Similarity Search) Flat is an exact nearest-neighbor search index implemented in optimized C++. It does not partition or compress the vector space; it performs exact brute-force distance scans.

### 🧮 How it Works & Complexity
- **Distance Metric:** `IndexFlatL2` computes squared Euclidean ($L2$) distance:
  $$D^2 = \|A - B\|^2 = \|A\|^2 + \|B\|^2 - 2 (A \cdot B)$$
  `IndexFlatIP` computes Inner Product (equivalent to Cosine Similarity when vectors are L2-normalized).
- **Algorithm:** Brute-force C++ exact search. It leverages highly optimized SIMD instructions (AVX2, ARM NEON) and OpenMP multithreading.
- **Search Complexity:** $O(N \cdot D)$ (but executes orders of magnitude faster than Python loops or basic NumPy).

### 🌟 Pros
- **Exact Accuracy:** 100% recall (identical results to our NumPy store for matching metrics).
- **Hardware Acceleration:** Highly optimized C++ backend utilizing SIMD and multicore parallelization.
- **Simplicity:** No hyperparameter tuning or training required.

### ⚠️ Cons
- **Linear Scaling Limits:** Still scales linearly $O(N)$ at search time. Latency will eventually degrade at millions of vectors.
- **No Compression:** Entire index must reside in RAM, which can become memory-expensive for high dimensions.

---

## 🌐 3. FAISS IndexIVF (Inverted File Approximate Search)

`IndexIVF` (Inverted File Index) is an **Approximate Nearest Neighbor (ANN)** search method. It accelerates searches at scale by partitioning the vector space into clusters.

### 🧮 How it Works & Complexity
- **Clustering (Training Phase):** The vector space is partitioned into $C$ Voronoi cells (clusters) using $k$-means clustering. Each vector is assigned to its nearest centroid.
- **Inverted Lists:** An inverted list is maintained for each centroid, mapping the centroid to all vectors belonging to its Voronoi cell.
- **Querying & Probing:** During search, the query vector is first compared against the centroids to find the closest clusters. Then, only the vectors within those closest clusters are scanned.
- **Hyperparameters:**
  - `nlist`: The total number of centroids/cells to partition into (requires training the index).
  - `nprobe`: The number of adjacent centroids to search/probe during query time.
- **Search Complexity:** $O(\text{centroids} \cdot D + \text{probed\_cells} \cdot \frac{N}{\text{centroids}} \cdot D)$
  - If `nprobe` is small (e.g. 1), search time is extremely fast (sub-linear), but recall drops.
  - If `nprobe == nlist`, it probes all clusters, degenerating into an exact Flat search with additional centroid search overhead.

### 🌟 Pros
- **Sub-Linear Search Scaling:** Breaks the $O(N)$ barrier. Search time can remain sub-millisecond even across millions of vectors.
- **Scalability:** The only viable option for large production search workloads.

### ⚠️ Cons
- **Loss of Exactness (Approximate Recall):** May miss the true nearest neighbor if `nprobe` is set too low (gives up accuracy for latency).
- **Index Training Overhead:** Requires a representative training dataset of vectors to perform $k$-means centroid allocation before indexing.
- **Complex Updates:** Adding new vectors can degrade centroid partitions over time, requiring periodic re-training and re-indexing.

---

## 📊 Summary Decision Matrix

| Metric / Feature | NumPy Store | FAISS Flat (`IndexFlatL2`) | FAISS IVF (`IndexIVFFlat`) |
| :--- | :--- | :--- | :--- |
| **Search Accuracy** | 100% Exact | 100% Exact | Approximate (depends on `nprobe`) |
| **Search Speed Scaling** | Linear $O(N)$ | Linear $O(N)$ (Highly Optimized) | Sub-Linear $O(N / \text{nlist})$ |
| **Best Scale Size** | Small ($N < 10,000$) | Medium ($N < 100,000$) | Large ($N > 1,000,000$) |
| **Dependencies** | NumPy only | `faiss-cpu` / `faiss-gpu` | `faiss-cpu` / `faiss-gpu` |
| **Memory footprint** | Low (Raw Arrays) | Moderate (C++ memory structures) | Low to Moderate (Centroids + lists) |
| **Training Required** | No | No | Yes (requires $k$-means training) |
| **When to Choose** | Lightweight, offline prototype RAG apps with small document pools. | Mid-scale RAG apps demanding 100% accuracy and fast SIMD-backed searches. | Enterprise-scale production engines searching across millions of documents. |

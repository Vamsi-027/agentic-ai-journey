import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two 1D numpy arrays.
    
    Formula: cos(theta) = (A . B) / (||A|| * ||B||)
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Computes a cosine similarity matrix between query and corpus arrays.
    
    query: np.ndarray of shape (D,) or (N, D)
    corpus: np.ndarray of shape (M, D)
    
    Returns:
        np.ndarray of shape (N, M) if query is 2D, or (M,) if query is 1D.
    """
    is_1d = query.ndim == 1
    if is_1d:
        Q = query[np.newaxis, :]  # shape (1, D)
    else:
        Q = query
        
    C = corpus  # shape (M, D)
    
    # Dot products of shape (N, M)
    dot_products = np.dot(Q, C.T)
    
    # Compute L2 norms along feature dimension (axis 1)
    norm_q = np.linalg.norm(Q, axis=1, keepdims=True)  # shape (N, 1)
    norm_c = np.linalg.norm(C, axis=1, keepdims=True).T  # shape (1, M)
    
    # Prevent division by zero by replacing zero norms with a tiny epsilon
    # or replacing the division inputs. Epsilon division is standard and clean.
    norm_q = np.where(norm_q == 0.0, 1e-9, norm_q)
    norm_c = np.where(norm_c == 0.0, 1e-9, norm_c)
    
    similarity = dot_products / (norm_q * norm_c)
    
    if is_1d:
        return similarity[0]
    return similarity

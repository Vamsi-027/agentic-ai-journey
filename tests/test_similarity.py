import pytest
import numpy as np
from src.core.retrieval.similarity import cosine_similarity, cosine_similarity_matrix

def test_cosine_similarity_identical():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    similarity = cosine_similarity(a, b)
    assert pytest.approx(similarity) == 1.0

def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    similarity = cosine_similarity(a, b)
    assert pytest.approx(similarity) == 0.0

def test_cosine_similarity_opposite():
    a = np.array([1.0, 1.0])
    b = np.array([-1.0, -1.0])
    similarity = cosine_similarity(a, b)
    assert pytest.approx(similarity) == -1.0

def test_cosine_similarity_zero_norm():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    similarity = cosine_similarity(a, b)
    assert similarity == 0.0

def test_cosine_similarity_matrix_1d_query():
    query = np.array([1.0, 2.0])
    corpus = np.array([
        [1.0, 2.0],
        [-1.0, -2.0],
        [2.0, 1.0]
    ])
    
    similarities = cosine_similarity_matrix(query, corpus)
    assert similarities.shape == (3,)
    
    # Verify matches loop
    for i in range(len(corpus)):
        single = cosine_similarity(query, corpus[i])
        assert pytest.approx(similarities[i]) == single

def test_cosine_similarity_matrix_2d_query():
    queries = np.array([
        [1.0, 2.0],
        [0.0, 1.0]
    ])
    corpus = np.array([
        [1.0, 2.0],
        [-1.0, -2.0],
        [2.0, 1.0],
        [0.0, 0.0]  # Zero vector check
    ])
    
    similarities = cosine_similarity_matrix(queries, corpus)
    assert similarities.shape == (2, 4)
    
    # Verify matches loop
    for i in range(len(queries)):
        for j in range(len(corpus)):
            single = cosine_similarity(queries[i], corpus[j])
            assert pytest.approx(similarities[i, j]) == single

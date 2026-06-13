import pytest
import tempfile
from pathlib import Path
from src.core.retrieval.chunker import (
    Chunk,
    chunk_character,
    chunk_sentences,
    chunk_python_file,
    char_idx_to_line
)

def test_char_idx_to_line():
    text = "line1\nline2\n\nline4"
    assert char_idx_to_line(text, 0) == 1   # 'l' in line1
    assert char_idx_to_line(text, 5) == 1   # '\n' after line1
    assert char_idx_to_line(text, 6) == 2   # 'l' in line2
    assert char_idx_to_line(text, 12) == 3  # '\n' on empty line

def test_chunk_character():
    text = "abcdefghij" # 10 chars
    # chunk_size = 4, overlap = 2
    # Expect: "abcd" (idx 0-4), "cdef" (idx 2-6), "efgh" (idx 4-8), "ghij" (idx 6-10)
    chunks = chunk_character(text, "test.txt", chunk_size=4, chunk_overlap=2)
    assert len(chunks) == 4
    assert chunks[0].text == "abcd"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 1
    assert chunks[0].type == "character"
    assert chunks[0].name == "chunk_0"
    
    assert chunks[1].text == "cdef"
    assert chunks[2].text == "efgh"
    assert chunks[3].text == "ghij"

def test_chunk_sentences():
    text = "Hello world! This is a test. Another sentence."
    # max_tokens = 5 words.
    # Sentences:
    # 1: "Hello world!" (2 words)
    # 2: " This is a test." (4 words)
    # 3: " Another sentence." (2 words)
    # Merging sentence 1 + 2: 6 words (exceeds 5). So sentence 1 is its own chunk.
    # Merging sentence 2 + 3: 6 words (exceeds 5). So sentence 2 is its own chunk.
    # Sentence 3 is its own chunk.
    chunks = chunk_sentences(text, "test.txt", max_tokens=5)
    assert len(chunks) == 3
    assert chunks[0].text == "Hello world!"
    assert chunks[1].text == " This is a test."
    assert chunks[2].text == " Another sentence."
    
    # Large max_tokens should merge them
    merged_chunks = chunk_sentences(text, "test.txt", max_tokens=20)
    assert len(merged_chunks) == 1
    assert merged_chunks[0].text == text

def test_chunk_python_file(tmp_path):
    code = (
        "import os\n"
        "X = 42\n"
        "\n"
        "class MyClass:\n"
        "    def method(self):\n"
        "        pass\n"
        "\n"
        "def helper():\n"
        "    return 1\n"
    )
    
    filepath = tmp_path / "sample.py"
    filepath.write_text(code, encoding="utf-8")
    
    chunks = chunk_python_file(str(filepath))
    
    # Let's inspect the extracted chunks:
    # Expected chunks:
    # 1. module chunk at top (import os, X = 42)
    # 2. class chunk (MyClass)
    # 3. function chunk (method)
    # 4. function chunk (helper)
    
    # We should have exactly 4 chunks
    assert len(chunks) == 4
    
    # Sort or check order by start_line
    assert chunks[0].type == "module"
    assert "import os" in chunks[0].text
    assert "X = 42" in chunks[0].text
    
    # Class chunk (MyClass definition)
    class_chunk = next(c for c in chunks if c.type == "class")
    assert class_chunk.name == "MyClass"
    assert "class MyClass" in class_chunk.text
    
    # Function chunks
    func_names = {c.name for c in chunks if c.type == "function"}
    assert func_names == {"method", "helper"}
    
    helper_chunk = next(c for c in chunks if c.name == "helper")
    assert "def helper()" in helper_chunk.text
    assert helper_chunk.start_line == 8
    assert helper_chunk.end_line == 9

import pytest
import os
from pathlib import Path
from unittest.mock import patch
from src.core.config import settings
from src.core.llm.tools import (
    validate_path,
    read_file,
    write_file,
    list_directory,
    search_in_files,
    edit_file
)

@pytest.fixture(autouse=True)
def mock_workspace_root(tmp_path):
    """Fixture to dynamically redirect settings.WORKSPACE_ROOT to pytest's tmp_path using env vars."""
    resolved_tmp = str(tmp_path.resolve())
    old_root = os.environ.get("WORKSPACE_ROOT")
    os.environ["WORKSPACE_ROOT"] = resolved_tmp
    yield tmp_path
    if old_root is not None:
        os.environ["WORKSPACE_ROOT"] = old_root
    else:
        os.environ.pop("WORKSPACE_ROOT", None)


# ==============================================================================
# 1. Path Safety Validation Tests
# ==============================================================================

def test_path_safety_validation(mock_workspace_root):
    # Path inside workspace is accepted and resolved
    valid_file = mock_workspace_root / "test.txt"
    resolved = validate_path(str(valid_file), settings.WORKSPACE_ROOT)
    assert isinstance(resolved, Path)
    assert resolved == valid_file.resolve()

    # Relative path inside workspace is resolved correctly
    resolved_rel = validate_path("./test.txt", settings.WORKSPACE_ROOT)
    assert isinstance(resolved_rel, Path)
    
    # Path escaping workspace to parent directory is blocked
    escaped_rel = validate_path("../outside.txt", settings.WORKSPACE_ROOT)
    assert escaped_rel == "Error: path escape — access denied"

    # Absolute path outside workspace is blocked
    escaped_abs = validate_path("/etc/passwd", settings.WORKSPACE_ROOT)
    assert escaped_abs == "Error: path escape — access denied"


# ==============================================================================
# 2. Read File Tests (Encoding, Missing, Escape)
# ==============================================================================

def test_read_file_success(mock_workspace_root):
    test_file = mock_workspace_root / "hello.txt"
    test_file.write_text("Hello, workspace!", encoding="utf-8")

    res = read_file(str(test_file))
    assert res == "Hello, workspace!"


def test_read_file_non_utf8_encoding(mock_workspace_root):
    test_file = mock_workspace_root / "latin1.txt"
    # Write using ISO-8859-1 (Latin-1)
    test_file.write_text("Café", encoding="latin-1")

    res = read_file(str(test_file))
    assert res == "Café"


def test_read_file_missing(mock_workspace_root):
    res = read_file("does_not_exist.txt")
    assert "Error:" in res
    assert "does not exist or is not a file" in res


def test_read_file_escape(mock_workspace_root):
    res = read_file("../escaped.txt")
    assert res == "Error: path escape — access denied"


# ==============================================================================
# 3. Write File Tests (Atomic, Parent Directories, Escape)
# ==============================================================================

def test_write_file_success(mock_workspace_root):
    target_path = mock_workspace_root / "nested/dir/atomic.txt"
    res = write_file(str(target_path), "Atomic content")
    
    assert "written successfully" in res
    assert target_path.exists()
    assert target_path.read_text(encoding="utf-8") == "Atomic content"


def test_write_file_escape(mock_workspace_root):
    res = write_file("../escaped_write.txt", "content")
    assert res == "Error: path escape — access denied"


# ==============================================================================
# 4. List Directory Tests (Glob pattern, Sizes, Escape)
# ==============================================================================

def test_list_directory_success(mock_workspace_root):
    # Setup files
    (mock_workspace_root / "file1.py").write_text("print('hello')", encoding="utf-8")
    (mock_workspace_root / "sub").mkdir()
    (mock_workspace_root / "sub/file2.py").write_text("x = 10", encoding="utf-8")
    (mock_workspace_root / "sub/data.json").write_text("{}", encoding="utf-8")

    # List all files (no pattern)
    res_all = list_directory(".")
    assert "file1.py" in res_all
    assert "sub/file2.py" in res_all
    assert "sub/data.json" in res_all
    assert "14 bytes" in res_all  # file1.py size
    assert "6 bytes" in res_all   # sub/file2.py size
    assert "2 bytes" in res_all   # sub/data.json size

    # List py files only
    res_py = list_directory(".", pattern="*.py")
    assert "file1.py" in res_py
    assert "sub/file2.py" in res_py
    assert "data.json" not in res_py

    # Directory escape
    res_esc = list_directory("../")
    assert res_esc == "Error: path escape — access denied"


# ==============================================================================
# 5. Search in Files Tests (Grep queries, Line number, Escape)
# ==============================================================================

def test_search_in_files_success(mock_workspace_root):
    # Setup files
    (mock_workspace_root / "f1.txt").write_text("first line\ntarget match here\nthird line", encoding="utf-8")
    (mock_workspace_root / "f2.txt").write_text("no match here\nanother target match line", encoding="utf-8")

    # Search for "target match"
    res = search_in_files(query="target match", path=".")
    assert "f1.txt:2: target match here" in res
    assert "f2.txt:2: another target match line" in res

    # Search with pattern filter
    res_filtered = search_in_files(query="target match", path=".", file_pattern="f1.*")
    assert "f1.txt:2: target match here" in res_filtered
    assert "f2.txt" not in res_filtered

    # Search escape
    res_esc = search_in_files(query="target", path="../")
    assert res_esc == "Error: path escape — access denied"


# ==============================================================================
# 6. Edit File Tests (Exact find-and-replace once, occurrences constraint)
# ==============================================================================

def test_edit_file_success(mock_workspace_root):
    test_file = mock_workspace_root / "edit_target.txt"
    test_file.write_text("Hello World!\nThis is a unique test line.\nGoodbye World!", encoding="utf-8")

    res = edit_file(str(test_file), old_str="unique test line", new_str="modified line")
    assert "edited successfully" in res
    assert "modified line" in test_file.read_text(encoding="utf-8")
    assert "unique test line" not in test_file.read_text(encoding="utf-8")


def test_edit_file_zero_occurrences(mock_workspace_root):
    test_file = mock_workspace_root / "edit_zero.txt"
    test_file.write_text("Line one\nLine two", encoding="utf-8")

    res = edit_file(str(test_file), old_str="non-existent", new_str="replacement")
    assert "Error: old_str not found in the file" in res


def test_edit_file_multiple_occurrences(mock_workspace_root):
    test_file = mock_workspace_root / "edit_multi.txt"
    test_file.write_text("duplicate line\nsome text\nduplicate line", encoding="utf-8")

    res = edit_file(str(test_file), old_str="duplicate line", new_str="replacement")
    assert "Error: old_str found multiple times (2)" in res
    assert test_file.read_text(encoding="utf-8") == "duplicate line\nsome text\nduplicate line"


def test_edit_file_escape(mock_workspace_root):
    res = edit_file("../escaped_edit.txt", "old", "new")
    assert res == "Error: path escape — access denied"

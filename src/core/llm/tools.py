import os
import sys
import subprocess
import tempfile
import charset_normalizer
from pathlib import Path
from typing import Optional
from src.core.llm.base import ToolDefinition
from src.core.config import settings

# ==============================================================================
# Helper functions for validation & encoding
# ==============================================================================

def validate_path(path: str, workspace_root: Path | str) -> Path | str:
    """Resolves target path and checks it remains strictly within the workspace root."""
    try:
        workspace_path = Path(workspace_root).resolve()
        
        # If relative, resolve it relative to the workspace root
        input_path = Path(path)
        if not input_path.is_absolute():
            target_path = (workspace_path / input_path).resolve()
        else:
            target_path = input_path.resolve()
        
        try:
            target_path.relative_to(workspace_path)
            return target_path
        except ValueError:
            return "Error: path escape — access denied"
    except Exception as e:
        return f"Error: invalid path format: {str(e)}"


_SKIP_PREFIXES = (".", "__")

def _should_skip(p: Path, base: Path) -> bool:
    try:
        return any(part.startswith(_SKIP_PREFIXES) for part in p.relative_to(base).parts)
    except ValueError:
        return True


def read_file_content_internal(path: Path) -> str:
    """Reads file content with robust encoding detection, attempting UTF-8 first."""
    try:
        with open(path, "rb") as f:
            raw_data = f.read()
        
        if not raw_data:
            return ""
            
        content = None
        # 1. Try UTF-8 first
        try:
            content = raw_data.decode("utf-8")
        except UnicodeDecodeError:
            pass
            
        # 2. Try UTF-16 if BOM is present
        if content is None:
            if raw_data.startswith((b'\xff\xfe', b'\xfe\xff')):
                try:
                    content = raw_data.decode("utf-16")
                except Exception:
                    pass
                
        # 3. Try charset_normalizer with high confidence threshold
        if content is None:
            detected = charset_normalizer.detect(raw_data)
            encoding = detected.get("encoding")
            confidence = detected.get("confidence", 0.0)
            
            if encoding and confidence > 0.85 and not (encoding.startswith("utf_16") and len(raw_data) < 10):
                try:
                    content = raw_data.decode(encoding)
                except Exception:
                    pass
                
        # 4. Fallback to ISO-8859-1 (Latin-1)
        if content is None:
            content = raw_data.decode("latin-1")

        MAX_CHARS = 50_000
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + f"\n\n... [truncated — {len(content):,} total chars, showing first {MAX_CHARS:,}]"
            
        return content
    except Exception as e:
        return f"Error: Failed to read or decode file: {str(e)}"


# ==============================================================================
# Tool Definitions (Schemas)
# ==============================================================================

WRITE_FILE_TOOL = ToolDefinition(
    name="write_file",
    description="Write content to a file at the specified path atomically using a temporary file. Creates parent directories if they do not exist.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The target file path (absolute or relative to current working directory)."
            },
            "content": {
                "type": "string",
                "description": "The exact text content to write to the file."
            }
        },
        "required": ["path", "content"]
    }
)

READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description="Read and return the complete contents of a file at the specified path with automatic encoding detection.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to read (absolute or relative to current working directory)."
            }
        },
        "required": ["path"]
    }
)

RUN_PYTHON_TOOL = ToolDefinition(
    name="run_python",
    description="Execute arbitrary Python code in a safe subprocess with a timeout, returning stdout and stderr.",
    input_schema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The complete Python script or code block to execute."
            }
        },
        "required": ["code"]
    }
)

SEARCH_WEB_TOOL = ToolDefinition(
    name="search_web",
    description="Search the web for the given query. Currently a stub.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search term or query to lookup on the web."
            }
        },
        "required": ["query"]
    }
)

LIST_DIRECTORY_TOOL = ToolDefinition(
    name="list_directory",
    description="List directory contents recursively as a file tree with sizes. Can filter files by an optional glob pattern (e.g. *.py).",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the directory to list (absolute or relative to current working directory)."
            },
            "pattern": {
                "type": "string",
                "description": "Optional glob pattern to filter files (e.g., '*.py')."
            }
        },
        "required": ["path"]
    }
)

SEARCH_IN_FILES_TOOL = ToolDefinition(
    name="search_in_files",
    description="Search recursively for an exact string query in files, returning matching lines with file names, line numbers, and snippets.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The exact string query to search for."
            },
            "path": {
                "type": "string",
                "description": "The directory or file path to search inside."
            },
            "file_pattern": {
                "type": "string",
                "description": "Optional glob pattern to filter which files to search (e.g., '*.py')."
            }
        },
        "required": ["query", "path"]
    }
)

EDIT_FILE_TOOL = ToolDefinition(
    name="edit_file",
    description="Perform a find-and-replace edit on a file. Replaces a unique block of text exactly once. Returns an error if the block is missing or not unique.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to edit."
            },
            "old_str": {
                "type": "string",
                "description": "The exact unique string/block of text in the file to replace."
            },
            "new_str": {
                "type": "string",
                "description": "The new replacement string/block of text."
            }
        },
        "required": ["path", "old_str", "new_str"]
    }
)


# ==============================================================================
# Tool Implementations
# ==============================================================================

def write_file(path: str, content: str) -> str:
    """Writes content to a file at path atomically using a temporary file."""
    val = validate_path(path, settings.WORKSPACE_ROOT)
    if isinstance(val, str):
        return val
        
    try:
        val.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write via temp file in target directory
        with tempfile.NamedTemporaryFile(dir=val.parent, delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
            
        os.replace(temp_path, val)
        return f"File written successfully to {path}"
    except Exception as e:
        if 'temp_path' in locals() and temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return f"Error: Failed to write file to {path}: {str(e)}"


def read_file(path: str) -> str:
    """Reads and returns the content of the file at path with automatic encoding detection."""
    val = validate_path(path, settings.WORKSPACE_ROOT)
    if isinstance(val, str):
        return val
        
    if not val.is_file():
        return f"Error: The file at '{path}' does not exist or is not a file."
        
    return read_file_content_internal(val)


def run_python(code: str, timeout: float = 10.0) -> str:
    """Executes python code in a subprocess with a timeout and returns output."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = []
        if result.stdout:
            output.append(result.stdout)
        if result.stderr:
            output.append(f"Stderr:\n{result.stderr}")
        if result.returncode != 0:
            output.append(f"Process exited with return code {result.returncode}")
        
        return "\n".join(output) if output else "Execution completed successfully with no output."
    except subprocess.TimeoutExpired:
        return f"Error: Python code execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing Python code: {str(e)}"


def search_web(query: str) -> str:
    """Stub returning web search not implemented."""
    return "Web search not yet implemented"


def list_directory(path: str, pattern: Optional[str] = None) -> str:
    """Returns a tree structure list of files under path with their sizes."""
    val = validate_path(path, settings.WORKSPACE_ROOT)
    if isinstance(val, str):
        return val
        
    if not val.is_dir():
        return f"Error: '{path}' is not a directory."
        
    try:
        glob_pattern = pattern or "*"
        files_found = []
        for p in val.rglob(glob_pattern):
            if p.is_file():
                if _should_skip(p, val):
                    continue
                # Validate path safety in case of symlinks
                file_val = validate_path(str(p), settings.WORKSPACE_ROOT)
                if isinstance(file_val, str):
                    continue
                    
                try:
                    rel_path = p.relative_to(val)
                    size = p.stat().st_size
                    files_found.append((str(rel_path), size))
                except Exception:
                    continue
                    
        if not files_found:
            return f"No files found matching '{glob_pattern}' in '{path}'."
            
        files_found.sort()
        lines = [f"- {rel} ({size} bytes)" for rel, size in files_found]
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing directory '{path}': {str(e)}"


def search_in_files(query: str, path: str, file_pattern: Optional[str] = None) -> str:
    """Grep-style recursive search returning file, line number, and matching snippet."""
    val = validate_path(path, settings.WORKSPACE_ROOT)
    if isinstance(val, str):
        return val
        
    if not val.exists():
        return f"Error: '{path}' does not exist."
        
    try:
        glob_pattern = file_pattern or "*"
        files_to_search = []
        
        if val.is_file():
            if file_pattern:
                if val.match(file_pattern):
                    files_to_search = [val]
            else:
                files_to_search = [val]
        else:
            files_to_search = [p for p in val.rglob(glob_pattern) if p.is_file()]
            
        results = []
        base_dir = val if val.is_dir() else val.parent
        for file_path in files_to_search:
            if _should_skip(file_path, base_dir):
                continue
            file_val = validate_path(str(file_path), settings.WORKSPACE_ROOT)
            if isinstance(file_val, str):
                continue
                
            content = read_file_content_internal(file_val)
            if content.startswith("Error"):
                continue
                
            lines = content.splitlines()
            for line_num, line in enumerate(lines, 1):
                if query in line:
                    try:
                        rel_path = file_path.relative_to(Path(settings.WORKSPACE_ROOT).resolve())
                    except Exception:
                        rel_path = file_path
                    results.append(f"{rel_path}:{line_num}: {line.strip()}")
                    if len(results) >= 100:
                        break
            if len(results) >= 100:
                results.append("... (additional matches truncated)")
                break
                
        if not results:
            return f"No matches found for query '{query}' in '{path}'."
        return "\n".join(results)
    except Exception as e:
        return f"Error searching in files: {str(e)}"


def edit_file(path: str, old_str: str, new_str: str) -> str:
    """Finds and replaces old_str with new_str exactly once."""
    val = validate_path(path, settings.WORKSPACE_ROOT)
    if isinstance(val, str):
        return val
        
    if not val.is_file():
        return f"Error: '{path}' is not a file."
        
    try:
        content = read_file_content_internal(val)
        if content.startswith("Error"):
            return content
            
        occurrences = content.count(old_str)
        if occurrences == 0:
            return f"Error: old_str not found in the file '{path}'."
        if occurrences > 1:
            return f"Error: old_str found multiple times ({occurrences}) in the file '{path}'. Be more specific."
            
        new_content = content.replace(old_str, new_str, 1)
        
        write_res = write_file(path, new_content)
        if write_res.startswith("Error"):
            return write_res
            
        return f"File '{path}' edited successfully."
    except Exception as e:
        return f"Error editing file '{path}': {str(e)}"

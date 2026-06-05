import os
import sys
import subprocess
from src.core.llm.base import ToolDefinition

# ==============================================================================
# Tool Definitions (Schemas)
# ==============================================================================

WRITE_FILE_TOOL = ToolDefinition(
    name="write_file",
    description="Write content to a file at the specified path. Creates parent directories if they do not exist.",
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
    description="Read and return the complete contents of a file at the specified path.",
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


# ==============================================================================
# Tool Implementations
# ==============================================================================

def write_file(path: str, content: str) -> str:
    """Writes content to a file at path, creating parent directories if necessary."""
    try:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File written successfully to {path}"
    except Exception as e:
        raise RuntimeError(f"Failed to write file to {path}: {str(e)}")


def read_file(path: str) -> str:
    """Reads and returns the content of the file at path."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file at '{path}' was not found.")
    except Exception as e:
        raise RuntimeError(f"Failed to read file at {path}: {str(e)}")


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
        raise TimeoutError(f"Error: Python code execution timed out after {timeout} seconds.")
    except Exception as e:
        raise RuntimeError(f"Error executing Python code: {str(e)}")


def search_web(query: str) -> str:
    """Stub returning web search not implemented."""
    return "Web search not yet implemented"

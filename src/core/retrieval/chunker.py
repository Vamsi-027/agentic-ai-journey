import re
import ast
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    start_line: int
    end_line: int
    type: str  # "function", "class", "module", "character", "sentence"
    name: str  # e.g., function/class name, or sequential identifier
    file_path: str


def char_idx_to_line(text: str, idx: int) -> int:
    """Helper to convert a character index to a 1-indexed line number."""
    return text.count("\n", 0, idx) + 1


def chunk_character(
    text: str, file_path: str, chunk_size: int = 500, chunk_overlap: int = 100
) -> list[Chunk]:
    """Slices text into overlapping character-level chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks = []
    start = 0
    idx = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end]

        s_line = char_idx_to_line(text, start)
        e_line = char_idx_to_line(text, max(start, end - 1))

        chunks.append(
            Chunk(
                text=chunk_text,
                start_line=s_line,
                end_line=e_line,
                type="character",
                name=f"chunk_{idx}",
                file_path=file_path,
            )
        )
        idx += 1
        if end == text_len:
            break
        start += chunk_size - chunk_overlap

    return chunks


def chunk_sentences(text: str, file_path: str, max_tokens: int = 150) -> list[Chunk]:
    """Splits text on sentence boundaries (. ! ?) and merges up to max_tokens (using word count heuristic)."""
    # Regex splits on sentences while keeping punctuation
    sentence_regex = re.compile(r"[^.!?]+(?:[.!?]+|\Z)", re.DOTALL)
    matches = list(sentence_regex.finditer(text))

    if not matches:
        return []

    chunks = []
    current_sentences = []
    current_tokens = 0
    current_start = -1
    chunk_idx = 0

    for match in matches:
        s_text = match.group(0)
        s_start = match.start()
        s_end = match.end()

        # Simple word count approximation for tokens
        s_tokens = len(s_text.split())

        if current_sentences and current_tokens + s_tokens > max_tokens:
            # Save the current chunk
            chunk_text = "".join(current_sentences)
            s_line = char_idx_to_line(text, current_start)
            e_line = char_idx_to_line(text, max(current_start, s_end - 1))
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_line=s_line,
                    end_line=e_line,
                    type="sentence",
                    name=f"sentence_{chunk_idx}",
                    file_path=file_path,
                )
            )
            chunk_idx += 1
            current_sentences = []
            current_tokens = 0
            current_start = -1

        if current_start == -1:
            current_start = s_start

        current_sentences.append(s_text)
        current_tokens += s_tokens

    if current_sentences:
        chunk_text = "".join(current_sentences)
        s_line = char_idx_to_line(text, current_start)
        e_line = char_idx_to_line(text, len(text) - 1)
        chunks.append(
            Chunk(
                text=chunk_text,
                start_line=s_line,
                end_line=e_line,
                type="sentence",
                name=f"sentence_{chunk_idx}",
                file_path=file_path,
            )
        )

    return chunks


def chunk_python_file(path: str) -> list[Chunk]:
    """Code-aware chunker using AST to extract function, class, and module-level scopes."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Python file not found at {path}")

    with open(p, "r", encoding="utf-8") as f:
        code = f.read()

    try:
        tree = ast.parse(code, filename=path)
    except SyntaxError as e:
        # If parsing fails, fall back to a character chunker or raise
        raise ValueError(f"Syntax error parsing {path}: {str(e)}")

    lines = code.splitlines()
    total_lines = len(lines)
    covered = [False] * (total_lines + 1)

    nodes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes.append(node)

    # Sort nodes by start line
    nodes = sorted(nodes, key=lambda n: getattr(n, "lineno", 0))

    chunks = []
    for node in nodes:
        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line)

        text = ast.get_source_segment(code, node) or ""
        ntype = "class" if isinstance(node, ast.ClassDef) else "function"

        chunks.append(
            Chunk(
                text=text,
                start_line=start_line,
                end_line=end_line,
                type=ntype,
                name=node.name,
                file_path=path,
            )
        )

        # Mark lines covered by classes/functions
        for l in range(start_line, end_line + 1):
            if l <= total_lines:
                covered[l] = True

    # Group contiguous uncovered lines as module chunks
    module_chunks = []
    in_block = False
    block_start = 0

    for l in range(1, total_lines + 1):
        if not covered[l]:
            if not in_block:
                in_block = True
                block_start = l
        else:
            if in_block:
                block_end = l - 1
                block_lines = lines[block_start - 1 : block_end]
                block_text = "\n".join(block_lines)

                # Create a module chunk only if it has actual code/non-comment content
                if block_text.strip() and not all(
                    line.strip().startswith("#") or not line.strip()
                    for line in block_lines
                ):
                    module_chunks.append(
                        Chunk(
                            text=block_text,
                            start_line=block_start,
                            end_line=block_end,
                            type="module",
                            name=f"{p.stem}_module_{len(module_chunks)}",
                            file_path=path,
                        )
                    )
                in_block = False

    if in_block:
        block_end = total_lines
        block_lines = lines[block_start - 1 : block_end]
        block_text = "\n".join(block_lines)
        if block_text.strip() and not all(
            line.strip().startswith("#") or not line.strip() for line in block_lines
        ):
            module_chunks.append(
                Chunk(
                    text=block_text,
                    start_line=block_start,
                    end_line=block_end,
                    type="module",
                    name=f"{p.stem}_module_{len(module_chunks)}",
                    file_path=path,
                )
            )

    all_chunks = chunks + module_chunks
    all_chunks.sort(key=lambda c: c.start_line)
    return all_chunks

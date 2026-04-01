"""Test all Python code blocks from documentation files.

Extracts Python code blocks from RST and Markdown files and executes them
to verify they work correctly. Blocks within the same file share a namespace
to support the common documentation pattern of building on prior examples.
"""

import glob
import re
import textwrap

import pytest


def _extract_python_blocks_rst(filepath):
    """Extract Python code blocks from RST files."""
    with open(filepath, "r") as f:
        content = f.read()

    blocks = []
    pattern = re.compile(
        r"\.\.\s+code-block::\s+python3?\s*\n"
        r"((?:\n|\s*\n|[ \t]+[^\n]*\n)*)",
        re.MULTILINE,
    )
    for match in pattern.finditer(content):
        block = match.group(1)
        lines = block.split("\n")
        code_lines = []
        started = False
        for line in lines:
            if not started:
                if line.strip():
                    started = True
                    code_lines.append(line)
            else:
                code_lines.append(line)

        code = textwrap.dedent("\n".join(code_lines)).strip()
        if code:
            blocks.append(code)
    return blocks


def _extract_python_blocks_md(filepath):
    """Extract Python code blocks from Markdown files."""
    with open(filepath, "r") as f:
        content = f.read()

    blocks = []
    pattern = re.compile(
        r"```python3?\s*\n(.*?)```",
        re.DOTALL,
    )
    for match in pattern.finditer(content):
        code = match.group(1).strip()
        if code:
            blocks.append(code)
    return blocks


# Lines that reference external resources or are demo-only
_SKIP_PATTERNS = [
    "s3://",
    "gs://",
    "https://example.com",
    "read_parquet",
    "read_csv",
    "to_csv",
    "to_parquet",
    "write_csv",
    "write_parquet",
    "reticulate",
    "library(",
    "read.csv",
    "read.table",
    "read_parquet",
]


def _should_skip(code):
    """Check if a code block should be skipped."""
    for pattern in _SKIP_PATTERNS:
        if pattern in code:
            return True
    return False


def _collect_doc_files():
    """Collect documentation file paths."""
    result = []
    doc_root = "docs"

    for rst_file in sorted(
        glob.glob(f"{doc_root}/**/*.rst", recursive=True),
    ):
        result.append(rst_file)

    if glob.glob("README.md"):
        result.append("README.md")

    return result


_DOC_FILES = _collect_doc_files()


@pytest.mark.parametrize("filepath", _DOC_FILES)
def test_doc_file_code_blocks(filepath):
    """Test that all Python code blocks in a doc file execute correctly.

    Blocks within a file share a namespace to support sequential examples.
    """
    if filepath.endswith(".rst"):
        blocks = _extract_python_blocks_rst(filepath)
    else:
        blocks = _extract_python_blocks_md(filepath)

    if not blocks:
        pytest.skip("No Python code blocks found")

    namespace = {}
    for i, code in enumerate(blocks):
        if _should_skip(code):
            continue
        try:
            exec(
                compile(code, f"{filepath}:block{i}", "exec"),
                namespace,
            )
        except Exception as e:
            raise AssertionError(
                f"Code block {i} in {filepath} failed:\n"
                f"---\n{code}\n---\n"
                f"Error: {e}"
            ) from e

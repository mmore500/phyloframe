"""Sybil-based documentation code block testing.

Extracts and executes Python code blocks from RST and Markdown files.
Blocks within the same file share a namespace to support sequential
examples that build on prior code.

Requires the ``sybil`` package (part of the [testing] extra).
When sybil is not installed, this module is a no-op so that the rest of
the test suite can run normally.
"""

try:
    from sybil import Sybil
    from sybil.parsers.markdown import CodeBlockParser as MdCodeBlockParser
    from sybil.parsers.rest import CodeBlockParser as RstCodeBlockParser
    from sybil.parsers.rest import SkipParser
except ImportError:
    pass
else:
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
    ]

    def _evaluate_python_block(example) -> None:  # noqa: ANN001
        """Execute a Python code block, skipping blocks with external deps."""
        source = example.parsed
        for pattern in _SKIP_PATTERNS:
            if pattern in source:
                return
        exec(  # noqa: S102
            compile(source, example.path, "exec"),
            example.namespace,
        )

    pytest_collect_file = (
        Sybil(
            parsers=[
                RstCodeBlockParser(
                    language="python",
                    evaluator=_evaluate_python_block,
                ),
                SkipParser(),
            ],
            patterns=["docs/**/*.rst"],
        )
        + Sybil(
            parsers=[
                MdCodeBlockParser(
                    language="python3",
                    evaluator=_evaluate_python_block,
                ),
            ],
            patterns=["*.md"],
        )
    ).pytest()

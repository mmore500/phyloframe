import argparse

_SUPPORTED_COMPRESSIONS = ("gzip", "bz2", "xz", "zip")


def add_compression_cli_arg(parser: argparse.ArgumentParser) -> None:
    """Add a ``--compression`` argument to *parser*.

    Supported compression formats are gzip, bz2, xz, and zip — all
    provided by the Python standard library.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to add the flag to.
    """
    parser.add_argument(
        "--compression",
        type=str,
        choices=_SUPPORTED_COMPRESSIONS,
        default=None,
        help=(
            "Compression format for the output file. "
            "Supported: gzip, bz2, xz, zip. "
            "If not set, output is written uncompressed."
        ),
    )

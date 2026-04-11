import argparse

import pytest

from phyloframe._auxlib import add_compression_cli_arg


def test_default_none():
    parser = argparse.ArgumentParser()
    add_compression_cli_arg(parser)
    args = parser.parse_args([])
    assert args.compression is None


@pytest.mark.parametrize(
    "compression",
    ["gzip", "bz2", "xz", "zip"],
)
def test_supported_compression(compression: str):
    parser = argparse.ArgumentParser()
    add_compression_cli_arg(parser)
    args = parser.parse_args(["--compression", compression])
    assert args.compression == compression


def test_unsupported_compression():
    parser = argparse.ArgumentParser()
    add_compression_cli_arg(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--compression", "zstd"])


def test_unsupported_compression_lz4():
    parser = argparse.ArgumentParser()
    add_compression_cli_arg(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--compression", "lz4"])


def test_with_other_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-file", type=str)
    add_compression_cli_arg(parser)
    args = parser.parse_args(
        ["--output-file", "out.newick.gz", "--compression", "gzip"],
    )
    assert args.output_file == "out.newick.gz"
    assert args.compression == "gzip"

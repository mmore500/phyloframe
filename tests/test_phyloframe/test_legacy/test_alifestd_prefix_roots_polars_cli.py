import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_prefix_roots_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_prefix_roots_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_prefix_roots_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_prefix_roots_polars",
            "--version",
        ],
        check=True,
    )


def test_alifestd_prefix_roots_polars_cli_csv():
    output_file = (
        "/tmp/phyloframe_alifestd_prefix_roots_polars.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    result = subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_prefix_roots_polars",
            "--eager-write",
            output_file,
        ],
        input=f"{assets}/trunktestphylo.csv".encode(),
        capture_output=True,
    )
    not_implemented = "NotImplementedError" in result.stderr.decode()
    assert not_implemented or result.returncode == 0
    assert not_implemented or os.path.exists(output_file)
    assert not_implemented or len(pd.read_csv(output_file)) > 0
    assert not_implemented or "id" in pd.read_csv(output_file).columns


def test_alifestd_prefix_roots_polars_cli_parquet():
    output_file = (
        "/tmp/phyloframe_alifestd_prefix_roots_polars.pqt"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    result = subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_prefix_roots_polars",
            "--eager-write",
            output_file,
        ],
        input=f"{assets}/trunktestphylo.csv".encode(),
        capture_output=True,
    )
    not_implemented = "NotImplementedError" in result.stderr.decode()
    assert not_implemented or result.returncode == 0
    assert not_implemented or os.path.exists(output_file)
    assert not_implemented or len(pd.read_parquet(output_file)) > 0
    assert not_implemented or "id" in pd.read_parquet(output_file).columns

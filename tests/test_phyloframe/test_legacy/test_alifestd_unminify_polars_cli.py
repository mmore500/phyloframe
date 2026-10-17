import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _make_minified_file(minified_file: str) -> str:
    pathlib.Path(minified_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_minify_polars",
            minified_file,
        ],
        check=True,
        input=f"{assets}/trunktestphylo.csv".encode(),
    )
    return minified_file


def test_alifestd_unminify_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_unminify_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_unminify_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_unminify_polars",
            "--version",
        ],
        check=True,
    )


def test_alifestd_unminify_polars_cli_csv():
    minified_file = _make_minified_file(
        "/tmp/phyloframe_alifestd_unminify_polars_input_csv.pqt"  # nosec B108
    )
    output_file = "/tmp/phyloframe_alifestd_unminify_polars.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_unminify_polars",
            output_file,
        ],
        check=True,
        input=minified_file.encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns
    assert "ancestor_id" in result_df.columns
    assert "ancestor_list" not in result_df.columns
    assert "ancestor_offset" not in result_df.columns


def test_alifestd_unminify_polars_cli_parquet():
    minified_file = _make_minified_file(
        "/tmp/phyloframe_alifestd_unminify_polars_input_parquet.pqt"  # nosec B108
    )
    output_file = "/tmp/phyloframe_alifestd_unminify_polars.pqt"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_unminify_polars",
            output_file,
        ],
        check=True,
        input=minified_file.encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns
    assert "ancestor_id" in result_df.columns
    assert "ancestor_list" not in result_df.columns

import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_minify_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_minify_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_minify_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_minify_polars",
            "--version",
        ],
        check=True,
    )


def test_alifestd_minify_polars_cli_csv():
    output_file = "/tmp/phyloframe_alifestd_minify_polars.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_minify_polars",
            output_file,
        ],
        check=True,
        input=f"{assets}/trunktestphylo.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "ancestor_offset" in result_df.columns
    assert "id" not in result_df.columns


def test_alifestd_minify_polars_cli_parquet():
    output_file = "/tmp/phyloframe_alifestd_minify_polars.pqt"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_minify_polars",
            output_file,
        ],
        check=True,
        input=f"{assets}/trunktestphylo.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert "ancestor_offset" in result_df.columns
    assert "id" not in result_df.columns


def test_alifestd_minify_polars_cli_no_shrink_dtype():
    output_file = (
        "/tmp/phyloframe_alifestd_minify_polars_noshrink.pqt"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_minify_polars",
            "--no-shrink-dtype",
            output_file,
        ],
        check=True,
        input=f"{assets}/trunktestphylo.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert result_df["ancestor_offset"].dtype == "int64"

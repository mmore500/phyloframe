import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_pipe_unary_ops_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_pipe_unary_ops_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_pipe_unary_ops_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_pipe_unary_ops_polars",
            "--version",
        ],
        check=True,
    )


def test_alifestd_pipe_unary_ops_polars_cli_csv_no_ops():
    output_file = (
        "/tmp/phyloframe_alifestd_pipe_unary_ops_polars.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_pipe_unary_ops_polars",
            output_file,
        ],
        check=True,
        input=f"{assets}/example-standard-toy-asexual-phylogeny.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns


def test_alifestd_pipe_unary_ops_polars_cli_csv_single_op():
    output_file = (
        "/tmp/phyloframe_alifestd_pipe_unary_ops_polars_op.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_pipe_unary_ops_polars",
            "--op",
            "lambda df: df.with_columns(pl.lit(True).alias('test_col'))",
            output_file,
        ],
        check=True,
        input=f"{assets}/example-standard-toy-asexual-phylogeny.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "test_col" in result_df.columns


def test_alifestd_pipe_unary_ops_polars_cli_csv_multiple_ops():
    output_file = "/tmp/phyloframe_alifestd_pipe_unary_ops_polars_multi.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_pipe_unary_ops_polars",
            "--op",
            "lambda df: df.with_columns(pl.lit(1).alias('col_a'))",
            "--op",
            "lambda df: df.with_columns((pl.col('col_a') + 1).alias('col_b'))",
            output_file,
        ],
        check=True,
        input=f"{assets}/example-standard-toy-asexual-phylogeny.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "col_a" in result_df.columns
    assert "col_b" in result_df.columns

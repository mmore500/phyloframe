import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_pipe_unary_ops_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_pipe_unary_ops",
            "--help",
        ],
        check=True,
    )


def test_alifestd_pipe_unary_ops_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_pipe_unary_ops",
            "--version",
        ],
        check=True,
    )


def test_alifestd_pipe_unary_ops_cli_csv_no_ops():
    output_file = (
        "/tmp/phyloframe_alifestd_pipe_unary_ops.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_pipe_unary_ops",
            output_file,
        ],
        check=True,
        input=f"{assets}/example-standard-toy-asexual-phylogeny.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns


def test_alifestd_pipe_unary_ops_cli_csv_single_op():
    output_file = (
        "/tmp/phyloframe_alifestd_pipe_unary_ops_op.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_pipe_unary_ops",
            "--op",
            "lambda df: df.assign(test_col=True)",
            output_file,
        ],
        check=True,
        input=f"{assets}/example-standard-toy-asexual-phylogeny.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "test_col" in result_df.columns


def test_alifestd_pipe_unary_ops_cli_csv_multiple_ops():
    output_file = (
        "/tmp/phyloframe_alifestd_pipe_unary_ops_multi.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_pipe_unary_ops",
            "--op",
            "lambda df: df.assign(col_a=1)",
            "--op",
            "lambda df: df.assign(col_b=df['col_a'] + 1)",
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

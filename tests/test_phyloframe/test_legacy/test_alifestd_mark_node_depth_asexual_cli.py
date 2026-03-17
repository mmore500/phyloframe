import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_mark_node_depth_asexual_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_node_depth_asexual",
            "--help",
        ],
        check=True,
    )


def test_alifestd_mark_node_depth_asexual_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_node_depth_asexual",
            "--version",
        ],
        check=True,
    )


def test_alifestd_mark_node_depth_asexual_cli_csv():
    output_file = (
        "/tmp/phyloframe_alifestd_mark_node_depth_asexual.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_node_depth_asexual",
            output_file,
        ],
        check=True,
        input=f"{assets}/example-standard-toy-asexual-phylogeny.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "node_depth" in result_df.columns


def test_alifestd_mark_node_depth_asexual_cli_parquet():
    output_file = (
        "/tmp/phyloframe_alifestd_mark_node_depth_asexual.pqt"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_node_depth_asexual",
            output_file,
        ],
        check=True,
        input=f"{assets}/example-standard-toy-asexual-phylogeny.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert "node_depth" in result_df.columns


def test_alifestd_mark_node_depth_asexual_cli_mark_as():
    output_file = "/tmp/phyloframe_alifestd_mark_node_depth_asexual_mark_as.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_node_depth_asexual",
            "--mark-as",
            "custom_col_name",
            output_file,
        ],
        check=True,
        input=f"{assets}/example-standard-toy-asexual-phylogeny.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "custom_col_name" in result_df.columns

import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_downsample_tips_clade_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_downsample_tips_clade_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_downsample_tips_clade_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_downsample_tips_clade_polars",
            "--version",
        ],
        check=True,
    )


def test_alifestd_downsample_tips_clade_polars_cli_csv(tmp_path):
    output_file = str(
        tmp_path / "phyloframe_alifestd_downsample_tips_clade_polars.csv"
    )
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_downsample_tips_clade_polars",
            "-n",
            "4",
            "--seed",
            "1",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/trunktestphylo.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns


def test_alifestd_downsample_tips_clade_polars_cli_parquet(tmp_path):
    output_file = str(
        tmp_path / "phyloframe_alifestd_downsample_tips_clade_polars.pqt"
    )
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_downsample_tips_clade_polars",
            "-n",
            "4",
            "--seed",
            "1",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/trunktestphylo.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns


def test_alifestd_downsample_tips_clade_polars_cli_empty(tmp_path):
    output_file = str(
        tmp_path / "phyloframe_alifestd_downsample_tips_clade_polars.csv"
    )
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_downsample_tips_clade_polars",
            "-n",
            "10",
            output_file,
        ],
        check=True,
        input=f"{assets}/empty.csv".encode(),
    )
    assert os.path.exists(output_file)


def test_alifestd_downsample_tips_clade_polars_cli_ignore_topological_sensitivity():  # noqa: E501
    output_file = "/tmp/phyloframe_alifestd_downsample_tips_clade_polars_ignore.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_downsample_tips_clade_polars",
            "-n",
            "4",
            "--seed",
            "1",
            "--ignore-topological-sensitivity",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/trunktestphylo.csv".encode(),
    )
    assert os.path.exists(output_file)


def test_alifestd_downsample_tips_clade_polars_cli_drop_topological_sensitivity():
    output_file = "/tmp/phyloframe_alifestd_downsample_tips_clade_polars_drop.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_downsample_tips_clade_polars",
            "-n",
            "4",
            "--seed",
            "1",
            "--drop-topological-sensitivity",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/trunktestphylo.csv".encode(),
    )
    assert os.path.exists(output_file)

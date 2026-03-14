import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_mark_num_descendants_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_num_descendants_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_mark_num_descendants_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_num_descendants_polars",
            "--version",
        ],
        check=True,
    )


def test_alifestd_mark_num_descendants_polars_cli_csv():
    output_file = "/tmp/phyloframe_alifestd_mark_num_descendants_polars.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_num_descendants_polars",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/trunktestphylo.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "num_descendants" in result_df.columns


def test_alifestd_mark_num_descendants_polars_cli_parquet():
    output_file = "/tmp/phyloframe_alifestd_mark_num_descendants_polars.pqt"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_num_descendants_polars",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/trunktestphylo.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert "num_descendants" in result_df.columns

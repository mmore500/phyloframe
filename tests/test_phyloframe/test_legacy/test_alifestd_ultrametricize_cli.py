import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_ultrametricize_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_ultrametricize",
            "--help",
        ],
        check=True,
    )


def test_alifestd_ultrametricize_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_ultrametricize",
            "--version",
        ],
        check=True,
    )


def test_alifestd_ultrametricize_cli_csv():
    output_file = "/tmp/phyloframe_alifestd_ultrametricize.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_ultrametricize",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_tournamentselection.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "origin_time" in result_df.columns


def test_alifestd_ultrametricize_cli_parquet():
    output_file = "/tmp/phyloframe_alifestd_ultrametricize.pqt"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_ultrametricize",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_tournamentselection.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert "origin_time" in result_df.columns


def test_alifestd_ultrametricize_cli_method():
    output_file = (
        "/tmp/phyloframe_alifestd_ultrametricize_method.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_ultrametricize",
            "--method",
            "extend",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_tournamentselection.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "origin_time" in result_df.columns

import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_mark_clade_duration_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_clade_duration_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_mark_clade_duration_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_clade_duration_polars",
            "--version",
        ],
        check=True,
    )


def test_alifestd_mark_clade_duration_polars_cli_csv():
    output_file = (
        "/tmp/phyloframe_alifestd_mark_clade_duration_polars.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_clade_duration_polars",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "clade_duration" in result_df.columns


def test_alifestd_mark_clade_duration_polars_cli_parquet():
    output_file = (
        "/tmp/phyloframe_alifestd_mark_clade_duration_polars.pqt"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_clade_duration_polars",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert "clade_duration" in result_df.columns


def test_alifestd_mark_clade_duration_polars_cli_mark_as():
    output_file = "/tmp/phyloframe_alifestd_mark_clade_duration_polars_mark_as.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_clade_duration_polars",
            "--mark-as",
            "custom_col_name",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "custom_col_name" in result_df.columns


def test_alifestd_mark_clade_duration_polars_create_parser():
    from phyloframe.legacy._alifestd_mark_clade_duration_polars import (
        _create_parser,
    )

    parser = _create_parser()
    assert parser is not None

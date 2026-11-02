import os
import pathlib
import subprocess

import pandas as pd

from phyloframe.legacy._alifestd_coerce_chronological_consistency_polars import (
    _create_parser,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_coerce_chronological_consistency_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_coerce_chronological_consistency_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_coerce_chronological_consistency_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_coerce_chronological_consistency_polars",
            "--version",
        ],
        check=True,
    )


def test_alifestd_coerce_chronological_consistency_polars_cli_csv():
    output_file = "/tmp/phyloframe_alifestd_coerce_chronological_consistency_polars.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_coerce_chronological_consistency_polars",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns


def test_alifestd_coerce_chronological_consistency_polars_cli_parquet():
    output_file = "/tmp/phyloframe_alifestd_coerce_chronological_consistency_polars.pqt"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_coerce_chronological_consistency_polars",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns


def test_alifestd_coerce_chronological_consistency_polars_cli_epsilon():
    output_file = "/tmp/phyloframe_alifestd_coerce_chronological_consistency_polars_epsilon.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_coerce_chronological_consistency_polars",
            "--epsilon",
            "1.5",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns


def test_alifestd_coerce_chronological_consistency_polars_create_parser():
    parser = _create_parser()
    assert parser is not None

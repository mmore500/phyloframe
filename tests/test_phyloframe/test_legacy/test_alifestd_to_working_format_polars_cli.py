import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_to_working_format_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_to_working_format_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_to_working_format_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_to_working_format_polars",
            "--version",
        ],
        check=True,
    )


def test_alifestd_to_working_format_polars_cli_csv():
    output_file = (
        "/tmp/phyloframe_alifestd_to_working_format_polars.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_to_working_format_polars",
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


def test_alifestd_to_working_format_polars_cli_parquet():
    output_file = (
        "/tmp/phyloframe_alifestd_to_working_format_polars.pqt"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_to_working_format_polars",
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


def test_alifestd_to_working_format_polars_cli_keep_ancestor_list():
    output_file = "/tmp/phyloframe_alifestd_to_working_format_polars_keep.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_to_working_format_polars",
            "--keep-ancestor-list",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/example-standard-toy-asexual-phylogeny.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert "ancestor_list" in result_df.columns
    assert "ancestor_id" in result_df.columns


def test_alifestd_to_working_format_polars_cli_drops_ancestor_list():
    output_file = "/tmp/phyloframe_alifestd_to_working_format_polars_drop.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_to_working_format_polars",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/example-standard-toy-asexual-phylogeny.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert "ancestor_list" not in result_df.columns
    assert "ancestor_id" in result_df.columns


def test_alifestd_to_working_format_polars_create_parser():
    from phyloframe.legacy._alifestd_to_working_format_polars import (
        _create_parser,
    )

    parser = _create_parser()
    assert parser is not None

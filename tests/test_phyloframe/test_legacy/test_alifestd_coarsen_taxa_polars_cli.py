import os
import pathlib
import subprocess

import pandas as pd

from phyloframe.legacy._alifestd_coarsen_taxa_polars import _create_parser

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODULE = "phyloframe.legacy._alifestd_coarsen_taxa_polars"


def test_create_parser():
    args, __ = _create_parser().parse_known_args(
        ["--by", "a", "--by", "b", "--no-nans-equal"],
    )
    assert args.by == ["a", "b"]
    assert args.agg is None
    assert args.nans_equal is False
    assert args.nulls_equal is None


def test_cli_help():
    subprocess.run(  # nosec B603
        ["python3", "-m", MODULE, "--help"],
        check=True,
    )


def test_cli_version():
    subprocess.run(  # nosec B603
        ["python3", "-m", MODULE, "--version"],
        check=True,
    )


def test_cli_csv():
    output_file = (
        "/tmp/phyloframe_alifestd_coarsen_taxa_polars.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--by",
            "origin_time",
            "--ignore-topological-sensitivity",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    input_df = pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv")
    assert 0 < len(result_df) < len(input_df)
    assert "is_root" in result_df.columns


def test_cli_parquet_agg():
    output_file = (
        "/tmp/phyloframe_alifestd_coarsen_taxa_polars.pqt"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--by",
            "origin_time",
            "--agg",
            'dict(n=pl.len(), origin_time=pl.col("origin_time").first())',
            "--ignore-topological-sensitivity",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert sorted(result_df.columns) == [
        "ancestor_id",
        "ancestor_list",
        "id",
        "n",
        "origin_time",
    ]

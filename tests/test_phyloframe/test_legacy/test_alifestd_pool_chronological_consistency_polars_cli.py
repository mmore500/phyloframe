import os
import pathlib
import subprocess

import pandas as pd

from phyloframe.legacy._alifestd_pool_chronological_consistency_polars import (
    _create_parser,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODULE = "phyloframe.legacy._alifestd_pool_chronological_consistency_polars"


def test_create_parser():
    args, __ = _create_parser().parse_known_args([])
    assert args.weight_by is None
    assert args.pin_by is None

    args, __ = _create_parser().parse_known_args(
        ["--weight-by", "num_mrca_leaf_pairs", "--pin-by", "is_leaf"],
    )
    assert args.weight_by == "num_mrca_leaf_pairs"
    assert args.pin_by == "is_leaf"


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
    output_file = "/tmp/phyloframe_alifestd_pool_chronological_consistency_polars.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        ["python3", "-m", MODULE, "--eager-write", output_file],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "origin_time" in result_df.columns


def test_cli_parquet():
    output_file = "/tmp/phyloframe_alifestd_pool_chronological_consistency_polars.pqt"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--weight-by",
            "tot_orgs",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert "origin_time" in result_df.columns

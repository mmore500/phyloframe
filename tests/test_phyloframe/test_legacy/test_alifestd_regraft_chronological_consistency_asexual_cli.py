import os
import pathlib
import subprocess

import pandas as pd

from phyloframe.legacy._alifestd_regraft_chronological_consistency_asexual import (
    _create_parser,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODULE = "phyloframe.legacy._alifestd_regraft_chronological_consistency_asexual"


def test_create_parser():
    args, __ = _create_parser().parse_known_args(
        ["--drop-topological-sensitivity"],
    )
    assert args.drop_topological_sensitivity
    assert not args.ignore_topological_sensitivity


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
    output_file = "/tmp/phyloframe_alifestd_regraft_chronological_consistency_asexual.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        ["python3", "-m", MODULE, output_file],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "ancestor_id" in result_df.columns


def test_cli_parquet():
    output_file = "/tmp/phyloframe_alifestd_regraft_chronological_consistency_asexual.pqt"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--ignore-topological-sensitivity",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert "ancestor_id" in result_df.columns

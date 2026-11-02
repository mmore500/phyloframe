import os
import pathlib
import subprocess

import pandas as pd

from phyloframe.legacy._alifestd_shuffle_working_format import (
    _create_parser,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODULE = "phyloframe.legacy._alifestd_shuffle_working_format"


def test_create_parser():
    args, __ = _create_parser().parse_known_args(["--seed", "42"])
    assert args.seed == 42


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
        "/tmp/phyloframe_alifestd_shuffle_working_format.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        ["python3", "-m", MODULE, "--seed", "1", output_file],
        check=True,
        input=f"{assets}/nk_ecoeaselection.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert result_df["id"].tolist() == list(range(len(result_df)))


def test_cli_parquet():
    output_file = (
        "/tmp/phyloframe_alifestd_shuffle_working_format.pqt"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        ["python3", "-m", MODULE, output_file],
        check=True,
        input=f"{assets}/nk_ecoeaselection.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert result_df["id"].tolist() == list(range(len(result_df)))

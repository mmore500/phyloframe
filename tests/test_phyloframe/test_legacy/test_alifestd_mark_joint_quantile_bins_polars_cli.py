import subprocess

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import alifestd_mark_joint_quantile_bins_polars
from phyloframe.legacy._alifestd_mark_joint_quantile_bins_polars import (
    _create_parser,
)

from .test_alifestd_tgs_allmdc_estimate_polars import _make_cherries_polars

MODULE = "phyloframe.legacy._alifestd_mark_joint_quantile_bins_polars"


@pytest.fixture
def input_pqt(tmp_path):
    path = tmp_path / "input.pqt"
    _make_cherries_polars(2_000).with_columns(
        p=pl.col("id") % 3
    ).write_parquet(path)
    return str(path)


def test_cli_help():
    subprocess.run(  # nosec B603
        ["python3", "-m", MODULE, "--bin-by", "id", "--help"],
        check=True,
    )


def test_cli_version():
    subprocess.run(  # nosec B603
        ["python3", "-m", MODULE, "--version"],
        check=True,
    )


def test_create_parser():
    parser = _create_parser()
    assert parser is not None


def test_cli_matches_function(input_pqt, tmp_path):
    output_file = str(tmp_path / "output.pqt")
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--bin-by",
            "origin_time",
            "id",
            "--partition-by",
            "p",
            "--max-bins",
            "64",
            "--min-bin-size",
            "8",
            "--mark-as",
            "b",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_pqt.encode(),
    )
    expected = alifestd_mark_joint_quantile_bins_polars(
        pl.read_parquet(input_pqt),
        bin_by=["origin_time", "id"],
        partition_by="p",
        max_bins=64,
        min_bin_size=8,
        mark_as="b",
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(output_file), expected.to_pandas(), check_dtype=False
    )

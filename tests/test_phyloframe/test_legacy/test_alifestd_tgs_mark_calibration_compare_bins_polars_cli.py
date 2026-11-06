import subprocess

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_tgs_mark_calibration_compare_bins_polars,
)
from phyloframe.legacy._alifestd_tgs_mark_calibration_compare_bins_polars import (  # noqa: E501
    _create_parser,
)

from .test_alifestd_tgs_allmdc_estimate_polars import _make_cherries_polars

MODULE = "phyloframe.legacy._alifestd_tgs_mark_calibration_compare_bins_polars"


@pytest.fixture
def input_pqt(tmp_path):
    path = tmp_path / "input.pqt"
    _make_cherries_polars(2_000).write_parquet(path)
    return str(path)


def test_cli_help():
    subprocess.run(  # nosec B603
        ["python3", "-m", MODULE, "--neutral-traits", "n1", "--help"],
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
            "--neutral-traits",
            "n1",
            "n2",
            "--max-bins",
            "64",
            "--size-ratio",
            "4.0",
            "--min-neutral-gains",
            "5",
            "--mark-partition-as",
            "p",
            "--mark-bin-as",
            "b",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_pqt.encode(),
    )
    expected = alifestd_tgs_mark_calibration_compare_bins_polars(
        pl.read_parquet(input_pqt),
        neutral_traits=["n1", "n2"],
        max_bins=64,
        size_ratio=4.0,
        min_neutral_gains=5,
        mark_partition_as="p",
        mark_bin_as="b",
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(output_file), expected.to_pandas(), check_dtype=False
    )

import subprocess

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import alifestd_tgs_allmdc_bootstrap_polars
from phyloframe.legacy._alifestd_tgs_allmdc_bootstrap_polars import (
    _create_parser,
)

from ._impl import mark_tgs_calibration_columns
from .test_alifestd_tgs_allmdc_estimate_polars import _make_cherries_polars

MODULE = "phyloframe.legacy._alifestd_tgs_allmdc_bootstrap_polars"
NEUTRAL_TRAITS = ["n1", "n2", "n3", "n4"]


@pytest.fixture
def input_pqt(tmp_path):
    path = tmp_path / "input.pqt"
    mark_tgs_calibration_columns(
        _make_cherries_polars(2_000), neutral_traits=NEUTRAL_TRAITS
    ).write_parquet(path)
    return str(path)


def test_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--trait-transitions",
            "focal",
            "--neutral-traits",
            "n1",
            "n2",
            "--help",
        ],
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
            "--trait-transitions",
            "focal",
            *NEUTRAL_TRAITS,
            "--neutral-traits",
            *NEUTRAL_TRAITS,
            "--calibration-partition-by",
            "clade_size_class",
            "--calibration-compare-by-bins",
            "joint_quantile_bin",
            "--num-bootstrap",
            "3",
            "--bootstrap-seed",
            "1",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_pqt.encode(),
    )
    expected = alifestd_tgs_allmdc_bootstrap_polars(
        pl.read_parquet(input_pqt),
        trait_transitions=["focal", *NEUTRAL_TRAITS],
        neutral_traits=NEUTRAL_TRAITS,
        calibration_partition_by="clade_size_class",
        calibration_compare_by_bins="joint_quantile_bin",
        num_bootstrap=3,
        seed=1,
    )
    result_df = pd.read_parquet(output_file)
    assert result_df["replicate"].tolist() == [
        r for r in range(3) for __ in range(5)
    ]
    pd.testing.assert_frame_equal(
        result_df, expected.to_pandas(), check_dtype=False
    )

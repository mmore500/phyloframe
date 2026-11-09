import subprocess

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_tgs_estimate_polars,
)
from phyloframe.legacy._alifestd_tgs_estimate_polars import (
    _create_parser,
)

from ._impl import mark_tgs_calibration_columns
from .test_alifestd_tgs_estimate_polars import _make_cherries_polars

MODULE = "phyloframe.legacy._alifestd_tgs_estimate_polars"
NEUTRAL_TRAITS = ["n1", "n2", "n3", "n4"]


@pytest.fixture
def input_pqt(tmp_path):
    """Create a parquet phylogeny with trait and calibration columns."""
    path = tmp_path / "input.pqt"
    mark_tgs_calibration_columns(
        _make_cherries_polars(5_000), neutral_traits=NEUTRAL_TRAITS
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
            "--num-bootstrap",
            "2",
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


@pytest.mark.parametrize("ext", [".csv", ".pqt"])
def test_cli(input_pqt, ext, tmp_path):
    output_file = str(tmp_path / f"output{ext}")
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
            "5",
            "--bootstrap-seed",
            "1",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_pqt.encode(),
    )
    result_df = (
        pd.read_csv(output_file)
        if ext == ".csv"
        else pd.read_parquet(output_file)
    )
    assert result_df["trait"].tolist() == ["focal", *NEUTRAL_TRAITS]
    assert "selection_coef_p" in result_df.columns
    assert result_df.loc[0, "gain_ratio"] > 1.0


def test_cli_matches_function(input_pqt, tmp_path):
    output_file = str(tmp_path / "output.pqt")
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--trait-transitions",
            "focal",
            "n1",
            "n2",
            "--neutral-traits",
            "n1",
            "n2",
            "--calibration-compare-by-bins",
            "joint_quantile_bin",
            "--calibration-compare-by-vals",
            "origin_time",
            "--calibrate-neighborhood-fractions",
            "0.1",
            "1.0",
            "--calibrate-max-rate-ratio",
            "4.0",
            "--length-rtol",
            "1e-2",
            "--loss-rate",
            "0.001",
            "--selection-coef-starts",
            "0.0",
            "-0.5",
            "--growth-exponent-bound",
            "300.0",
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
    expected = alifestd_tgs_estimate_polars(
        pl.read_parquet(input_pqt),
        trait_transitions=["focal", "n1", "n2"],
        neutral_traits=["n1", "n2"],
        calibration_compare_by_bins="joint_quantile_bin",
        calibration_compare_by_vals=["origin_time"],
        calibrate_neighborhood_fractions=(0.1, 1.0),
        calibrate_max_rate_ratio=4.0,
        length_rtol=1e-2,
        loss_rate=0.001,
        selection_coef_starts=(0.0, -0.5),
        growth_exponent_bound=300.0,
        num_bootstrap=3,
        seed=1,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(output_file), expected.to_pandas(), check_dtype=False
    )

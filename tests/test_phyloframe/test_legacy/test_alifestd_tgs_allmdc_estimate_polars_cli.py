import subprocess

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy._alifestd_tgs_allmdc_estimate_polars import (
    _create_parser,
)

from .test_alifestd_tgs_allmdc_estimate_polars import _make_cherries_polars

MODULE = "phyloframe.legacy._alifestd_tgs_allmdc_estimate_polars"


@pytest.fixture
def input_pqt(tmp_path):
    """Create a parquet phylogeny with trait columns."""
    path = tmp_path / "input.pqt"
    _make_cherries_polars(5_000).write_parquet(path)
    return str(path)


def test_cli_help():
    subprocess.run(  # nosec B603
        ["python3", "-m", MODULE, "--trait-transitions", "focal", "--help"],
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
            "n1",
            "n2",
            "n3",
            "n4",
            "--neutral-traits",
            "n1",
            "n2",
            "n3",
            "n4",
            "--calibrate",
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
    result_df = (
        pd.read_csv(output_file)
        if ext == ".csv"
        else pd.read_parquet(output_file)
    )
    assert result_df["trait"].tolist() == ["focal", "n1", "n2", "n3", "n4"]
    assert "selection_coef_p" in result_df.columns
    assert result_df.loc[0, "gain_ratio"] > 1.0


def test_cli_matches_function(input_pqt, tmp_path):
    from phyloframe.legacy import alifestd_tgs_allmdc_estimate_polars

    output_file = str(tmp_path / "output.pqt")
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--trait-transitions",
            "focal",
            "n1",
            "--loss-rate",
            "0.001",
            "--calibrate-size-partition-ratio",
            "4",
            "--calibrate-size-partition-min-neutral-gains",
            "5",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_pqt.encode(),
    )
    expected = alifestd_tgs_allmdc_estimate_polars(
        pl.read_parquet(input_pqt),
        trait_transitions=["focal", "n1"],
        loss_rate=0.001,
        calibrate_size_partition_ratio=4.0,
        calibrate_size_partition_min_neutral_gains=5,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(output_file), expected.to_pandas(), check_dtype=False
    )

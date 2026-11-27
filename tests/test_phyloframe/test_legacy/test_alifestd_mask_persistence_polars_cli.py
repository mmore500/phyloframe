import os
import subprocess

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import alifestd_mask_persistence_polars
from phyloframe.legacy._alifestd_mask_persistence_polars import (
    _create_parser,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODULE = "phyloframe.legacy._alifestd_mask_persistence_polars"


@pytest.fixture
def input_csv_with_masks(tmp_path):
    """Create a CSV with mask columns from the test asset."""
    df = pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv")
    df["is_candidate"] = df["origin_time"].between(1000, 2000)
    df["is_extant"] = df["origin_time"] >= 3000
    path = tmp_path / "input_with_masks.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--candidate-mask",
            "is_candidate",
            "--extant-mask",
            "is_extant",
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
def test_cli(input_csv_with_masks, ext, tmp_path):
    output_file = str(tmp_path / f"output{ext}")
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--candidate-mask",
            "is_candidate",
            "--extant-mask",
            "is_extant",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_csv_with_masks.encode(),
    )
    result_df = (
        pd.read_csv(output_file)
        if ext == ".csv"
        else pd.read_parquet(output_file)
    )
    assert len(result_df) > 0
    col = "alifestd_mask_persistence_polars"
    assert col in result_df.columns
    assert result_df[col].any()
    assert result_df.loc[result_df[col], "is_candidate"].all()
    assert not result_df.loc[result_df["is_candidate"], col].all()


def test_cli_criterion_mark_as(input_csv_with_masks, tmp_path):
    output_file = str(tmp_path / "output.csv")
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--candidate-mask",
            "is_candidate",
            "--criterion-cherry",
            "depth",
            "--criterion-trunk",
            "depth",
            "--extant-mask",
            "is_extant",
            "--mark-as",
            "custom_col_name",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_csv_with_masks.encode(),
    )
    result_df = pd.read_csv(output_file)
    assert "custom_col_name" in result_df.columns
    assert result_df["custom_col_name"].any()


@pytest.mark.parametrize("name", ["criterion_cherry", "criterion_trunk"])
def test_cli_criterion_not_counted(name, input_csv_with_masks, tmp_path):
    output_file = str(tmp_path / "output.csv")
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--candidate-mask",
            "is_candidate",
            "--extant-mask",
            "is_extant",
            f"--{name.replace('_', '-')}",
            "",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_csv_with_masks.encode(),
    )
    expected = alifestd_mask_persistence_polars(
        pl.read_csv(input_csv_with_masks),
        candidate_mask="is_candidate",
        extant_mask="is_extant",
        **{name: None},
    )
    col = "alifestd_mask_persistence_polars"
    assert pd.read_csv(output_file)[col].tolist() == expected[col].to_list()
    assert expected[col].any()

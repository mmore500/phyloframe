import os
import subprocess

import pandas as pd
import pytest

from phyloframe.legacy import alifestd_mark_leaves
from phyloframe.legacy._alifestd_screen_trait_defined_clades_fitch_polars import (
    _create_parser,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODULE = "phyloframe.legacy._alifestd_screen_trait_defined_clades_fitch_polars"


@pytest.fixture
def input_csv_with_trait(tmp_path):
    """Create a CSV with trait mask columns from the test asset."""
    df = pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv")
    df = alifestd_mark_leaves(df)
    df["absent"] = df["is_leaf"] & (df["id"] % 7 != 0)
    df["present"] = df["is_leaf"] & (df["id"] % 7 == 0)
    path = tmp_path / "input_with_trait.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--mask-trait-absent",
            "absent",
            "--mask-trait-present",
            "present",
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
def test_cli(input_csv_with_trait, ext, tmp_path):
    # unique per test, as xdist workers run parametrizations concurrently
    output_file = str(tmp_path / f"output{ext}")
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--mask-trait-absent",
            "absent",
            "--mask-trait-present",
            "present",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_csv_with_trait.encode(),
    )
    assert os.path.exists(output_file)
    result_df = (
        pd.read_csv(output_file)
        if ext == ".csv"
        else pd.read_parquet(output_file)
    )
    assert len(result_df) > 0
    col = "alifestd_screen_trait_defined_clades_fitch_polars"
    assert col in result_df.columns
    assert result_df[col].any()


def test_cli_mark_as(input_csv_with_trait, tmp_path):
    output_file = str(tmp_path / "output.csv")
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--mask-trait-absent",
            "absent",
            "--mask-trait-present",
            "present",
            "--mark-as",
            "custom_col_name",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_csv_with_trait.encode(),
    )
    result_df = pd.read_csv(output_file)
    assert "custom_col_name" in result_df.columns

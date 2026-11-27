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


CANDIDATE_MASK = 'pl.col("origin_time").is_between(1000, 2000)'
EXTANT_MASK = 'pl.col("origin_time") >= 3000'


@pytest.fixture
def input_csv():
    return f"{assets}/nk_ecoeaselection-workingformat.csv"


def _expected(input_csv: str, **kwargs) -> pl.Series:
    return alifestd_mask_persistence_polars(
        pl.read_csv(input_csv),
        candidate_mask=pl.col("origin_time").is_between(1000, 2000),
        extant_mask=pl.col("origin_time") >= 3000,
        mark_as="expected",
        **kwargs,
    )["expected"]


def test_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--candidate-mask",
            CANDIDATE_MASK,
            "--extant-mask",
            EXTANT_MASK,
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
def test_cli(input_csv, ext, tmp_path):
    output_file = str(tmp_path / f"output{ext}")
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--candidate-mask",
            CANDIDATE_MASK,
            "--extant-mask",
            EXTANT_MASK,
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_csv.encode(),
    )
    result_df = (
        pd.read_csv(output_file)
        if ext == ".csv"
        else pd.read_parquet(output_file)
    )
    col = "alifestd_mask_persistence_polars"
    expected = _expected(input_csv)
    assert result_df[col].tolist() == expected.to_list()
    assert expected.any()
    assert not expected.all()


def test_cli_column_names(input_csv, tmp_path):
    output_file = str(tmp_path / "output.csv")
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--with-column",
            f"({CANDIDATE_MASK}).alias('is_candidate')",
            "--with-column",
            f"({EXTANT_MASK}).alias('is_extant')",
            "--candidate-mask",
            '"is_candidate"',
            "--criterion-cherry",
            '"depth"',
            "--criterion-trunk",
            '"depth"',
            "--extant-mask",
            '"is_extant"',
            "--mark-as",
            "custom_col_name",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_csv.encode(),
    )
    result_df = pd.read_csv(output_file)
    expected = _expected(
        input_csv, criterion_cherry="depth", criterion_trunk="depth"
    )
    assert result_df["custom_col_name"].tolist() == expected.to_list()
    assert expected.any()


@pytest.mark.parametrize(
    "name", ["criterion_cherry", "criterion_tiebreak", "criterion_trunk"]
)
def test_cli_criterion_none(name, input_csv, tmp_path):
    output_file = str(tmp_path / "output.csv")
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--candidate-mask",
            CANDIDATE_MASK,
            "--extant-mask",
            EXTANT_MASK,
            f"--{name.replace('_', '-')}",
            "None",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_csv.encode(),
    )
    expected = _expected(input_csv, **{name: None})
    col = "alifestd_mask_persistence_polars"
    assert pd.read_csv(output_file)[col].tolist() == expected.to_list()
    assert expected.any()

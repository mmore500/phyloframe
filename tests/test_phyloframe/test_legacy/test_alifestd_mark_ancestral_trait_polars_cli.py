import os
import pathlib
import subprocess

import pandas as pd
import pytest

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODULE = "phyloframe.legacy._alifestd_mark_ancestral_trait_polars"


@pytest.fixture
def input_csv_with_trait(tmp_path):
    """Create a CSV with trait columns from the test asset."""
    df = pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv")
    df["marked"] = df["depth"] == 300
    path = tmp_path / "input_with_trait.csv"
    df.to_csv(path, index=False)
    return str(path)


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


@pytest.mark.parametrize("ext", [".csv", ".pqt"])
def test_cli(input_csv_with_trait, ext):
    output_file = f"/tmp/phyloframe_alifestd_mark_ancestral_trait_polars{ext}"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--with-column",
            'pl.lit(-1).alias("default")',
            "--trait-default",
            "default",
            "--trait-mask",
            "marked",
            "--trait-values",
            "id",
            "--mark-as",
            "depth300_ancestor",
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
    assert len(result_df) == len(pd.read_csv(input_csv_with_trait))
    has_ancestor = result_df["depth"] >= 300
    assert (result_df.loc[~has_ancestor, "depth300_ancestor"] == -1).all()
    assert (result_df.loc[has_ancestor, "depth300_ancestor"] >= 0).all()


def test_cli_create_parser():
    from phyloframe.legacy._alifestd_mark_ancestral_trait_polars import (
        _create_parser,
    )

    parser = _create_parser()
    assert parser is not None
    args, __ = parser.parse_known_args(
        [
            "--trait-default",
            "d",
            "--trait-mask",
            "m",
            "--trait-values",
            "v",
            "--mark-as",
            "x",
        ]
    )
    assert (args.trait_default, args.trait_mask, args.trait_values) == (
        "d",
        "m",
        "v",
    )
    assert args.mark_as == "x"
    args, __ = parser.parse_known_args([])
    assert args.mark_as == "ancestral_trait"

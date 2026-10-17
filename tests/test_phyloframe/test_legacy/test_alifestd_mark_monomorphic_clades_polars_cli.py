import os
import pathlib
import subprocess

import pandas as pd
import pytest

from phyloframe.legacy import alifestd_mark_leaves

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODULE = "phyloframe.legacy._alifestd_mark_monomorphic_clades_polars"


@pytest.fixture
def input_csv_with_trait(tmp_path):
    """Create a CSV with a 'trait' column from the test asset."""
    df = pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv")
    df["trait"] = df["depth"] < 300
    df = alifestd_mark_leaves(df)
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
@pytest.mark.parametrize("trait_mask", [[], ["--trait-mask", "is_leaf"]])
def test_cli(input_csv_with_trait, ext, trait_mask):
    output_file = f"/tmp/phyloframe_alifestd_mark_monomorphic_clades_polars{ext}"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--by",
            "trait",
            *trait_mask,
            "--mark-as",
            "is_monomorphic",
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
    assert result_df["is_monomorphic"].any()


def test_cli_create_parser():
    from phyloframe.legacy._alifestd_mark_monomorphic_clades_polars import (
        _create_parser,
    )

    parser = _create_parser()
    assert parser is not None
    args, __ = parser.parse_known_args(
        ["--by", "a", "--by", "b", "--trait-mask", "is_leaf", "--mark-as", "m"]
    )
    assert args.by == ["a", "b"]
    assert args.trait_mask == "is_leaf"
    assert args.mark_as == "m"
    args, __ = parser.parse_known_args([])
    assert args.by == []
    assert args.trait_mask is None
    assert args.mark_as == "alifestd_mark_monomorphic_clades_polars"

import os
import pathlib
import subprocess

import pandas as pd
import pytest

from phyloframe.legacy._alifestd_mark_founder_id_polars import _create_parser

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODULE = "phyloframe.legacy._alifestd_mark_founder_id_polars"


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
def test_cli(ext):
    input_file = f"{assets}/nk_ecoeaselection-workingformat.csv"
    output_file = (
        f"/tmp/phyloframe_alifestd_mark_founder_id_polars{ext}"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--with-column",
            '(pl.col("depth") == 300).alias("is_founder")',
            "--founder-mask",
            "is_founder",
            "--mark-as",
            "depth300_founder",
            output_file,
        ],
        check=True,
        input=input_file.encode(),
    )
    assert os.path.exists(output_file)
    result_df = (
        pd.read_csv(output_file)
        if ext == ".csv"
        else pd.read_parquet(output_file)
    )
    assert len(result_df) == len(pd.read_csv(input_file))
    unfounded = result_df["depth"] < 300
    assert (
        result_df.loc[unfounded, "depth300_founder"]
        == result_df.loc[unfounded, "id"]
    ).all()


def test_cli_create_parser():
    parser = _create_parser()
    assert parser is not None
    args, __ = parser.parse_known_args(
        ["--founder-mask", "m", "--mark-as", "x"]
    )
    assert (args.founder_mask, args.mark_as) == ("m", "x")
    args, __ = parser.parse_known_args([])
    assert args.founder_mask is None
    assert args.mark_as == "founder_id"

import os
import pathlib
import subprocess

import pandas as pd

from phyloframe.legacy._alifestd_mark_lineage_affine_polars import (
    _create_parser,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODULE = "phyloframe.legacy._alifestd_mark_lineage_affine_polars"


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


def test_cli_csv():
    output_file = (
        "/tmp/phyloframe_alifestd_mark_lineage_affine_polars.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--scale",
            "origin_time",
            "--offset",
            "origin_time",
            "--mark-as",
            "blueberry",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "blueberry" in result_df.columns


def test_create_parser():
    parser = _create_parser()
    args, __ = parser.parse_known_args(["--scale", "m", "--offset", "a"])
    assert args.scale == "m"
    assert args.offset == "a"
    assert args.mark_as == "lineage_affine"

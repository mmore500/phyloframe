import os
import pathlib
import subprocess

import pandas as pd

from phyloframe.legacy._alifestd_mark_num_mrca_pairs_asexual import (
    _create_parser,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODULE = "phyloframe.legacy._alifestd_mark_num_mrca_pairs_asexual"


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
    output_file = "/tmp/phyloframe_alifestd_mark_num_mrca_pairs_asexual.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
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
    num_leaves = (~result_df["id"].isin(result_df["ancestor_id"])).sum()
    assert result_df["blueberry"].sum() == num_leaves * (num_leaves - 1) // 2


def test_create_parser():
    parser = _create_parser()
    args, __ = parser.parse_known_args([])
    assert args.mark_as == "num_mrca_pairs"

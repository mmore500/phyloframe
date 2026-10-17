import os
import pathlib
import subprocess

import pandas as pd

from phyloframe.legacy._alifestd_aggregate_phylogeny_partitions import (
    _create_parser,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODULE = "phyloframe.legacy._alifestd_aggregate_phylogeny_partitions"

INPUTS = [
    f"{assets}/nk_ecoeaselection-workingformat.csv",
    f"{assets}/trunktestphylo.csv",
]


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
    output_file = "/tmp/phyloframe_alifestd_aggregate_phylogeny_partitions.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--with-column",
            'pl.lit(filepath).alias("source")',
            "--partition-by",
            "source",
            "--how",
            "diagonal_relaxed",
            "--eager-write",
            output_file,
        ],
        check=True,
        input="\n".join(INPUTS).encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    input_dfs = [*map(pd.read_csv, INPUTS)]
    assert len(result_df) == sum(map(len, input_dfs))
    assert result_df["id"].is_unique
    assert result_df["source"].drop_duplicates().tolist() == INPUTS


def test_cli_create_parser():
    parser = _create_parser()
    assert parser is not None
    args, __ = parser.parse_known_args(
        ["--partition-by", "a", "--partition-by", "b"]
    )
    assert args.partition_by == ["a", "b"]

import subprocess

import pandas as pd

from phyloframe.legacy._alifestd_consolidate_roots_polars import _create_parser


def test_alifestd_consolidate_roots_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_consolidate_roots_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_consolidate_roots_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_consolidate_roots_polars",
            "--version",
        ],
        check=True,
    )


def test_alifestd_consolidate_roots_polars_cli_csv(tmp_path):
    input_file = str(tmp_path / "input.csv")
    output_file = str(tmp_path / "output.csv")
    # two roots, with ancestor_id only
    pd.DataFrame(
        {"id": [0, 1, 2, 3, 4], "ancestor_id": [0, 0, 2, 2, 3]},
    ).to_csv(input_file, index=False)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_consolidate_roots_polars",
            "--ignore-topological-sensitivity",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_file.encode(),
    )
    result_df = pd.read_csv(output_file)
    assert result_df["id"].tolist() == [0, 1, 3, 4]
    assert result_df["ancestor_id"].tolist() == [0, 0, 0, 3]


def test_alifestd_consolidate_roots_polars_create_parser():
    parser = _create_parser()
    assert parser is not None


def test_alifestd_consolidate_roots_polars_cli_ancestor_list(tmp_path):
    input_file = str(tmp_path / "input.csv")
    output_file = str(tmp_path / "output.csv")
    # standard alife csv, with ancestor_list only
    pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[none]", "[none]", "[1]", "[2]"],
        },
    ).to_csv(input_file, index=False)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_consolidate_roots_polars",
            "--ignore-topological-sensitivity",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_file.encode(),
    )
    result_df = pd.read_csv(output_file)
    assert result_df["id"].tolist() == [0, 2, 3]
    assert result_df["ancestor_list"].tolist() == ["[none]", "[0]", "[2]"]

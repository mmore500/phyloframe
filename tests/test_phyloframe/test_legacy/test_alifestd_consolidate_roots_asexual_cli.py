import subprocess

import pandas as pd

from phyloframe.legacy._alifestd_consolidate_roots_asexual import (
    _create_parser,
)


def test_alifestd_consolidate_roots_asexual_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_consolidate_roots_asexual",
            "--help",
        ],
        check=True,
    )


def test_alifestd_consolidate_roots_asexual_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_consolidate_roots_asexual",
            "--version",
        ],
        check=True,
    )


def test_alifestd_consolidate_roots_asexual_cli_csv(tmp_path):
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
            "phyloframe.legacy._alifestd_consolidate_roots_asexual",
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


def test_alifestd_consolidate_roots_asexual_create_parser():
    parser = _create_parser()
    assert parser is not None

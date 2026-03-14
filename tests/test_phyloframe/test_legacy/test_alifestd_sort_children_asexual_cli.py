import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_sort_children_asexual_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_sort_children_asexual",
            "--help",
        ],
        check=True,
    )


def test_alifestd_sort_children_asexual_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_sort_children_asexual",
            "--version",
        ],
        check=True,
    )


def test_alifestd_sort_children_asexual_cli_csv():
    # First add num_leaves column, then sort by it
    intermediate_file = (
        "/tmp/phyloframe_sort_children_intermediate.csv"  # nosec B108
    )
    output_file = (
        "/tmp/phyloframe_alifestd_sort_children_asexual.csv"  # nosec B108
    )
    pathlib.Path(intermediate_file).unlink(missing_ok=True)
    pathlib.Path(output_file).unlink(missing_ok=True)

    # Step 1: add num_leaves column
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_num_leaves_asexual",
            intermediate_file,
        ],
        check=True,
        input=f"{assets}/example-standard-toy-asexual-phylogeny.csv".encode(),
    )

    # Step 2: sort children by num_leaves
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_sort_children_asexual",
            "--criterion",
            "num_leaves",
            output_file,
        ],
        check=True,
        input=intermediate_file.encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns


def test_alifestd_sort_children_asexual_cli_reverse():
    intermediate_file = (
        "/tmp/phyloframe_sort_children_intermediate_rev.csv"  # nosec B108
    )
    output_file = (
        "/tmp/phyloframe_alifestd_sort_children_asexual_rev.csv"  # nosec B108
    )
    pathlib.Path(intermediate_file).unlink(missing_ok=True)
    pathlib.Path(output_file).unlink(missing_ok=True)

    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_num_leaves_asexual",
            intermediate_file,
        ],
        check=True,
        input=f"{assets}/example-standard-toy-asexual-phylogeny.csv".encode(),
    )

    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_sort_children_asexual",
            "--criterion",
            "num_leaves",
            "--reverse",
            output_file,
        ],
        check=True,
        input=intermediate_file.encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns

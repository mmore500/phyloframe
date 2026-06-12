import os
import pathlib
import subprocess

import pandas as pd


def test_alifestd_from_newick_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_from_newick",
            "--help",
        ],
        check=True,
    )


def test_alifestd_from_newick_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_from_newick",
            "--version",
        ],
        check=True,
    )


def test_alifestd_from_newick_cli_csv(tmp_path: pathlib.Path):
    input_file = tmp_path / "tree.newick"
    input_file.write_text("(Homo_sapiens,Pan_troglodytes)root;")
    output_file = tmp_path / "out.csv"
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_from_newick",
            "--input-file",
            str(input_file),
            "-o",
            str(output_file),
        ],
        check=True,
    )
    assert os.path.exists(output_file)
    result = pd.read_csv(output_file)
    assert "Homo_sapiens" in set(result["taxon_label"])


def test_alifestd_from_newick_cli_replace_unquoted(tmp_path: pathlib.Path):
    input_file = tmp_path / "tree.newick"
    input_file.write_text("(Homo_sapiens,'keep_me':1)root_node;")
    output_file = tmp_path / "out.csv"
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_from_newick",
            "--input-file",
            str(input_file),
            "-o",
            str(output_file),
            "--replace-unquoted",
            "{'_': ' '}",
        ],
        check=True,
    )
    result = pd.read_csv(output_file)
    labels = set(result["taxon_label"])
    # unquoted underscores -> spaces; quoted label left verbatim
    assert "Homo sapiens" in labels
    assert "root node" in labels
    assert "keep_me" in labels

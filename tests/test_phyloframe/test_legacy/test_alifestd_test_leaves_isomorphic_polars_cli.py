import os
import subprocess

import pytest

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_test_leaves_isomorphic_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_test_leaves_isomorphic_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_test_leaves_isomorphic_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_test_leaves_isomorphic_polars",
            "--version",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "input_file",
    [
        "nk_ecoeaselection.csv",
        "nk_lexicaseselection.csv",
        "nk_tournamentselection.csv",
    ],
)
def test_alifestd_test_leaves_isomorphic_polars_cli_self(input_file: str):
    cmd = [
        "python3",
        "-m",
        "phyloframe.legacy._alifestd_test_leaves_isomorphic_polars",
        f"{assets}/{input_file}",
        f"{assets}/{input_file}",
        "--taxon-label",
        "id",
    ]
    result = subprocess.run(cmd)  # nosec B603
    assert result.returncode == 0


def test_alifestd_test_leaves_isomorphic_polars_cli_different():
    cmd = [
        "python3",
        "-m",
        "phyloframe.legacy._alifestd_test_leaves_isomorphic_polars",
        f"{assets}/nk_ecoeaselection.csv",
        f"{assets}/nk_lexicaseselection.csv",
        "--taxon-label",
        "id",
    ]
    result = subprocess.run(cmd)  # nosec B603
    assert result.returncode == 1


def test_alifestd_test_leaves_isomorphic_polars_cli_tweaked():
    cmd = [
        "python3",
        "-m",
        "phyloframe.legacy._alifestd_test_leaves_isomorphic_polars",
        f"{assets}/nk_ecoeaselection.csv",
        f"{assets}/nk_ecoeaselection_tweaked.csv",
        "--taxon-label",
        "id",
    ]
    result = subprocess.run(cmd)  # nosec B603
    assert result.returncode == 1

import os
import subprocess

import pandas as pd
import pytest

from phyloframe.legacy import alifestd_to_working_format

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _prep_csv(src: str, tmp_path) -> str:
    """Convert ``src`` CSV to working format and write under ``tmp_path``.

    Drops ``ancestor_list`` because the polars implementation rejects it
    (matching other polars functions in the library).
    """
    df = alifestd_to_working_format(pd.read_csv(src))
    if "ancestor_list" in df.columns:
        df = df.drop(columns=["ancestor_list"])
    out = str(tmp_path / os.path.basename(src))
    df.to_csv(out, index=False)
    return out


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
def test_alifestd_test_leaves_isomorphic_polars_cli_self(
    input_file: str, tmp_path
):
    prepared = _prep_csv(f"{assets}/{input_file}", tmp_path)
    cmd = [
        "python3",
        "-m",
        "phyloframe.legacy._alifestd_test_leaves_isomorphic_polars",
        prepared,
        prepared,
        "--taxon-label",
        "id",
    ]
    result = subprocess.run(cmd)  # nosec B603
    assert result.returncode == 0


def test_alifestd_test_leaves_isomorphic_polars_cli_different(tmp_path):
    a = _prep_csv(f"{assets}/nk_ecoeaselection.csv", tmp_path)
    b = _prep_csv(f"{assets}/nk_lexicaseselection.csv", tmp_path)
    cmd = [
        "python3",
        "-m",
        "phyloframe.legacy._alifestd_test_leaves_isomorphic_polars",
        a,
        b,
        "--taxon-label",
        "id",
    ]
    result = subprocess.run(cmd)  # nosec B603
    assert result.returncode == 1


def test_alifestd_test_leaves_isomorphic_polars_cli_tweaked(tmp_path):
    a = _prep_csv(f"{assets}/nk_ecoeaselection.csv", tmp_path)
    b = _prep_csv(f"{assets}/nk_ecoeaselection_tweaked.csv", tmp_path)
    cmd = [
        "python3",
        "-m",
        "phyloframe.legacy._alifestd_test_leaves_isomorphic_polars",
        a,
        b,
        "--taxon-label",
        "id",
    ]
    result = subprocess.run(cmd)  # nosec B603
    assert result.returncode == 1

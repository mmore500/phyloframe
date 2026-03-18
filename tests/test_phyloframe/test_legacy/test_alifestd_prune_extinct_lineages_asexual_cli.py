import os
import pathlib
import subprocess

import numpy as np
import pandas as pd
import pytest

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


@pytest.fixture
def input_csv_with_extant(tmp_path):
    """Create a CSV with an 'extant' column from the test asset."""
    df = pd.read_csv(f"{assets}/nk_ecoeaselection.csv")
    df["extant"] = ~np.isfinite(df["destruction_time"])
    path = tmp_path / "input_with_extant.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_alifestd_prune_extinct_lineages_asexual_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_prune_extinct_lineages_asexual",
            "--help",
        ],
        check=True,
    )


def test_alifestd_prune_extinct_lineages_asexual_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_prune_extinct_lineages_asexual",
            "--version",
        ],
        check=True,
    )


def test_alifestd_prune_extinct_lineages_asexual_cli_csv(
    input_csv_with_extant,
):
    output_file = "/tmp/phyloframe_alifestd_prune_extinct_lineages_asexual.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_prune_extinct_lineages_asexual",
            output_file,
        ],
        check=True,
        input=input_csv_with_extant.encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns


def test_alifestd_prune_extinct_lineages_asexual_cli_parquet(
    input_csv_with_extant,
):
    output_file = "/tmp/phyloframe_alifestd_prune_extinct_lineages_asexual.pqt"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_prune_extinct_lineages_asexual",
            output_file,
        ],
        check=True,
        input=input_csv_with_extant.encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns


def test_alifestd_prune_extinct_lineages_asexual_cli_ignore_topological_sensitivity(
    input_csv_with_extant,
):
    output_file = "/tmp/phyloframe_alifestd_prune_extinct_lineages_asexual_ignore.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_prune_extinct_lineages_asexual",
            "--ignore-topological-sensitivity",
            output_file,
        ],
        check=True,
        input=input_csv_with_extant.encode(),
    )
    assert os.path.exists(output_file)


def test_alifestd_prune_extinct_lineages_asexual_cli_drop_topological_sensitivity(
    input_csv_with_extant,
):
    output_file = "/tmp/phyloframe_alifestd_prune_extinct_lineages_asexual_drop.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_prune_extinct_lineages_asexual",
            "--drop-topological-sensitivity",
            output_file,
        ],
        check=True,
        input=input_csv_with_extant.encode(),
    )
    assert os.path.exists(output_file)

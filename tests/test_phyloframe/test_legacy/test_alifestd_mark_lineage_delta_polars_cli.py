import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

MODULE = "phyloframe.legacy._alifestd_mark_lineage_delta_polars"


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
        "/tmp/phyloframe_alifestd_mark_lineage_delta_polars.csv"  # nosec B108
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--values",
            "origin_time",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "lineage_delta" in result_df.columns


def test_cli_mark_as():
    output_file = "/tmp/phyloframe_alifestd_mark_lineage_delta_polars_mark_as.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--values",
            "origin_time",
            "--mark-as",
            "custom_col_name",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "custom_col_name" in result_df.columns


def test_cli_basal_value():
    output_file = "/tmp/phyloframe_alifestd_mark_lineage_delta_polars_basal.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--values",
            "origin_time",
            "--basal-value",
            "0",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/nk_ecoeaselection-workingformat.csv".encode(),
    )
    result_df = pd.read_csv(output_file)
    is_root = result_df["id"] == result_df["ancestor_id"]
    assert (
        result_df.loc[is_root, "lineage_delta"]
        == result_df.loc[is_root, "origin_time"]
    ).all()

import os
import pathlib
import subprocess

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_delete_trunk_asexual_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_delete_trunk_asexual_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_delete_trunk_asexual_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_delete_trunk_asexual_polars",
            "--version",
        ],
        check=True,
    )


def test_alifestd_delete_trunk_asexual_polars_cli_csv():
    output_file = "/tmp/phyloframe_alifestd_delete_trunk_asexual_polars.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    result = subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_delete_trunk_asexual_polars",
            "--eager-write",
            "--ignore-topological-sensitivity",
            output_file,
        ],
        input=f"{assets}/trunktestphylo_with_trunk.csv".encode(),
        capture_output=True,
    )
    assert result.returncode != 0
    assert "NotImplementedError" in result.stderr.decode()


def test_alifestd_delete_trunk_asexual_polars_cli_parquet():
    output_file = "/tmp/phyloframe_alifestd_delete_trunk_asexual_polars.pqt"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    result = subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_delete_trunk_asexual_polars",
            "--eager-write",
            "--ignore-topological-sensitivity",
            output_file,
        ],
        input=f"{assets}/trunktestphylo_with_trunk.csv".encode(),
        capture_output=True,
    )
    assert result.returncode != 0
    assert "NotImplementedError" in result.stderr.decode()

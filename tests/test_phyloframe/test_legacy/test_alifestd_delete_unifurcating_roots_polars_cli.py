import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_delete_unifurcating_roots_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_delete_unifurcating_roots_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_delete_unifurcating_roots_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_delete_unifurcating_roots_polars",
            "--version",
        ],
        check=True,
    )


def test_alifestd_delete_unifurcating_roots_polars_cli_csv():
    output_file = "/tmp/phyloframe_alifestd_delete_unifurcating_roots_polars.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_delete_unifurcating_roots_polars",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/trunktestphylo.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0


def test_alifestd_delete_unifurcating_roots_polars_create_parser():
    from phyloframe.legacy._alifestd_delete_unifurcating_roots_polars import (
        _create_parser,
    )

    parser = _create_parser()
    assert parser is not None

import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_mark_sample_tips_lineage_asexual_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_sample_tips_lineage_asexual",
            "--help",
        ],
        check=True,
    )


def test_alifestd_mark_sample_tips_lineage_asexual_cli_csv():
    output_file = "/tmp/phyloframe_alifestd_mark_sample_tips_lineage_asexual.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_sample_tips_lineage_asexual",
            "-n",
            "1",
            "--seed",
            "42",
            "--criterion-delta",
            "id",
            "--criterion-target",
            "id",
            output_file,
        ],
        check=True,
        input=f"{assets}/example-standard-toy-asexual-phylogeny.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "id" in result_df.columns

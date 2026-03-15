import subprocess


def test_alifestd_mark_ot_mrca_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_ot_mrca_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_mark_ot_mrca_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_ot_mrca_polars",
            "--version",
        ],
        check=True,
    )

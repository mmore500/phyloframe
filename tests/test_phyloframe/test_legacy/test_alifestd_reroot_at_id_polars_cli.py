import subprocess


def test_alifestd_reroot_at_id_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_reroot_at_id_polars",
            "--new-root-id",
            "0",
            "--help",
        ],
        check=True,
    )


def test_alifestd_reroot_at_id_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_reroot_at_id_polars",
            "--new-root-id",
            "0",
            "--version",
        ],
        check=True,
    )


def test_alifestd_reroot_at_id_polars_create_parser():
    from phyloframe.legacy._alifestd_reroot_at_id_polars import _create_parser

    parser = _create_parser()
    assert parser is not None

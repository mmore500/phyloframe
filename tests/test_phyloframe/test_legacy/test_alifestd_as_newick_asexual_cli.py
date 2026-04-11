import bz2
import gzip
import lzma
import os
import pathlib
import subprocess
import typing
import zipfile

import pytest

_compression_ext_map = {
    "gzip": ".gz",
    "bz2": ".bz2",
    "xz": ".xz",
    "zip": ".zip",
}

_compression_opener = {
    "gzip": lambda p: gzip.open(p, "rt"),
    "bz2": lambda p: bz2.open(p, "rt"),
    "xz": lambda p: lzma.open(p, "rt"),
}

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_as_newick_asexual_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_as_newick_asexual",
            "--help",
        ],
        check=True,
    )


def test_alifestd_as_newick_asexual_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_as_newick_asexual",
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
@pytest.mark.parametrize(
    "taxon_label",
    [
        None,
        "id",
    ],
)
def test_alifestd_as_newick_asexual_cli_csv(
    input_file: str, taxon_label: typing.Optional[str]
):
    output_file = (
        "/tmp/phyloframe-as_newick_asexual_cli-"  # nosec B108
        f"{taxon_label}-{input_file}.newick"
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    cmd = [
        "python3",
        "-m",
        "phyloframe.legacy._alifestd_as_newick_asexual",
        "--input-file",
        f"{assets}/{input_file}",
        "-o",
        output_file,
    ]
    if taxon_label is not None:
        cmd.extend(["--taxon-label", taxon_label])
    subprocess.run(cmd, check=True)  # nosec B603
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0


@pytest.mark.parametrize(
    "input_file",
    [
        "nk_ecoeaselection.csv",
        "nk_lexicaseselection.csv",
        "nk_tournamentselection.csv",
    ],
)
@pytest.mark.parametrize(
    "taxon_label",
    [
        None,
        "id",
    ],
)
@pytest.mark.parametrize(
    "input_engine",
    [
        "pandas",
        "polars",
    ],
)
def test_alifestd_as_newick_asexual_cli_csv_engine(
    input_file: str,
    taxon_label: typing.Optional[str],
    input_engine: str,
):
    output_file = (
        "/tmp/phyloframe-as_newick_asexual_cli-"  # nosec B108
        f"{input_engine}-{taxon_label}-{input_file}.newick"
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    cmd = [
        "python3",
        "-m",
        "phyloframe.legacy._alifestd_as_newick_asexual",
        "--input-file",
        f"{assets}/{input_file}",
        "--input-engine",
        input_engine,
        "-o",
        output_file,
    ]
    if taxon_label is not None:
        cmd.extend(["--taxon-label", taxon_label])
    subprocess.run(cmd, check=True)  # nosec B603
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0


@pytest.mark.parametrize(
    "input_file",
    [
        "nk_ecoeaselection.csv",
        "nk_tournamentselection.csv",
    ],
)
def test_alifestd_as_newick_asexual_cli_csv_input_kwarg(
    input_file: str,
):
    output_file = (
        "/tmp/phyloframe-as_newick_asexual_cli-"  # nosec B108
        f"kwarg-{input_file}.newick"
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    cmd = [
        "python3",
        "-m",
        "phyloframe.legacy._alifestd_as_newick_asexual",
        "--input-file",
        f"{assets}/{input_file}",
        "--input-engine",
        "polars",
        "--input-kwarg",
        "infer_schema_length=None",
        "-o",
        output_file,
    ]
    subprocess.run(cmd, check=True)  # nosec B603
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0


@pytest.mark.parametrize("compression", ["gzip", "bz2", "xz", "zip"])
def test_alifestd_as_newick_asexual_cli_compression(compression: str):
    ext = _compression_ext_map[compression]
    output_file = (
        "/tmp/phyloframe-as_newick_asexual_cli-"  # nosec B108
        f"compression-{compression}.newick{ext}"
    )
    pathlib.Path(output_file).unlink(missing_ok=True)
    cmd = [
        "python3",
        "-m",
        "phyloframe.legacy._alifestd_as_newick_asexual",
        "--input-file",
        f"{assets}/nk_ecoeaselection.csv",
        "--compression",
        compression,
        "-o",
        output_file,
    ]
    subprocess.run(cmd, check=True)  # nosec B603
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0

    if compression == "zip":
        with zipfile.ZipFile(output_file, "r") as zf:
            content = zf.read(zf.namelist()[0]).decode()
    else:
        with _compression_opener[compression](output_file) as f:
            content = f.read()
    assert ";" in content


@pytest.mark.parametrize(
    "ext",
    [".gz", ".bz2", ".xz", ".zip", ".zst"],
)
def test_alifestd_as_newick_asexual_cli_extension_without_flag(ext: str):
    output_file = (
        "/tmp/phyloframe-as_newick_asexual_cli-"  # nosec B108
        f"ext-mismatch{ext}"
    )
    cmd = [
        "python3",
        "-m",
        "phyloframe.legacy._alifestd_as_newick_asexual",
        "--input-file",
        f"{assets}/nk_ecoeaselection.csv",
        "-o",
        output_file,
    ]
    result = subprocess.run(cmd, capture_output=True)  # nosec B603
    assert result.returncode != 0

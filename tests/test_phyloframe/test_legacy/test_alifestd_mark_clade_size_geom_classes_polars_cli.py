import subprocess

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import alifestd_mark_clade_size_geom_classes_polars
from phyloframe.legacy._alifestd_mark_clade_size_geom_classes_polars import (
    _create_parser,
)

MODULE = "phyloframe.legacy._alifestd_mark_clade_size_geom_classes_polars"


def _make_cherries_polars(num_cherries: int, seed: int = 1) -> pl.DataFrame:
    """Root with `num_cherries` cherries hanging off it, with numeric
    columns `n1` and `n2`."""
    rng = np.random.default_rng(seed)
    num_nodes = 1 + 3 * num_cherries
    inner = np.arange(1, num_nodes, 3)
    ancestor_id = np.zeros(num_nodes, dtype=np.int64)
    ancestor_id[inner + 1] = ancestor_id[inner + 2] = inner
    origin_time = np.zeros(num_nodes)
    origin_time[inner] = rng.uniform(0.5, 30, num_cherries)
    origin_time[inner + 1] = origin_time[inner] + rng.uniform(
        1, 60, num_cherries
    )
    origin_time[inner + 2] = origin_time[inner] + rng.uniform(
        1, 60, num_cherries
    )
    return pl.DataFrame(
        {
            "id": np.arange(num_nodes),
            "ancestor_id": ancestor_id,
            "origin_time": origin_time,
            "n1": rng.integers(0, 2, num_nodes),
            "n2": rng.integers(0, 2, num_nodes),
        },
    )


@pytest.fixture
def input_pqt(tmp_path):
    path = tmp_path / "input.pqt"
    _make_cherries_polars(2_000).write_parquet(path)
    return str(path)


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


def test_create_parser():
    parser = _create_parser()
    assert parser is not None


def test_cli_matches_function(input_pqt, tmp_path):
    output_file = str(tmp_path / "output.pqt")
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            MODULE,
            "--size-ratio",
            "4.0",
            "--weight-by",
            "n1",
            "n2",
            "--min-class-weight",
            "5",
            "--mark-as",
            "c",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=input_pqt.encode(),
    )
    expected = alifestd_mark_clade_size_geom_classes_polars(
        pl.read_parquet(input_pqt),
        size_ratio=4.0,
        weight_by=["n1", "n2"],
        min_class_weight=5,
        mark_as="c",
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(output_file), expected.to_pandas(), check_dtype=False
    )

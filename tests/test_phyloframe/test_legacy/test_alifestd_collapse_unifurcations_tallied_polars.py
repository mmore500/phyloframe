import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_collapse_unifurcations_tallied_asexual,
    alifestd_collapse_unifurcations_tallied_polars,
    alifestd_to_working_format,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _make_df() -> pl.DataFrame:
    """Tree:
    0 (root)
    +-- 1 (unifurcation)
        +-- 2
            +-- 3 (unifurcation)
            |   +-- 5 (unifurcation)
            |       +-- 6
            +-- 4
    """
    return pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 1, 2, 2, 3, 5],
            "tally": pl.Series([0, 4, 1, 0, 2, 0, 3], dtype=pl.UInt32),
        },
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple_tree(apply: typing.Callable):
    res = (
        alifestd_collapse_unifurcations_tallied_polars(
            apply(_make_df()), tally_as="tally"
        )
        .lazy()
        .collect()
    )
    assert dict(zip(res["id"], res["tally"])) == {0: 0, 2: 6, 4: 2, 6: 5}
    assert res["tally"].dtype == pl.UInt32


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/collapse_unifurcations_testphylo.csv"),
    ],
)
def test_matches_asexual(phylogeny_df: pd.DataFrame):
    df = alifestd_to_working_format(phylogeny_df).drop(
        columns=["ancestor_list"], errors="ignore"
    )
    expected = alifestd_collapse_unifurcations_tallied_asexual(df)
    res = alifestd_collapse_unifurcations_tallied_polars(pl.from_pandas(df))
    assert res["id"].to_list() == expected["id"].tolist()
    assert (
        res["collapsed_unifurcations"].to_list()
        == expected["collapsed_unifurcations"].tolist()
    )
    assert res["collapsed_unifurcations"].sum() == len(df) - len(res)


def test_missing_tally_starts_from_zero():
    res = alifestd_collapse_unifurcations_tallied_polars(
        _make_df().drop("tally")
    )
    assert dict(zip(res["id"], res["collapsed_unifurcations"])) == {
        0: 0,
        2: 1,
        4: 0,
        6: 2,
    }


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_tally_not_implemented(missing: typing.Optional[float]):
    df = _make_df().with_columns(
        pl.when(pl.col("id") == 2)
        .then(pl.lit(missing, dtype=pl.Float64))
        .otherwise(pl.col("tally"))
        .alias("tally"),
    )
    with pytest.raises(NotImplementedError):
        alifestd_collapse_unifurcations_tallied_polars(df, tally_as="tally")


def test_empty():
    df = pl.DataFrame(
        {"id": [], "ancestor_id": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64},
    )
    res = alifestd_collapse_unifurcations_tallied_polars(df)
    assert "collapsed_unifurcations" in res.columns
    assert len(res) == 0

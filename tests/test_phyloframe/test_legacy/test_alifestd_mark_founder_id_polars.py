import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_aggregate_phylogenies,
)
from phyloframe.legacy import (
    alifestd_mark_founder_id_polars as alifestd_mark_founder_id_polars_,
)
from phyloframe.legacy import (
    alifestd_mark_root_id_polars,
    alifestd_to_working_format,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_founder_id_polars = enforce_dtype_stability_polars(
    alifestd_mark_founder_id_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _make_toy_df() -> pl.DataFrame:
    #        0
    #      /   \
    #     1     2
    #    / \    |
    #   3   4   5
    return pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_id": [0, 0, 0, 1, 1, 2],
            "is_founder": [False, True, False, True, False, True],
        }
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_founder_id_polars_toy(apply: typing.Callable):
    df = _make_toy_df()
    res = (
        alifestd_mark_founder_id_polars(apply(df), founder_mask="is_founder")
        .lazy()
        .collect()
    )
    assert res.columns == [*df.columns, "founder_id"]
    assert res.drop("founder_id").equals(df)
    # oldest founder along lineage wins; unfounded taxa are own founder
    assert res["founder_id"].to_list() == [0, 1, 2, 1, 1, 5]


def test_alifestd_mark_founder_id_polars_expr_mark_as():
    df = _make_toy_df()
    res = alifestd_mark_founder_id_polars(
        df, founder_mask=pl.col("id") == 2, mark_as="clade"
    )
    assert res["clade"].to_list() == [0, 1, 2, 3, 4, 2]


def test_alifestd_mark_founder_id_polars_none_marked():
    df = _make_toy_df()
    res = alifestd_mark_founder_id_polars(df, founder_mask=pl.lit(False))
    assert res["founder_id"].to_list() == df["id"].to_list()


def test_alifestd_mark_founder_id_polars_empty():
    df = _make_toy_df().head(0)
    res = alifestd_mark_founder_id_polars(df, founder_mask="is_founder")
    assert res.is_empty()
    assert res.columns == [*df.columns, "founder_id"]


def test_alifestd_mark_founder_id_polars_missing_column():
    with pytest.raises(ValueError):
        alifestd_mark_founder_id_polars(_make_toy_df(), founder_mask="missing")


def test_alifestd_mark_founder_id_polars_noncontiguous():
    df = _make_toy_df().with_columns(
        id=pl.col("id") * 10 + 3,
        ancestor_id=pl.col("ancestor_id") * 10 + 3,
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_founder_id_polars(df, founder_mask="is_founder")


_assets = [
    pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
    pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
    pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    alifestd_aggregate_phylogenies(
        [
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        ]
    ),
]


@pytest.mark.parametrize("phylogeny_df", _assets)
def test_alifestd_mark_founder_id_polars_all_marked_is_root_id(
    phylogeny_df: pd.DataFrame,
):
    phylogeny_df = pl.from_pandas(
        alifestd_to_working_format(phylogeny_df)[["id", "ancestor_id"]]
    )
    res = alifestd_mark_founder_id_polars(
        phylogeny_df, founder_mask=pl.lit(True)
    )
    expected = alifestd_mark_root_id_polars(phylogeny_df)["root_id"]
    assert res["founder_id"].to_list() == expected.to_list()


@pytest.mark.parametrize("phylogeny_df", _assets)
@pytest.mark.parametrize("p_founder", [0.01, 0.2])
def test_alifestd_mark_founder_id_polars_reference(
    phylogeny_df: pd.DataFrame, p_founder: float
):
    phylogeny_df = alifestd_to_working_format(phylogeny_df)[
        ["id", "ancestor_id"]
    ].copy()
    n = len(phylogeny_df)
    is_founder = np.random.default_rng(1).random(n) < p_founder
    phylogeny_df["is_founder"] = is_founder

    res = alifestd_mark_founder_id_polars(
        pl.from_pandas(phylogeny_df), founder_mask="is_founder"
    )

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    expected = []
    for id_ in range(n):
        founder, cur = id_, id_
        while True:  # walk to root, keeping oldest founder
            if is_founder[cur]:
                founder = cur
            if ancestor_ids[cur] == cur:
                break
            cur = ancestor_ids[cur]
        expected.append(founder)
    assert res["founder_id"].to_list() == expected

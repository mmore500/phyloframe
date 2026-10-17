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
    alifestd_mark_ancestral_trait_polars as alifestd_mark_ancestral_trait_polars_,
)
from phyloframe.legacy import (
    alifestd_to_working_format,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_ancestral_trait_polars = enforce_dtype_stability_polars(
    alifestd_mark_ancestral_trait_polars_
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
        }
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_ancestral_trait_polars_toy(apply: typing.Callable):
    df = _make_toy_df()
    res = (
        alifestd_mark_ancestral_trait_polars(
            apply(df),
            trait_default=-1,
            trait_mask=np.array([False, True, False, True, False, True]),
            trait_value=np.array([10, 11, 12, 13, 14, 15]),
        )
        .lazy()
        .collect()
    )
    assert res.columns == [*df.columns, "ancestral_trait"]
    # oldest marked taxon along lineage wins, e.g., 3 takes 1's value
    assert res["ancestral_trait"].to_list() == [-1, 11, -1, 11, 11, 15]


def test_alifestd_mark_ancestral_trait_polars_array_default():
    df = _make_toy_df()
    res = alifestd_mark_ancestral_trait_polars(
        df,
        mark_as="group",
        trait_default=np.arange(6) * 100,
        trait_mask=np.array([False, True, False, False, False, False]),
        trait_value=np.arange(6),
    )
    assert res["group"].to_list() == [0, 1, 200, 1, 1, 500]


def test_alifestd_mark_ancestral_trait_polars_str():
    df = _make_toy_df()
    res = alifestd_mark_ancestral_trait_polars(
        df,
        trait_default="none",
        trait_mask=np.array([True, False, False, False, False, False]),
        trait_value=np.array(list("abcdef")),
    )
    assert res["ancestral_trait"].to_list() == ["a"] * 6


def test_alifestd_mark_ancestral_trait_polars_none_marked():
    df = _make_toy_df()
    res = alifestd_mark_ancestral_trait_polars(
        df,
        trait_default=7,
        trait_mask=np.zeros(6, dtype=bool),
        trait_value=np.arange(6),
    )
    assert res["ancestral_trait"].to_list() == [7] * 6


def test_alifestd_mark_ancestral_trait_polars_ancestor_list():
    df = _make_toy_df().drop("ancestor_id")
    df = df.with_columns(
        ancestor_list=pl.Series(["[none]", "[0]", "[0]", "[1]", "[1]", "[2]"]),
    )
    res = alifestd_mark_ancestral_trait_polars(
        df,
        trait_default=-1,
        trait_mask=np.array([False, False, True, False, False, False]),
        trait_value=np.arange(6),
    )
    assert res["ancestral_trait"].to_list() == [-1, -1, 2, -1, -1, 2]


def test_alifestd_mark_ancestral_trait_polars_empty():
    df = _make_toy_df().head(0)
    res = alifestd_mark_ancestral_trait_polars(
        df,
        trait_default=-1,
        trait_mask=np.array([], dtype=bool),
        trait_value=np.array([], dtype=int),
    )
    assert res.is_empty()
    assert res.columns == [*df.columns, "ancestral_trait"]


def test_alifestd_mark_ancestral_trait_polars_noncontiguous():
    df = _make_toy_df().with_columns(
        id=pl.col("id") * 10 + 3,
        ancestor_id=pl.col("ancestor_id") * 10 + 3,
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_ancestral_trait_polars(
            df,
            trait_default=-1,
            trait_mask=np.ones(6, dtype=bool),
            trait_value=np.arange(6),
        )


def test_alifestd_mark_ancestral_trait_polars_unsorted():
    df = pl.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 2, 0]})
    with pytest.raises(NotImplementedError):
        alifestd_mark_ancestral_trait_polars(
            df,
            trait_default=-1,
            trait_mask=np.ones(3, dtype=bool),
            trait_value=np.arange(3),
        )


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
        alifestd_aggregate_phylogenies(
            [
                pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
                pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
            ]
        ),
    ],
)
@pytest.mark.parametrize("p_marked", [0.0, 0.01, 0.2, 1.0])
@pytest.mark.parametrize("array_default", [False, True])
def test_alifestd_mark_ancestral_trait_polars_reference(
    phylogeny_df: pd.DataFrame, p_marked: float, array_default: bool
):
    phylogeny_df = alifestd_to_working_format(phylogeny_df)[
        ["id", "ancestor_id"]
    ]
    n = len(phylogeny_df)
    rng = np.random.default_rng(1)
    trait_mask = rng.random(n) < p_marked
    trait_value = rng.integers(0, 1000, n)
    trait_default = -np.arange(n) - 1 if array_default else -1

    res = alifestd_mark_ancestral_trait_polars(
        pl.from_pandas(phylogeny_df),
        trait_default=trait_default,
        trait_mask=trait_mask,
        trait_value=trait_value,
    )

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    expected = []
    for id_ in range(n):
        value = trait_default[id_] if array_default else trait_default
        cur = id_
        while True:  # walk to root, keeping oldest marked taxon's value
            if trait_mask[cur]:
                value = trait_value[cur]
            if ancestor_ids[cur] == cur:
                break
            cur = ancestor_ids[cur]
        expected.append(value)
    assert res["ancestral_trait"].to_list() == expected

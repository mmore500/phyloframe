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
            "marked": [False, True, False, True, False, True],
            "value": [10, 11, 12, 13, 14, 15],
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
            trait_default=pl.lit(-1),
            trait_mask="marked",
            trait_values="value",
        )
        .lazy()
        .collect()
    )
    assert res.columns == [*df.columns, "ancestral_trait"]
    assert res.drop("ancestral_trait").equals(df)
    # oldest marked taxon along lineage wins, e.g., 3 takes 1's value
    assert res["ancestral_trait"].to_list() == [-1, 11, -1, 11, 11, 15]


def test_alifestd_mark_ancestral_trait_polars_expr():
    df = _make_toy_df()
    res = alifestd_mark_ancestral_trait_polars(
        df,
        mark_as="group",
        trait_default=pl.col("id") * 100,
        trait_mask=pl.col("id") == 1,
        trait_values=pl.col("id"),
    )
    assert res["group"].to_list() == [0, 1, 200, 1, 1, 500]


def test_alifestd_mark_ancestral_trait_polars_column_default():
    df = _make_toy_df().with_columns(default=pl.col("value") * -1)
    res = alifestd_mark_ancestral_trait_polars(
        df, trait_default="default", trait_mask="marked", trait_values="value"
    )
    assert res["ancestral_trait"].to_list() == [-10, 11, -12, 11, 11, 15]


def test_alifestd_mark_ancestral_trait_polars_str():
    df = _make_toy_df().with_columns(label=pl.Series(list("abcdef")))
    res = alifestd_mark_ancestral_trait_polars(
        df,
        trait_default=pl.lit("none"),
        trait_mask=pl.col("id") == 2,
        trait_values="label",
    )
    assert res["ancestral_trait"].to_list() == [
        "none",
        "none",
        "c",
        "none",
        "none",
        "c",
    ]


def test_alifestd_mark_ancestral_trait_polars_none_marked():
    df = _make_toy_df()
    res = alifestd_mark_ancestral_trait_polars(
        df,
        trait_default=pl.lit(7),
        trait_mask=pl.lit(False),
        trait_values="value",
    )
    assert res["ancestral_trait"].to_list() == [7] * 6


def test_alifestd_mark_ancestral_trait_polars_ancestor_list():
    df = _make_toy_df().drop("ancestor_id")
    df = df.with_columns(
        ancestor_list=pl.Series(["[none]", "[0]", "[0]", "[1]", "[1]", "[2]"]),
    )
    res = alifestd_mark_ancestral_trait_polars(
        df,
        trait_default=pl.lit(-1),
        trait_mask="marked",
        trait_values="value",
    )
    assert res["ancestral_trait"].to_list() == [-1, 11, -1, 11, 11, 15]


def test_alifestd_mark_ancestral_trait_polars_empty():
    df = _make_toy_df().head(0)
    res = alifestd_mark_ancestral_trait_polars(
        df, trait_default=pl.lit(-1), trait_mask="marked", trait_values="value"
    )
    assert res.is_empty()
    assert res.columns == [*df.columns, "ancestral_trait"]


@pytest.mark.parametrize(
    "arg", ["trait_default", "trait_mask", "trait_values"]
)
def test_alifestd_mark_ancestral_trait_polars_missing_column(arg: str):
    kwargs = dict(
        trait_default=pl.lit(-1), trait_mask="marked", trait_values="value"
    )
    kwargs[arg] = "missing"
    with pytest.raises(ValueError):
        alifestd_mark_ancestral_trait_polars(_make_toy_df(), **kwargs)


def test_alifestd_mark_ancestral_trait_polars_noncontiguous():
    df = _make_toy_df().with_columns(
        id=pl.col("id") * 10 + 3,
        ancestor_id=pl.col("ancestor_id") * 10 + 3,
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_ancestral_trait_polars(
            df,
            trait_default=pl.lit(-1),
            trait_mask="marked",
            trait_values="value",
        )


def test_alifestd_mark_ancestral_trait_polars_unsorted():
    df = pl.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 2, 0], "marked": [True] * 3},
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_ancestral_trait_polars(
            df,
            trait_default=pl.lit(-1),
            trait_mask="marked",
            trait_values="id",
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
@pytest.mark.parametrize("column_default", [False, True])
def test_alifestd_mark_ancestral_trait_polars_reference(
    phylogeny_df: pd.DataFrame, p_marked: float, column_default: bool
):
    phylogeny_df = alifestd_to_working_format(phylogeny_df)[
        ["id", "ancestor_id"]
    ].copy()
    n = len(phylogeny_df)
    rng = np.random.default_rng(1)
    phylogeny_df["marked"] = rng.random(n) < p_marked
    phylogeny_df["value"] = rng.integers(0, 1000, n)
    phylogeny_df["default"] = -np.arange(n) - 1

    res = alifestd_mark_ancestral_trait_polars(
        pl.from_pandas(phylogeny_df),
        trait_default="default" if column_default else pl.lit(-1),
        trait_mask="marked",
        trait_values="value",
    )

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    marked = phylogeny_df["marked"].to_numpy()
    values = phylogeny_df["value"].to_numpy()
    expected = []
    for id_ in range(n):
        value = phylogeny_df["default"].iat[id_] if column_default else -1
        cur = id_
        while True:  # walk to root, keeping oldest marked taxon's value
            if marked[cur]:
                value = values[cur]
            if ancestor_ids[cur] == cur:
                break
            cur = ancestor_ids[cur]
        expected.append(value)
    assert res["ancestral_trait"].to_list() == expected

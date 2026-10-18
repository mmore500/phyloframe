import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_aggregate_phylogenies,
    alifestd_mark_leaves,
)
from phyloframe.legacy import (
    alifestd_mark_monomorphic_clades_polars as alifestd_mark_monomorphic_clades_polars_,
)
from phyloframe.legacy import (
    alifestd_mask_monomorphic_clades_asexual,
    alifestd_to_working_format,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_monomorphic_clades_polars = enforce_dtype_stability_polars(
    alifestd_mark_monomorphic_clades_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _add_inherited_traits(
    phylogeny_df: pd.DataFrame, seed: int = 1
) -> pd.DataFrame:
    """Add trait columns that are inherited with occasional mutation.

    Requires working format.
    """
    rng = np.random.default_rng(seed)
    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    trait_a = np.zeros(len(phylogeny_df), dtype=np.int64)
    trait_b = np.zeros(len(phylogeny_df), dtype=object)
    trait_b[:] = "x"
    for idx, ancestor_id in enumerate(ancestor_ids):
        if idx == ancestor_id:
            continue
        trait_a[idx] = trait_a[ancestor_id] + (rng.random() < 0.05)
        trait_b[idx] = (
            rng.choice(["x", "y"])
            if rng.random() < 0.05
            else trait_b[ancestor_id]
        )
    phylogeny_df["trait_a"] = trait_a
    phylogeny_df["trait_b"] = trait_b.astype(str)
    return alifestd_mark_leaves(phylogeny_df, mutate=True)


def _make_toy_df() -> pl.DataFrame:
    #        0 (a)
    #      /       \
    #   1 (a)      2 (b)
    #   /   \      /   \
    # 3(a) 4(a)  5(b)  6(c)
    #             |
    #            7(b)
    return pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2, 5],
            "trait": ["a", "a", "b", "a", "a", "b", "c", "b"],
            "is_leaf": [False, False, False, True, True, False, True, True],
        }
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_monomorphic_clades_polars_toy(apply: typing.Callable):
    df = _make_toy_df()
    res = (
        alifestd_mark_monomorphic_clades_polars(apply(df), "trait")
        .lazy()
        .collect()
    )
    assert res.columns == [
        *df.columns,
        "alifestd_mark_monomorphic_clades_polars",
    ]
    assert res.drop("alifestd_mark_monomorphic_clades_polars").equals(df)
    assert res["alifestd_mark_monomorphic_clades_polars"].to_list() == [
        False,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_alifestd_mark_monomorphic_clades_polars_multiple_by():
    df = _make_toy_df().with_columns(
        other=pl.Series([0, 0, 0, 0, 1, 0, 0, 0]),
    )
    res = alifestd_mark_monomorphic_clades_polars(df, ["trait", "other"])
    assert res["alifestd_mark_monomorphic_clades_polars"].to_list() == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_alifestd_mark_monomorphic_clades_polars_expr():
    df = _make_toy_df()
    res = alifestd_mark_monomorphic_clades_polars(
        df, pl.col("trait").is_in(["a", "b"])
    )
    assert res["alifestd_mark_monomorphic_clades_polars"].to_list() == [
        False,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_alifestd_mark_monomorphic_clades_polars_nulls():
    df = _make_toy_df().with_columns(
        trait=pl.when(pl.col("id").is_in([1, 3, 4]))
        .then(pl.lit(None))
        .otherwise(pl.col("trait")),
    )
    res = alifestd_mark_monomorphic_clades_polars(df, "trait")
    assert res["alifestd_mark_monomorphic_clades_polars"][1]


@pytest.mark.parametrize(
    "trait_mask", ["is_leaf", pl.col("id").is_in([3, 4, 6, 7])]
)
def test_alifestd_mark_monomorphic_clades_polars_trait_mask(trait_mask):
    df = _make_toy_df().with_columns(
        trait=pl.when(pl.col("id").is_in([0, 1, 5]))
        .then(pl.lit("z"))
        .otherwise(pl.col("trait")),
    )
    res = alifestd_mark_monomorphic_clades_polars(
        df, "trait", trait_mask=trait_mask
    )
    assert res["alifestd_mark_monomorphic_clades_polars"].to_list() == [
        False,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_alifestd_mark_monomorphic_clades_polars_mark_as():
    df = _make_toy_df()
    res = alifestd_mark_monomorphic_clades_polars(df, "trait", mark_as="mono")
    assert res.columns == [*df.columns, "mono"]

    # existing column is overwritten
    res = alifestd_mark_monomorphic_clades_polars(res, "trait", mark_as="mono")
    assert res.columns == [*df.columns, "mono"]

    # mark_as may match mask function's output column
    mask_col = "alifestd_mask_monomorphic_clades_polars"
    res = alifestd_mark_monomorphic_clades_polars(
        df, "trait", mark_as=mask_col
    )
    assert res.columns == [*df.columns, mask_col]
    assert res[mask_col].equals(
        alifestd_mark_monomorphic_clades_polars(df, "trait")[
            "alifestd_mark_monomorphic_clades_polars"
        ],
        check_names=False,
    )


def test_alifestd_mark_monomorphic_clades_polars_empty():
    df = _make_toy_df().head(0)
    res = alifestd_mark_monomorphic_clades_polars(
        df, "trait", trait_mask="is_leaf"
    )
    assert res.is_empty()
    assert res.columns == [
        *df.columns,
        "alifestd_mark_monomorphic_clades_polars",
    ]


@pytest.mark.parametrize("col", ["missing", None])
def test_alifestd_mark_monomorphic_clades_polars_missing_column(col):
    df = _make_toy_df()
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        alifestd_mark_monomorphic_clades_polars(
            df,
            "trait" if col is None else col,
            trait_mask="missing" if col is None else None,
        )


def test_alifestd_mark_monomorphic_clades_polars_noncontiguous():
    df = _make_toy_df().with_columns(
        id=pl.col("id") * 10 + 3,
        ancestor_id=pl.col("ancestor_id") * 10 + 3,
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_monomorphic_clades_polars(df, "trait")


def test_alifestd_mark_monomorphic_clades_polars_unsorted():
    df = pl.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 2, 0], "trait": ["a"] * 3},
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_monomorphic_clades_polars(df, "trait")


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
@pytest.mark.parametrize(
    "trait_by", [["trait_a"], ["trait_b"], ["trait_a", "trait_b"]]
)
@pytest.mark.parametrize("trait_mask", [None, "is_leaf"])
def test_alifestd_mark_monomorphic_clades_polars_matches_pandas(
    phylogeny_df: pd.DataFrame,
    trait_by: list,
    trait_mask: typing.Optional[str],
):
    phylogeny_df = _add_inherited_traits(
        alifestd_to_working_format(phylogeny_df)
    ).drop(columns=["ancestor_list", "phenotype"], errors="ignore")

    expected = alifestd_mask_monomorphic_clades_asexual(
        phylogeny_df,
        trait_mask=(
            np.ones(len(phylogeny_df), dtype=bool)
            if trait_mask is None
            else phylogeny_df[trait_mask].to_numpy()
        ),
        trait_values=phylogeny_df.groupby(trait_by, dropna=False)
        .ngroup()
        .to_numpy(),
    )["alifestd_mask_monomorphic_clades_asexual"]

    res = alifestd_mark_monomorphic_clades_polars(
        pl.from_pandas(phylogeny_df), trait_by, trait_mask=trait_mask
    )
    result = res["alifestd_mark_monomorphic_clades_polars"]
    assert result.to_list() == expected.tolist()
    assert result.any()


@pytest.mark.parametrize("dtype", [pl.Int64, pl.Float64, pl.String])
def test_alifestd_mark_monomorphic_clades_polars_single_column_nulls(dtype):
    df = pl.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 0, 0], "t": [None] * 3},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64, "t": dtype},
    )
    res = alifestd_mark_monomorphic_clades_polars(df, "t")
    assert res["alifestd_mark_monomorphic_clades_polars"].all()

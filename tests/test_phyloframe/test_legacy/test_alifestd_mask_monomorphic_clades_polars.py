import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_aggregate_phylogenies,
    alifestd_mask_monomorphic_clades_asexual,
)
from phyloframe.legacy import (
    alifestd_mask_monomorphic_clades_polars as alifestd_mask_monomorphic_clades_polars_,
)
from phyloframe.legacy import (
    alifestd_to_working_format,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mask_monomorphic_clades_polars = enforce_dtype_stability_polars(
    alifestd_mask_monomorphic_clades_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


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
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
@pytest.mark.parametrize("dtype", [int, float, str, tuple])
@pytest.mark.parametrize("p_defined", [0.05, 0.5, 1.0])
def test_alifestd_mask_monomorphic_clades_polars_matches_pandas(
    phylogeny_df: pd.DataFrame,
    apply: typing.Callable,
    dtype: type,
    p_defined: float,
):
    phylogeny_df = alifestd_to_working_format(phylogeny_df).drop(
        columns=["ancestor_list", "phenotype"], errors="ignore"
    )
    rng = np.random.default_rng(1)
    trait_mask = rng.random(len(phylogeny_df)) < p_defined
    trait_values = np.array(rng.integers(0, 2, len(phylogeny_df)), dtype=dtype)
    original_trait_mask = trait_mask.copy()
    original_trait_values = trait_values.copy()

    expected = alifestd_mask_monomorphic_clades_asexual(
        phylogeny_df, trait_mask=trait_mask, trait_values=trait_values
    )["alifestd_mask_monomorphic_clades_asexual"]
    result_df = (
        alifestd_mask_monomorphic_clades_polars(
            apply(pl.from_pandas(phylogeny_df)),
            trait_mask=trait_mask,
            trait_values=trait_values,
        )
        .lazy()
        .collect()
    )

    assert result_df.drop("alifestd_mask_monomorphic_clades_polars").equals(
        pl.from_pandas(phylogeny_df)
    )
    result = result_df["alifestd_mask_monomorphic_clades_polars"]
    assert result.to_list() == expected.tolist()
    if p_defined >= 0.5:  # sparse traits may leave all clades monomorphic
        assert 0 < result.sum() < len(result)
    assert np.array_equal(trait_mask, original_trait_mask)
    assert np.array_equal(trait_values, original_trait_values)


@pytest.mark.parametrize("dtype", [int, float, str, tuple])
@pytest.mark.parametrize(
    "ancestor_ids, trait_mask",
    [
        # clade without defined traits precedes defined sibling
        ([0, 0, 0, 1], [False, False, True, False]),
        # clade without defined traits follows defined sibling
        ([0, 0, 0, 2], [False, True, False, False]),
    ],
)
def test_alifestd_mask_monomorphic_clades_polars_undefined_clade(
    dtype: type, ancestor_ids: list, trait_mask: list
):
    phylogeny_df = pl.DataFrame(
        {"id": [0, 1, 2, 3], "ancestor_id": ancestor_ids},
    )
    result_df = alifestd_mask_monomorphic_clades_polars(
        phylogeny_df,
        trait_mask=np.array(trait_mask, dtype=bool),
        trait_values=np.array([7, 8, 9, 6], dtype=dtype),
    )
    assert result_df["alifestd_mask_monomorphic_clades_polars"].all()


def test_alifestd_mask_monomorphic_clades_polars_polymorphic():
    phylogeny_df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[none]", "[0]", "[0]", "[1]", "[1]"],
        },
    )
    result_df = alifestd_mask_monomorphic_clades_polars(
        phylogeny_df,
        trait_mask=np.array([False, False, True, True, True]),
        trait_values=np.array(["a", "z", "a", "a", "b"]),
    )
    assert result_df["alifestd_mask_monomorphic_clades_polars"].to_list() == [
        False,
        False,
        True,
        True,
        True,
    ]


def test_alifestd_mask_monomorphic_clades_polars_empty():
    phylogeny_df = pl.DataFrame(
        {"id": [], "ancestor_id": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64},
    )
    result_df = alifestd_mask_monomorphic_clades_polars(
        phylogeny_df,
        trait_mask=np.array([], dtype=bool),
        trait_values=np.array([], dtype=int),
    )
    assert "alifestd_mask_monomorphic_clades_polars" in result_df.columns
    assert result_df.is_empty()


def test_alifestd_mask_monomorphic_clades_polars_noncontiguous():
    phylogeny_df = pl.DataFrame({"id": [0, 2], "ancestor_id": [0, 0]})
    with pytest.raises(NotImplementedError):
        alifestd_mask_monomorphic_clades_polars(
            phylogeny_df,
            trait_mask=np.array([True, True]),
            trait_values=np.array([0, 0]),
        )


def test_alifestd_mask_monomorphic_clades_polars_unsorted():
    phylogeny_df = pl.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 2, 0]})
    with pytest.raises(NotImplementedError):
        alifestd_mask_monomorphic_clades_polars(
            phylogeny_df,
            trait_mask=np.array([True, True, True]),
            trait_values=np.array([0, 0, 0]),
        )


def test_alifestd_mask_monomorphic_clades_polars_mark_as():
    phylogeny_df = pl.DataFrame(
        {"id": [0, 1, 2, 3], "ancestor_id": [0, 0, 0, 1]},
    )
    result_df = alifestd_mask_monomorphic_clades_polars(
        phylogeny_df,
        mark_as="mask",
        trait_mask=np.array([True, True, True, True]),
        trait_values=np.array([0, 0, 1, 0]),
    )
    assert result_df.columns == ["id", "ancestor_id", "mask"]
    assert result_df["mask"].to_list() == [False, True, True, True]

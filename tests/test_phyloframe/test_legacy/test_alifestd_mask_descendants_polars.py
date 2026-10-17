import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_aggregate_phylogenies,
    alifestd_mask_descendants_asexual,
)
from phyloframe.legacy import (
    alifestd_mask_descendants_polars as alifestd_mask_descendants_polars_,
)
from phyloframe.legacy import (
    alifestd_to_working_format,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mask_descendants_polars = enforce_dtype_stability_polars(
    alifestd_mask_descendants_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mask_descendants_polars_simple(apply: typing.Callable):
    phylogeny_df = pl.DataFrame(
        {"id": [0, 1, 2, 3], "ancestor_id": [0, 0, 0, 1]},
    )
    result_df = (
        alifestd_mask_descendants_polars(
            apply(phylogeny_df),
            ancestor_mask=np.array([False, True, False, False]),
        )
        .lazy()
        .collect()
    )
    assert result_df.columns == [
        "id",
        "ancestor_id",
        "alifestd_mask_descendants_polars",
    ]
    assert result_df["alifestd_mask_descendants_polars"].to_list() == [
        False,
        True,
        False,
        True,
    ]


def test_alifestd_mask_descendants_polars_mark_as():
    phylogeny_df = pl.DataFrame(
        {"id": [0, 1, 2, 3], "ancestor_id": [0, 0, 0, 1]},
    )
    result_df = alifestd_mask_descendants_polars(
        phylogeny_df,
        ancestor_mask=np.array([False, True, False, False]),
        mark_as="mask",
    )
    assert result_df.columns == ["id", "ancestor_id", "mask"]
    assert result_df["mask"].to_list() == [False, True, False, True]


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        alifestd_aggregate_phylogenies(
            [
                pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
                pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
            ]
        ),
    ],
)
@pytest.mark.parametrize("p_ancestor", [0.0, 0.01, 0.2])
def test_alifestd_mask_descendants_polars_matches_pandas(
    phylogeny_df: pd.DataFrame, p_ancestor: float
):
    phylogeny_df = alifestd_to_working_format(phylogeny_df)[
        ["id", "ancestor_id"]
    ]
    ancestor_mask = (
        np.random.default_rng(1).random(len(phylogeny_df)) < p_ancestor
    )
    expected = alifestd_mask_descendants_asexual(
        phylogeny_df, ancestor_mask=ancestor_mask
    )["alifestd_mask_descendants_asexual"]
    result_df = alifestd_mask_descendants_polars(
        pl.from_pandas(phylogeny_df), ancestor_mask=ancestor_mask
    )
    assert (
        result_df["alifestd_mask_descendants_polars"].to_list()
        == expected.tolist()
    )

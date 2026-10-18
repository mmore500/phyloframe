import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_screen_trait_defined_clades_fitch_asexual,
    alifestd_screen_trait_defined_clades_fitch_polars,
    alifestd_to_working_format,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_empty(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Int64),
                "ancestor_id": pl.Series([], dtype=pl.Int64),
                "absent": pl.Series([], dtype=pl.Boolean),
                "present": pl.Series([], dtype=pl.Boolean),
            }
        ),
    )
    res = (
        alifestd_screen_trait_defined_clades_fitch_polars(
            df_pl,
            mask_trait_absent="absent",
            mask_trait_present="present",
        )
        .lazy()
        .collect()
    )
    assert res.is_empty()
    assert "alifestd_screen_trait_defined_clades_fitch_polars" in res.columns


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple_tree(apply: typing.Callable):
    # Tree structure:
    #         0
    #       /   \
    #      1-    2
    #     / \   / \
    #    3   4 5   6
    #    -   + +   +
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6],
                "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
                "absent": [False, True, False, True, False, False, False],
                "present": [False, False, False, False, True, True, True],
            }
        ),
    )
    res = (
        alifestd_screen_trait_defined_clades_fitch_polars(
            df_pl,
            mask_trait_absent="absent",
            mask_trait_present="present",
        )
        .lazy()
        .collect()
    )
    assert res[
        "alifestd_screen_trait_defined_clades_fitch_polars"
    ].to_list() == [False, False, True, False, True, False, False]


def test_expr_and_mark_as():
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            "trait": [None, -1, None, -1, 1, 1, 1],
        }
    )
    res = alifestd_screen_trait_defined_clades_fitch_polars(
        df_pl,
        mask_trait_absent=pl.col("trait") == -1,
        mask_trait_present=pl.col("trait") == 1,
        mark_as="is_mdc",
    )
    assert res["is_mdc"].to_list() == [
        False,
        False,
        True,
        False,
        True,
        False,
        False,
    ]


def test_conflicting_masks():
    df_pl = pl.DataFrame(
        {
            "id": [0, 1],
            "ancestor_id": [0, 0],
            "absent": [False, True],
            "present": [False, True],
        }
    )
    with pytest.raises(ValueError):
        alifestd_screen_trait_defined_clades_fitch_polars(
            df_pl,
            mask_trait_absent="absent",
            mask_trait_present="present",
        )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_non_contiguous_ids(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
                "absent": [False, True, False],
                "present": [False, False, True],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_screen_trait_defined_clades_fitch_polars(
            df_pl,
            mask_trait_absent="absent",
            mask_trait_present="present",
        ).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_unsorted(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
                "absent": [False, True, False],
                "present": [False, False, True],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_screen_trait_defined_clades_fitch_polars(
            df_pl,
            mask_trait_absent="absent",
            mask_trait_present="present",
        ).lazy().collect()


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize("seed", range(4))
def test_matches_pandas(phylogeny_df: pd.DataFrame, seed: int):
    phylogeny_df = alifestd_to_working_format(phylogeny_df)
    rng = np.random.default_rng(seed)
    trait_set = rng.integers(0, 3, size=len(phylogeny_df))
    phylogeny_df["absent"] = trait_set == 1
    phylogeny_df["present"] = trait_set == 2

    res_pd = alifestd_screen_trait_defined_clades_fitch_asexual(
        phylogeny_df,
        mask_trait_absent=phylogeny_df["absent"],
        mask_trait_present=phylogeny_df["present"],
    )
    res_pl = alifestd_screen_trait_defined_clades_fitch_polars(
        pl.from_pandas(
            phylogeny_df[["id", "ancestor_id", "absent", "present"]]
        ),
        mask_trait_absent="absent",
        mask_trait_present="present",
    )
    np.testing.assert_array_equal(
        res_pd,
        res_pl["alifestd_screen_trait_defined_clades_fitch_polars"].to_numpy(),
    )

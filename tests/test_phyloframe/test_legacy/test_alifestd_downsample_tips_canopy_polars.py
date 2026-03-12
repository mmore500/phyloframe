import os

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_aggregate_phylogenies,
    alifestd_find_leaf_ids,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_downsample_tips_canopy_asexual import (
    alifestd_downsample_tips_canopy_asexual,
)
from phyloframe.legacy._alifestd_downsample_tips_canopy_polars import (
    alifestd_downsample_tips_canopy_polars as alifestd_downsample_tips_canopy_polars_,
)

from ._impl import assert_dtype_consistency

alifestd_downsample_tips_canopy_polars = assert_dtype_consistency(alifestd_downsample_tips_canopy_polars_)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _count_leaf_nodes_polars(phylogeny_df: pl.DataFrame) -> int:
    """Count leaf nodes in a polars dataframe (works with any ids)."""
    all_ids = set(phylogeny_df["id"].to_list())
    internal_ids = set(
        phylogeny_df.filter(pl.col("ancestor_id") != pl.col("id"))
        .select("ancestor_id")
        .to_series()
        .to_list()
    )
    return len(all_ids - internal_ids)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            alifestd_aggregate_phylogenies(
                [
                    pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
                    pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
                ]
            )
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
        ),
    ],
)
@pytest.mark.parametrize(
    "n_downsample",
    [1, pytest.param(5, marks=pytest.mark.heavy), 10, 100000000],
)
def test_alifestd_downsample_tips_canopy_polars(
    phylogeny_df: pd.DataFrame,
    n_downsample: int,
):
    phylogeny_df_pl = pl.from_pandas(phylogeny_df)

    original_len = len(phylogeny_df_pl)
    original_n_downsample = _count_leaf_nodes_polars(phylogeny_df_pl)

    result_df = alifestd_downsample_tips_canopy_polars(
        phylogeny_df_pl,
        n_downsample,
        criterion="id",
    )

    assert len(result_df) <= original_len
    assert "extant" not in result_df.columns
    assert set(result_df["id"].to_list()).issubset(
        set(phylogeny_df_pl["id"].to_list())
    )
    assert _count_leaf_nodes_polars(result_df) == min(
        original_n_downsample, n_downsample
    )


@pytest.mark.parametrize("n_downsample", [0, 1])
def test_alifestd_downsample_tips_canopy_polars_empty(n_downsample: int):
    phylogeny_df = pl.DataFrame(
        {"id": [], "ancestor_id": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64},
    )

    result_df = alifestd_downsample_tips_canopy_polars(
        phylogeny_df, n_downsample, criterion="id"
    )

    assert result_df.is_empty()


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            alifestd_aggregate_phylogenies(
                [
                    pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
                    pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
                ]
            )
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
        ),
    ],
)
@pytest.mark.parametrize(
    "n_downsample", [1, pytest.param(5, marks=pytest.mark.heavy), 10]
)
def test_alifestd_downsample_tips_canopy_polars_matches_pandas(
    phylogeny_df: pd.DataFrame,
    n_downsample: int,
):
    """Verify polars result matches pandas result for same prepared input."""
    phylogeny_df_pl = pl.from_pandas(phylogeny_df)

    result_pd = alifestd_downsample_tips_canopy_asexual(
        phylogeny_df, n_downsample, mutate=False, criterion="id"
    )
    result_pl = alifestd_downsample_tips_canopy_polars(
        phylogeny_df_pl, n_downsample, criterion="id"
    )

    assert set(result_pd["id"]) == set(result_pl["id"].to_list())
    assert len(result_pd) == len(result_pl)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
        ),
    ],
)
def test_alifestd_downsample_tips_canopy_polars_retains_highest_ids(
    phylogeny_df: pd.DataFrame,
):
    """Verify that the retained tips are the ones with the highest ids."""
    n_downsample = 5
    phylogeny_df_pl = pl.from_pandas(phylogeny_df)
    result_df = alifestd_downsample_tips_canopy_polars(
        phylogeny_df_pl, n_downsample, criterion="id"
    )

    original_tips = alifestd_find_leaf_ids(phylogeny_df)
    expected_kept = set(sorted(original_tips)[-n_downsample:])

    result_tips_all = set(result_df["id"].to_list())
    # find leaf ids in result: ids that don't appear as ancestor_id of others
    internal_ids = set(
        result_df.filter(pl.col("ancestor_id") != pl.col("id"))
        .select("ancestor_id")
        .to_series()
        .to_list()
    )
    result_tips = result_tips_all - internal_ids
    assert result_tips == expected_kept


def test_alifestd_downsample_tips_canopy_polars_no_ancestor_id():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[none]", "[0]", "[1]"],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_downsample_tips_canopy_polars(df, 1, criterion="id")


def test_alifestd_downsample_tips_canopy_polars_simple():
    """Test a simple hand-crafted tree.

    Tree structure:
        0 (root)
        +-- 1
        |   +-- 3 (leaf)
        |   +-- 4 (leaf)
        +-- 2 (leaf)

    With n_downsample=2, keep leaves 3 and 4 (highest ids), result is 0, 1, 3, 4.
    With n_downsample=1, keep leaf 4 (highest id), result is 0, 1, 4.
    """
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
            "destruction_time": [float("inf")] * 5,
        }
    )

    result2 = alifestd_downsample_tips_canopy_polars(df, 2, criterion="id")
    assert set(result2["id"].to_list()) == {0, 1, 3, 4}

    result1 = alifestd_downsample_tips_canopy_polars(df, 1, criterion="id")
    assert set(result1["id"].to_list()) == {0, 1, 4}


def test_alifestd_downsample_tips_canopy_polars_all_tips():
    """Requesting more tips than exist should return the full phylogeny."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
            "destruction_time": [float("inf")] * 5,
        }
    )

    result = alifestd_downsample_tips_canopy_polars(df, 100000, criterion="id")

    assert len(result) == 5


def test_alifestd_downsample_tips_canopy_polars_tied_criterion():
    """When all leaves share the same criterion value, exactly n_downsample
    should still be retained (ties broken arbitrarily)."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
            "destruction_time": [float("inf")] * 5,
            "time": [0, 0, 0, 0, 0],
        }
    )
    # leaves are 2, 3, 4 — all have time=0
    for n_downsample in (1, 2, 3):
        result = alifestd_downsample_tips_canopy_polars(
            df, n_downsample, criterion="time"
        )
        assert _count_leaf_nodes_polars(result) == n_downsample


def test_alifestd_downsample_tips_canopy_polars_n_none():
    """When n_downsample is None, keep only leaves with the max criterion value."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
            "destruction_time": [float("inf")] * 5,
            "origin_time": [0, 1, 2, 3, 3],
        }
    )
    # leaves are 2 (origin_time=2), 3 (origin_time=3), 4 (origin_time=3)
    # max origin_time among leaves is 3, shared by leaves 3 and 4
    result = alifestd_downsample_tips_canopy_polars(
        df, criterion="origin_time"
    )
    result_ids = set(result["id"].to_list())
    # should keep only leaves 3 and 4 (plus ancestors 0 and 1)
    assert 3 in result_ids
    assert 4 in result_ids
    assert 2 not in result_ids
    assert _count_leaf_nodes_polars(result) == 2


def test_alifestd_downsample_tips_canopy_polars_n_none_single_max():
    """When n_downsample is None and only one leaf has the max, keep just that one."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
            "destruction_time": [float("inf")] * 5,
            "origin_time": [0, 1, 2, 3, 4],
        }
    )
    # leaves are 2 (origin_time=2), 3 (origin_time=3), 4 (origin_time=4)
    # max origin_time among leaves is 4, only leaf 4
    result = alifestd_downsample_tips_canopy_polars(
        df, criterion="origin_time"
    )
    result_ids = set(result["id"].to_list())
    assert 4 in result_ids
    assert 2 not in result_ids
    assert 3 not in result_ids
    assert _count_leaf_nodes_polars(result) == 1


def test_alifestd_downsample_tips_canopy_polars_missing_criterion():
    """Verify ValueError when criterion column is missing."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="criterion column"):
        alifestd_downsample_tips_canopy_polars(df, 1, criterion="nonexistent")

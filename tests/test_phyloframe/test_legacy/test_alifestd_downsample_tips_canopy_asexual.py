import os
import warnings

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_aggregate_phylogenies,
    alifestd_count_leaf_nodes,
    alifestd_find_leaf_ids,
    alifestd_prune_extinct_lineages_asexual,
    alifestd_try_add_ancestor_id_col,
    alifestd_validate,
)
from phyloframe.legacy._alifestd_downsample_tips_canopy_asexual import (
    alifestd_downsample_tips_canopy_asexual,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


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
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize("n_downsample", [1, 5, 10, 100000000])
@pytest.mark.parametrize("mutate", [True, False])
def test_alifestd_downsample_tips_canopy_asexual(
    phylogeny_df, n_downsample, mutate
):
    original_df = phylogeny_df.copy()

    result_df = alifestd_downsample_tips_canopy_asexual(
        phylogeny_df, n_downsample, mutate, criterion="id"
    )

    assert len(result_df) <= len(original_df)
    assert "extant" not in result_df.columns

    if not mutate:
        pd.testing.assert_frame_equal(phylogeny_df, original_df)

    assert all(result_df["id"].isin(original_df["id"]))
    assert alifestd_count_leaf_nodes(result_df) == min(
        alifestd_count_leaf_nodes(original_df), n_downsample
    )


@pytest.mark.parametrize("n_downsample", [0, 1])
def test_alifestd_downsample_tips_canopy_asexual_with_zero_tips(n_downsample):
    phylogeny_df = pd.DataFrame({"id": [], "parent_id": [], "ancestor_id": []})

    result_df = alifestd_downsample_tips_canopy_asexual(
        phylogeny_df, n_downsample, criterion="id"
    )

    assert result_df.empty


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
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize("n_downsample", [1, 5, 10])
def test_downsample_canopy_vs_manual(phylogeny_df, n_downsample):
    """Verify canopy prune matches manually marking top tips as extant and
    then pruning extinct lineages."""
    original_df = phylogeny_df.copy()
    canopy_df = alifestd_downsample_tips_canopy_asexual(
        phylogeny_df, n_downsample, mutate=False, criterion="id"
    )

    # manually replicate: find top tips, mark extant, prune
    tips = alifestd_find_leaf_ids(original_df)
    tips_sorted = np.sort(tips)
    kept = tips_sorted[-n_downsample:]

    original_df = alifestd_try_add_ancestor_id_col(original_df)
    original_df["extant"] = original_df["id"].isin(kept)
    manual_df = alifestd_prune_extinct_lineages_asexual(
        original_df, mutate=True
    ).drop(columns=["extant"])

    pd.testing.assert_frame_equal(canopy_df, manual_df)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
def test_alifestd_downsample_tips_canopy_asexual_retains_highest_ids(
    phylogeny_df,
):
    """Verify that the retained tips are the ones with the highest ids."""
    n_downsample = 5
    result_df = alifestd_downsample_tips_canopy_asexual(
        phylogeny_df, n_downsample, criterion="id"
    )

    original_tips = alifestd_find_leaf_ids(phylogeny_df)
    expected_kept = set(sorted(original_tips)[-n_downsample:])

    result_tips = set(alifestd_find_leaf_ids(result_df))
    assert result_tips == expected_kept


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
def test_alifestd_downsample_tips_canopy_asexual_validates(phylogeny_df):
    n_downsample = 5
    result_df = alifestd_downsample_tips_canopy_asexual(
        phylogeny_df, n_downsample, criterion="id"
    )
    assert alifestd_validate(result_df)


def test_alifestd_downsample_tips_canopy_asexual_simple():
    """Test a simple hand-crafted tree.

    Tree structure:
        0 (root)
        +-- 1
        |   +-- 3 (leaf)
        |   +-- 4 (leaf)
        +-- 2 (leaf)

    With n_downsample=2, keep leaves 3 and 4 (highest ids), result is
    0, 1, 3, 4.
    With n_downsample=1, keep leaf 4 (highest id), result is 0, 1, 4.
    """
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[none]", "[0]", "[0]", "[1]", "[1]"],
        }
    )

    result2 = alifestd_downsample_tips_canopy_asexual(df, 2, criterion="id")
    assert set(result2["id"]) == {0, 1, 3, 4}

    result1 = alifestd_downsample_tips_canopy_asexual(df, 1, criterion="id")
    assert set(result1["id"]) == {0, 1, 4}


def test_alifestd_downsample_tips_canopy_asexual_all_tips():
    """Requesting more tips than exist should return the full phylogeny."""
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[none]", "[0]", "[0]", "[1]", "[1]"],
        }
    )

    result = alifestd_downsample_tips_canopy_asexual(
        df, 100000, criterion="id"
    )
    assert len(result) == 5


def test_alifestd_downsample_tips_canopy_asexual_tied_criterion():
    """When all leaves share the same criterion value, exactly n_downsample
    should still be retained (ties broken arbitrarily)."""
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[none]", "[0]", "[0]", "[1]", "[1]"],
            "time": [0, 0, 0, 0, 0],
        }
    )
    # leaves are 2, 3, 4 — all have time=0
    for n_downsample in (1, 2, 3):
        result = alifestd_downsample_tips_canopy_asexual(
            df, n_downsample, criterion="time"
        )
        assert alifestd_count_leaf_nodes(result) == n_downsample


def test_alifestd_downsample_tips_canopy_asexual_n_none():
    """When n_downsample is None, keep only leaves with the max criterion value."""
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[none]", "[0]", "[0]", "[1]", "[1]"],
            "origin_time": [0, 1, 2, 3, 3],
        }
    )
    # leaves are 2 (origin_time=2), 3 (origin_time=3), 4 (origin_time=3)
    # max origin_time among leaves is 3, shared by leaves 3 and 4
    result = alifestd_downsample_tips_canopy_asexual(
        df, criterion="origin_time"
    )
    result_ids = set(result["id"])
    assert 3 in result_ids
    assert 4 in result_ids
    assert 2 not in result_ids
    assert alifestd_count_leaf_nodes(result) == 2


def test_alifestd_downsample_tips_canopy_asexual_n_none_single_max():
    """When n_downsample is None and only one leaf has the max, keep just that one."""
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[none]", "[0]", "[0]", "[1]", "[1]"],
            "origin_time": [0, 1, 2, 3, 4],
        }
    )
    # leaves are 2 (origin_time=2), 3 (origin_time=3), 4 (origin_time=4)
    # max origin_time among leaves is 4, only leaf 4
    result = alifestd_downsample_tips_canopy_asexual(
        df, criterion="origin_time"
    )
    result_ids = set(result["id"])
    assert 4 in result_ids
    assert 2 not in result_ids
    assert 3 not in result_ids
    assert alifestd_count_leaf_nodes(result) == 1


def test_alifestd_downsample_tips_canopy_asexual_missing_criterion():
    """Verify ValueError when criterion column is missing."""
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[none]", "[0]", "[0]"],
        }
    )

    with pytest.raises(ValueError, match="criterion column"):
        alifestd_downsample_tips_canopy_asexual(df, 1, criterion="nonexistent")


def test_alifestd_downsample_tips_canopy_asexual_num_tips_deprecated():
    """Verify that passing num_tips triggers a DeprecationWarning."""
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[none]", "[0]", "[0]"],
        }
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        alifestd_downsample_tips_canopy_asexual(df, num_tips=1, criterion="id")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "num_tips" in str(w[0].message)

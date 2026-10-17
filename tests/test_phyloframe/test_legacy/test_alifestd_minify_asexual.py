import os

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
    alifestd_minify_asexual,
    alifestd_test_leaves_isomorphic_asexual,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_list_col,
    alifestd_unminify_asexual,
    alifestd_validate,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
        ),
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny-tworoots.csv"
        ),
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
        pd.read_csv(f"{assets_path}/trunktestphylo.csv"),
    ],
)
def test_fuzz(phylogeny_df: pd.DataFrame):
    original = phylogeny_df.copy()

    result = alifestd_minify_asexual(phylogeny_df)

    assert original.equals(phylogeny_df)
    assert len(result) == len(phylogeny_df)
    assert "id" not in result.columns
    assert "ancestor_id" not in result.columns
    assert "ancestor_list" not in result.columns
    assert "ancestor_offset" in result.columns

    offsets = result["ancestor_offset"].to_numpy()
    positions = np.arange(len(result))
    assert (offsets >= 0).all()
    assert (offsets <= positions).all()

    # preorder: each subtree occupies contiguous rows directly after its root
    ancestor_positions = positions - offsets
    subtree_end = positions + 1
    for pos in reversed(range(len(result))):
        anc = ancestor_positions[pos]
        if anc != pos:
            subtree_end[anc] = max(subtree_end[anc], subtree_end[pos])
    for pos in range(len(result)):
        anc = ancestor_positions[pos]
        if anc != pos:
            assert anc < pos < subtree_end[anc]
            assert subtree_end[pos] <= subtree_end[anc]

    # round trip preserves topology
    labeled = alifestd_to_working_format(phylogeny_df)
    labeled["original_id"] = labeled["id"]
    restored = alifestd_unminify_asexual(alifestd_minify_asexual(labeled))
    assert alifestd_validate(alifestd_try_add_ancestor_list_col(restored))
    assert alifestd_test_leaves_isomorphic_asexual(
        labeled, restored, taxon_label="original_id"
    )
    # stricter check, including inner nodes; ids are contiguous, so index
    # positionally
    original_ids = restored["original_id"].to_numpy()
    np.testing.assert_array_equal(
        original_ids[restored["ancestor_id"]],
        labeled["ancestor_id"].to_numpy()[original_ids],
    )


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate: bool):
    """Tree structure:
        0 (root)
        +-- 1
        |   +-- 3
        +-- 2
            +-- 4

    Preorder: 0, 1, 3, 2, 4.
    """
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 2],
            "name": ["a", "b", "c", "d", "e"],
        }
    )
    original = phylogeny_df.copy()

    result = alifestd_minify_asexual(phylogeny_df, mutate=mutate)

    if not mutate:
        assert original.equals(phylogeny_df)
    assert result.columns.tolist() == ["name", "ancestor_offset"]
    assert result["name"].tolist() == ["a", "b", "d", "c", "e"]
    assert result["ancestor_offset"].tolist() == [0, 1, 1, 3, 1]


def test_non_contiguous_unsorted():
    phylogeny_df = pd.DataFrame(
        {
            "id": [10, 5, 7, 3],
            "ancestor_list": ["[5]", "[none]", "[5]", "[10]"],
            "name": ["c", "a", "b", "d"],
        }
    )

    result = alifestd_minify_asexual(phylogeny_df)

    assert result.columns.tolist() == ["name", "ancestor_offset"]
    assert result["name"].tolist() == ["a", "c", "d", "b"]
    assert result["ancestor_offset"].tolist() == [0, 1, 1, 3]


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64])
def test_dtype_no_shrink(dtype: type):
    phylogeny_df = pd.DataFrame(
        {
            "id": np.array([0, 1, 2], dtype=dtype),
            "ancestor_id": np.array([0, 0, 1], dtype=dtype),
        }
    )

    result = alifestd_minify_asexual(phylogeny_df, shrink_dtype=False)

    assert result["ancestor_offset"].dtype == dtype
    assert result["ancestor_offset"].tolist() == [0, 1, 1]


@pytest.mark.parametrize(
    "num_rows, expected_dtype",
    [(1, np.int8), (3, np.int8), (200, np.int16), (40000, np.int32)],
)
def test_dtype_shrink(num_rows: int, expected_dtype: type):
    # star topology: maximum offset is num_rows - 1
    phylogeny_df = pd.DataFrame(
        {
            "id": np.arange(num_rows),
            "ancestor_id": np.zeros(num_rows, dtype=int),
        }
    )

    result = alifestd_minify_asexual(phylogeny_df)

    assert result["ancestor_offset"].dtype == expected_dtype
    assert result["ancestor_offset"].tolist() == list(range(num_rows))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint32, np.uint64])
def test_dtype_shrink_unsigned(dtype: type):
    phylogeny_df = pd.DataFrame(
        {
            "id": np.array([0, 1, 2], dtype=dtype),
            "ancestor_id": np.array([0, 0, 1], dtype=dtype),
        }
    )

    result = alifestd_minify_asexual(phylogeny_df)

    assert result["ancestor_offset"].dtype == np.int8
    assert result["ancestor_offset"].tolist() == [0, 1, 1]


def test_warns_existing_ancestor_offset():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "ancestor_offset": [9, 9, 9],
        }
    )

    with pytest.warns(UserWarning, match="ancestor_offset"):
        result = alifestd_minify_asexual(phylogeny_df)

    assert result["ancestor_offset"].tolist() == [0, 1, 1]


def test_empty():
    result = alifestd_minify_asexual(alifestd_make_empty())
    assert len(result) == 0
    assert "ancestor_offset" in result.columns
    assert "id" not in result.columns

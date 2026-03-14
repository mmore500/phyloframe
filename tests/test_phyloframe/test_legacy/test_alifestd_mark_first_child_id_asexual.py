import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
)
from phyloframe.legacy import (
    alifestd_mark_first_child_id_asexual as alifestd_mark_first_child_id_asexual_,
)
from phyloframe.legacy import (
    alifestd_validate,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_mark_first_child_id_asexual = enforce_dtype_stability_pandas(
    alifestd_mark_first_child_id_asexual_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
        ),
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
def test_fuzz(phylogeny_df: pd.DataFrame):
    original = phylogeny_df.copy()

    result = alifestd_mark_first_child_id_asexual(phylogeny_df)

    assert alifestd_validate(result)
    assert original.equals(phylogeny_df)

    valid_ids = set(result["id"])
    assert all(fid in valid_ids for fid in result["first_child_id"])


def test_empty():
    res = alifestd_mark_first_child_id_asexual(alifestd_make_empty())
    assert "first_child_id" in res
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_chain(mutate: bool):
    """Chain: 0 -> 1 -> 2."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
        }
    )
    original_df = phylogeny_df.copy()
    result_df = alifestd_mark_first_child_id_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert result_df.loc[0, "first_child_id"] == 1
    assert result_df.loc[1, "first_child_id"] == 2
    assert result_df.loc[2, "first_child_id"] == 2  # leaf: own id

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate: bool):
    """Tree:
    0 (root)
    +-- 1
    |   +-- 3
    |   +-- 4
    +-- 2
    """
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
        }
    )
    original_df = phylogeny_df.copy()
    result_df = alifestd_mark_first_child_id_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert result_df.loc[0, "first_child_id"] == 1
    assert result_df.loc[1, "first_child_id"] == 3
    assert result_df.loc[2, "first_child_id"] == 2  # leaf
    assert result_df.loc[3, "first_child_id"] == 3  # leaf
    assert result_df.loc[4, "first_child_id"] == 4  # leaf

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_single_root(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "id": [0],
            "ancestor_list": ["[None]"],
        }
    )
    result_df = alifestd_mark_first_child_id_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert result_df.loc[0, "first_child_id"] == 0  # leaf: own id


@pytest.mark.parametrize("mutate", [True, False])
def test_star_topology(mutate: bool):
    """Root with many direct children."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_list": [
                "[None]",
                "[0]",
                "[0]",
                "[0]",
                "[0]",
                "[0]",
            ],
        }
    )
    result_df = alifestd_mark_first_child_id_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert result_df.loc[0, "first_child_id"] == 1  # first child is 1
    for i in range(1, 6):
        assert result_df.loc[i, "first_child_id"] == i  # all leaves


@pytest.mark.parametrize("mutate", [True, False])
def test_depth_first_traversal(mutate: bool):
    """Test navigating the tree depth-first using first_child_id.

    Tree:
        0 (root)
        +-- 1
        |   +-- 3
        |   |   +-- 7
        |   |   +-- 8
        |   +-- 4
        +-- 2
        |   +-- 5
        |   +-- 6
    """
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "ancestor_list": [
                "[None]",
                "[0]",
                "[0]",
                "[1]",
                "[1]",
                "[2]",
                "[2]",
                "[3]",
                "[3]",
            ],
        }
    )
    result_df = alifestd_mark_first_child_id_asexual(
        phylogeny_df,
        mutate=mutate,
    )

    # navigate down the leftmost path: 0 -> 1 -> 3 -> 7 (leaf)
    cur = 0
    path = [cur]
    while result_df.loc[cur, "first_child_id"] != cur:
        cur = result_df.loc[cur, "first_child_id"]
        path.append(cur)

    assert path == [0, 1, 3, 7]

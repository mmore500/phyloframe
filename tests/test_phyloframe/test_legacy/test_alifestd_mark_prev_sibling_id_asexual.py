import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
)
from phyloframe.legacy import (
    alifestd_mark_prev_sibling_id_asexual as alifestd_mark_prev_sibling_id_asexual_,
)
from phyloframe.legacy import (
    alifestd_validate,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_mark_prev_sibling_id_asexual = enforce_dtype_stability_pandas(
    alifestd_mark_prev_sibling_id_asexual_
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

    result = alifestd_mark_prev_sibling_id_asexual(phylogeny_df)

    assert alifestd_validate(result)
    assert original.equals(phylogeny_df)

    valid_ids = set(result["id"])
    assert all(sid in valid_ids for sid in result["prev_sibling_id"])


def test_empty():
    res = alifestd_mark_prev_sibling_id_asexual(alifestd_make_empty())
    assert "prev_sibling_id" in res
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_chain(mutate: bool):
    """Chain: 0 -> 1 -> 2. No siblings."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
        }
    )
    original_df = phylogeny_df.copy()
    result_df = alifestd_mark_prev_sibling_id_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert result_df.loc[0, "prev_sibling_id"] == 0  # root, no sibling
    assert result_df.loc[1, "prev_sibling_id"] == 1  # only child
    assert result_df.loc[2, "prev_sibling_id"] == 2  # only child

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
    result_df = alifestd_mark_prev_sibling_id_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert result_df.loc[0, "prev_sibling_id"] == 0  # root
    assert result_df.loc[1, "prev_sibling_id"] == 1  # first sib, self
    assert result_df.loc[2, "prev_sibling_id"] == 1  # prev sib is 1
    assert result_df.loc[3, "prev_sibling_id"] == 3  # first sib, self
    assert result_df.loc[4, "prev_sibling_id"] == 3  # prev sib is 3

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
    result_df = alifestd_mark_prev_sibling_id_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert result_df.loc[0, "prev_sibling_id"] == 0


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
    result_df = alifestd_mark_prev_sibling_id_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert result_df.loc[0, "prev_sibling_id"] == 0  # root
    assert result_df.loc[1, "prev_sibling_id"] == 1  # first, self
    assert result_df.loc[2, "prev_sibling_id"] == 1
    assert result_df.loc[3, "prev_sibling_id"] == 2
    assert result_df.loc[4, "prev_sibling_id"] == 3
    assert result_df.loc[5, "prev_sibling_id"] == 4


@pytest.mark.parametrize("mutate", [True, False])
def test_reverse_sibling_traversal(mutate: bool):
    """Test navigating siblings in reverse using prev_sibling_id.

    Tree:
        0 (root)
        +-- 1
        |   +-- 4
        |   +-- 5
        +-- 2
        |   +-- 6
        +-- 3
    """
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_list": [
                "[None]",
                "[0]",
                "[0]",
                "[0]",
                "[1]",
                "[1]",
                "[2]",
            ],
        }
    )
    result_df = alifestd_mark_prev_sibling_id_asexual(
        phylogeny_df,
        mutate=mutate,
    )

    # navigate siblings of root's children backward: 3 -> 2 -> 1 (end)
    cur = 3
    siblings = [cur]
    while result_df.loc[cur, "prev_sibling_id"] != cur:
        cur = result_df.loc[cur, "prev_sibling_id"]
        siblings.append(cur)

    assert siblings == [3, 2, 1]

    # navigate siblings of node 1's children backward: 5 -> 4 (end)
    cur = 5
    siblings = [cur]
    while result_df.loc[cur, "prev_sibling_id"] != cur:
        cur = result_df.loc[cur, "prev_sibling_id"]
        siblings.append(cur)

    assert siblings == [5, 4]

import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_find_root_ids,
    alifestd_make_empty,
    alifestd_try_add_ancestor_id_col,
)
from phyloframe.legacy import (
    alifestd_unfurl_traversal_topological_asexual as alifestd_unfurl_traversal_topological_asexual_,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_unfurl_traversal_topological_asexual = enforce_dtype_stability_pandas(
    alifestd_unfurl_traversal_topological_asexual_
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

    result = alifestd_unfurl_traversal_topological_asexual(phylogeny_df)

    assert original.equals(phylogeny_df)

    assert len(result) == len(phylogeny_df)
    assert set(result) == set(phylogeny_df["id"])

    # Verify valid topological order: every parent appears before child
    ref_df = alifestd_try_add_ancestor_id_col(phylogeny_df.copy())
    ancestor_of = dict(zip(ref_df["id"], ref_df["ancestor_id"]))
    pos = {node: i for i, node in enumerate(result)}
    for node_id in result:
        ancestor_id = ancestor_of[node_id]
        if node_id != ancestor_id:
            assert pos[ancestor_id] < pos[node_id]

    # First elements should be roots
    assert result[0] in alifestd_find_root_ids(phylogeny_df)


def test_empty():
    res = alifestd_unfurl_traversal_topological_asexual(
        alifestd_make_empty(),
    )
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_chain(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
        },
    )
    original_df = phylogeny_df.copy()
    result = alifestd_unfurl_traversal_topological_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # Chain: 0 -> 1 -> 2, already sorted
    assert result.tolist() == [0, 1, 2]

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_scrambled(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "id": [3, 0, 1, 2],
            "ancestor_list": ["[0]", "[1]", "[None]", "[1]"],
        },
    )
    original_df = phylogeny_df.copy()
    result = alifestd_unfurl_traversal_topological_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # Tree: 1 is root, children 0 and 2; 0 has child 3
    # Valid topological: 1 before {0, 2}, 0 before 3
    pos = {node: i for i, node in enumerate(result)}
    assert pos[1] < pos[0]
    assert pos[1] < pos[2]
    assert pos[0] < pos[3]
    assert len(result) == 4
    assert set(result) == {0, 1, 2, 3}

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_two_roots(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "id": [4, 0, 2, 3],
            "ancestor_list": ["[None]", "[None]", "[0]", "[4]"],
        },
    )
    original_df = phylogeny_df.copy()
    result = alifestd_unfurl_traversal_topological_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # Two roots: 0 (child 2) and 4 (child 3)
    pos = {node: i for i, node in enumerate(result)}
    assert pos[0] < pos[2]
    assert pos[4] < pos[3]
    assert len(result) == 4
    assert set(result) == {0, 2, 3, 4}

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_branching(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]"],
        },
    )
    original_df = phylogeny_df.copy()
    result = alifestd_unfurl_traversal_topological_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # Tree: 0 -> {1, 2}, 1 -> {3}
    # Already topologically sorted, so return ids in order
    assert result.tolist() == [0, 1, 2, 3]

    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_already_sorted_returns_ids():
    """When already topologically sorted, return ids directly."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 0, 1],
        },
    )
    result = alifestd_unfurl_traversal_topological_asexual(phylogeny_df)
    assert result.tolist() == [0, 1, 2, 3]


def test_reverse_order():
    """Test with ids in reverse topological order."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [2, 1, 0],
            "ancestor_list": ["[1]", "[0]", "[None]"],
        },
    )
    result = alifestd_unfurl_traversal_topological_asexual(phylogeny_df)
    pos = {node: i for i, node in enumerate(result)}
    assert pos[0] < pos[1]
    assert pos[1] < pos[2]

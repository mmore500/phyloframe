import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
)
from phyloframe.legacy import (
    alifestd_mark_csr_children_asexual as alifestd_mark_csr_children_asexual_,
)
from phyloframe.legacy import (
    alifestd_mark_csr_offsets_asexual,
    alifestd_mark_num_children_asexual,
    alifestd_validate,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_mark_csr_children_asexual = enforce_dtype_stability_pandas(
    alifestd_mark_csr_children_asexual_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _get_children(result_df, node_id):
    """Helper: extract children of a node from CSR columns."""
    start = result_df.loc[node_id, "csr_offsets"]
    nc = result_df.loc[node_id, "num_children"]
    flat = result_df["csr_children"].tolist()
    return flat[start : start + nc]


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

    result = alifestd_mark_num_children_asexual(phylogeny_df)
    result = alifestd_mark_csr_offsets_asexual(result, mutate=True)
    result = alifestd_mark_csr_children_asexual(result, mutate=True)

    assert alifestd_validate(result)
    assert original.equals(phylogeny_df)

    result.index = result["id"]

    # Verify each node's children in the CSR match the actual parent-child
    # relationship from ancestor_id
    for _, row in result.iterrows():
        nid = row["id"]
        children = _get_children(result, nid)
        # Each listed child should have this node as its ancestor
        for child_id in children:
            assert result.loc[child_id, "ancestor_id"] == nid
        # Count should match
        assert len(children) == row["num_children"]


def test_empty():
    res = alifestd_mark_csr_children_asexual(alifestd_make_empty())
    assert "csr_children" in res
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
    result_df = alifestd_mark_num_children_asexual(phylogeny_df)
    result_df = alifestd_mark_csr_offsets_asexual(result_df, mutate=True)
    result_df = alifestd_mark_csr_children_asexual(
        result_df,
        mutate=mutate,
    )
    # csr_offsets: [0, 1, 2], num_children: [1, 1, 0]
    # csr_children: [1, 2, ...]  (3rd entry unused)
    flat = result_df["csr_children"].tolist()
    assert flat[0] == 1  # child of node 0
    assert flat[1] == 2  # child of node 1

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
    result_df = alifestd_mark_num_children_asexual(phylogeny_df)
    result_df = alifestd_mark_csr_offsets_asexual(result_df, mutate=True)
    result_df = alifestd_mark_csr_children_asexual(
        result_df,
        mutate=mutate,
    )
    flat = result_df["csr_children"].tolist()
    # node 0's children at [0:2]: {1, 2}
    assert set(flat[0:2]) == {1, 2}
    # node 1's children at [2:4]: {3, 4}
    assert set(flat[2:4]) == {3, 4}

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
    result_df = alifestd_mark_csr_children_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert "csr_children" in result_df.columns


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
    result_df = alifestd_mark_num_children_asexual(phylogeny_df)
    result_df = alifestd_mark_csr_offsets_asexual(result_df, mutate=True)
    result_df = alifestd_mark_csr_children_asexual(
        result_df,
        mutate=mutate,
    )
    flat = result_df["csr_children"].tolist()
    # node 0's children at [0:5]: {1, 2, 3, 4, 5}
    assert set(flat[0:5]) == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("mutate", [True, False])
def test_uses_existing_csr_offsets(mutate: bool):
    """Verify csr_children uses existing csr_offsets column if present."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
        }
    )
    result_df = alifestd_mark_csr_offsets_asexual(phylogeny_df)
    result_df = alifestd_mark_csr_children_asexual(
        result_df,
        mutate=mutate,
    )
    flat = result_df["csr_children"].tolist()
    assert flat[0] == 1
    assert flat[1] == 2

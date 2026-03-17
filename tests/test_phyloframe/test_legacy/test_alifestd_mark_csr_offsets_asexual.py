import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
)
from phyloframe.legacy import (
    alifestd_mark_csr_offsets_asexual as alifestd_mark_csr_offsets_asexual_,
)
from phyloframe.legacy import (
    alifestd_validate,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_mark_csr_offsets_asexual = enforce_dtype_stability_pandas(
    alifestd_mark_csr_offsets_asexual_
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

    result = alifestd_mark_csr_offsets_asexual(phylogeny_df)

    assert alifestd_validate(result)
    assert original.equals(phylogeny_df)

    # csr_offsets should be non-negative
    offsets = result["csr_offsets"].tolist()
    assert all(o >= 0 for o in offsets)


def test_empty():
    res = alifestd_mark_csr_offsets_asexual(alifestd_make_empty())
    assert "csr_offsets" in res
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
    result_df = alifestd_mark_csr_offsets_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # node 0 has 1 child (1), node 1 has 1 child (2), node 2 has 0
    # offsets: [0, 1, 2]
    assert result_df.loc[0, "csr_offsets"] == 0
    assert result_df.loc[1, "csr_offsets"] == 1
    assert result_df.loc[2, "csr_offsets"] == 2

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
    result_df = alifestd_mark_csr_offsets_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # num_children: [2, 2, 0, 0, 0]
    # csr_offsets: [0, 2, 4, 4, 4]
    assert result_df.loc[0, "csr_offsets"] == 0
    assert result_df.loc[1, "csr_offsets"] == 2
    assert result_df.loc[2, "csr_offsets"] == 4
    assert result_df.loc[3, "csr_offsets"] == 4
    assert result_df.loc[4, "csr_offsets"] == 4

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
    result_df = alifestd_mark_csr_offsets_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert result_df.loc[0, "csr_offsets"] == 0


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
    result_df = alifestd_mark_csr_offsets_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # num_children: [5, 0, 0, 0, 0, 0]
    # csr_offsets: [0, 5, 5, 5, 5, 5]
    assert result_df.loc[0, "csr_offsets"] == 0
    for i in range(1, 6):
        assert result_df.loc[i, "csr_offsets"] == 5


@pytest.mark.parametrize("mutate", [True, False])
def test_non_contiguous_ids(mutate: bool):
    """Non-contiguous ids use slow path."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [1, 0, 2, 3],
            "ancestor_list": ["[0]", "[None]", "[0]", "[1]"],
        }
    )
    original_df = phylogeny_df.copy()
    result_df = alifestd_mark_csr_offsets_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    result_df.index = result_df["id"]
    # node 0 has children 1,2 -> offset 0
    # node 1 has child 3 -> offset 2
    # node 2 has 0 children -> offset 3
    # node 3 has 0 children -> offset 3
    assert result_df.loc[0, "csr_offsets"] == 0
    assert result_df.loc[1, "csr_offsets"] == 2
    assert result_df.loc[2, "csr_offsets"] == 3
    assert result_df.loc[3, "csr_offsets"] == 3

    if not mutate:
        assert original_df.equals(phylogeny_df)

import os
import typing

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_find_leaf_ids,
    alifestd_find_root_ids,
    alifestd_make_empty,
    alifestd_mark_csr_children_asexual,
    alifestd_mark_csr_offsets_asexual,
    alifestd_mark_first_child_id_asexual,
    alifestd_mark_next_sibling_id_asexual,
    alifestd_mark_num_children_asexual,
    alifestd_try_add_ancestor_id_col,
)
from phyloframe.legacy import (
    alifestd_unfurl_traversal_postorder_asexual as alifestd_unfurl_traversal_postorder_asexual_,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_unfurl_traversal_postorder_asexual = enforce_dtype_stability_pandas(
    alifestd_unfurl_traversal_postorder_asexual_
)


def _prep_noop(df):
    return df


def _prep_csr(df):
    df = alifestd_mark_num_children_asexual(df, mutate=True)
    df = alifestd_mark_csr_offsets_asexual(df, mutate=True)
    df = alifestd_mark_csr_children_asexual(df, mutate=True)
    return df


def _prep_linked_list(df):
    df = alifestd_mark_first_child_id_asexual(df, mutate=True)
    df = alifestd_mark_next_sibling_id_asexual(df, mutate=True)
    return df


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
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(_prep_noop, id="no-precompute"),
        pytest.param(_prep_csr, id="csr"),
        pytest.param(_prep_linked_list, id="linked-list"),
    ],
)
def test_fuzz(phylogeny_df: pd.DataFrame, apply: typing.Callable):
    original = phylogeny_df.copy()

    result = alifestd_unfurl_traversal_postorder_asexual(
        apply(phylogeny_df.copy()),
    )

    assert original.equals(phylogeny_df)

    assert len(result) == len(phylogeny_df)
    assert set(result) == set(phylogeny_df["id"])

    # Verify valid postorder: every child appears before its parent.
    df_ref = alifestd_try_add_ancestor_id_col(phylogeny_df.copy())
    ancestor_of = dict(zip(df_ref["id"], df_ref["ancestor_id"]))
    pos = {node: i for i, node in enumerate(result)}
    for node_id in result:
        anc = ancestor_of[node_id]
        if node_id != anc:
            assert pos[node_id] < pos[anc]

    assert result[-1] in alifestd_find_root_ids(phylogeny_df)
    assert result[0] in alifestd_find_leaf_ids(phylogeny_df)


def test_empty():
    res = alifestd_unfurl_traversal_postorder_asexual(alifestd_make_empty())
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [True, False])
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(_prep_noop, id="no-precompute"),
        pytest.param(_prep_csr, id="csr"),
        pytest.param(_prep_linked_list, id="linked-list"),
    ],
)
def test_simple1(mutate: bool, apply: typing.Callable):
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
        },
    )
    original_df = phylogeny_df.copy()
    result = alifestd_unfurl_traversal_postorder_asexual(
        apply(phylogeny_df.copy()),
        mutate=mutate,
    )
    assert result.tolist() == [2, 1, 0]

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_simple2(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "id": [3, 0, 1, 2],
            "ancestor_list": ["[0]", "[1]", "[None]", "[1]"],
        },
    )
    original_df = phylogeny_df.copy()
    result = alifestd_unfurl_traversal_postorder_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert result.tolist() == [3, 2, 0, 1]

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_simple3(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "id": [4, 0, 2, 3],
            "ancestor_list": ["[None]", "[None]", "[0]", "[4]"],
        },
    )
    original_df = phylogeny_df.copy()
    result = alifestd_unfurl_traversal_postorder_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert result.tolist() == [3, 2, 4, 0]

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(_prep_noop, id="no-precompute"),
        pytest.param(_prep_csr, id="csr"),
        pytest.param(_prep_linked_list, id="linked-list"),
    ],
)
def test_simple4(mutate: bool, apply: typing.Callable):
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]"],
        },
    )
    original_df = phylogeny_df.copy()
    result = alifestd_unfurl_traversal_postorder_asexual(
        apply(phylogeny_df.copy()),
        mutate=mutate,
    )

    # Valid postorder: children before parents.
    assert set(result) == {0, 1, 2, 3}
    pos = {v: i for i, v in enumerate(result)}
    assert pos[3] < pos[1]  # child 3 before parent 1
    assert pos[1] < pos[0]  # child 1 before parent 0
    assert pos[2] < pos[0]  # child 2 before parent 0

    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_with_node_depth_col():
    """Test that pre-existing node_depth column is reused on fast path."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 0, 1],
            "node_depth": [0, 1, 1, 2],
        },
    )
    result = alifestd_unfurl_traversal_postorder_asexual(phylogeny_df)
    assert result.tolist() == [3, 2, 1, 0]


def test_with_sibling_cols():
    """Test sibling fast path with first_child_id + next_sibling_id."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 0, 1],
            "first_child_id": [1, 3, 2, 3],
            "next_sibling_id": [0, 2, 2, 3],
        },
    )
    result = alifestd_unfurl_traversal_postorder_asexual(phylogeny_df)
    # Valid postorder: children before parents
    result_list = result.tolist()
    pos = {v: i for i, v in enumerate(result_list)}
    assert pos[3] < pos[1]  # child 3 before parent 1
    assert pos[1] < pos[0]  # child 1 before parent 0
    assert pos[2] < pos[0]  # child 2 before parent 0


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
def test_sibling_fast_path_valid_postorder(phylogeny_df: pd.DataFrame):
    """Verify sibling-based fast path produces valid postorder."""
    from phyloframe.legacy import (
        alifestd_mark_first_child_id_asexual,
        alifestd_mark_next_sibling_id_asexual,
        alifestd_try_add_ancestor_id_col,
    )

    df_sib = alifestd_try_add_ancestor_id_col(phylogeny_df.copy())
    df_sib = alifestd_mark_first_child_id_asexual(df_sib, mutate=True)
    df_sib = alifestd_mark_next_sibling_id_asexual(df_sib, mutate=True)
    result = alifestd_unfurl_traversal_postorder_asexual(df_sib)

    assert len(result) == len(phylogeny_df)
    assert set(result) == set(df_sib["id"])

    ancestor_of = dict(zip(df_sib["id"], df_sib["ancestor_id"]))
    pos = {node: i for i, node in enumerate(result)}
    for node_id in result:
        ancestor_id = ancestor_of[node_id]
        if node_id != ancestor_id:
            assert pos[node_id] < pos[ancestor_id]

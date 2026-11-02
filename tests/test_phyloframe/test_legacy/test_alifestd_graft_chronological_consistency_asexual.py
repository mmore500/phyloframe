import os

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_find_chronological_inconsistency,
)
from phyloframe.legacy import (
    alifestd_graft_chronological_consistency_asexual as alifestd_graft_chronological_consistency_asexual_,
)
from phyloframe.legacy import (
    alifestd_has_contiguous_ids,
    alifestd_is_topologically_sorted,
    alifestd_make_empty,
    alifestd_parse_ancestor_ids,
    alifestd_shuffle_working_format,
    alifestd_test_leaves_isomorphic_asexual,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_graft_chronological_consistency_asexual = (
    enforce_dtype_stability_pandas(
        alifestd_graft_chronological_consistency_asexual_,
    )
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _reference_impl(
    ancestor_ids: np.ndarray, origin_times: np.ndarray
) -> np.ndarray:
    """Naive reference: nearest eligible ancestor along original lineage."""
    result = []
    for idx, ancestor_id in enumerate(ancestor_ids):
        while ancestor_id != idx:
            if origin_times[ancestor_id] <= origin_times[idx]:
                break
            if ancestor_ids[ancestor_id] == ancestor_id:
                ancestor_id = idx
            else:
                ancestor_id = ancestor_ids[ancestor_id]
        result.append(ancestor_id)
    return np.array(result)


def _is_lineage_ancestor(
    ancestor_ids: np.ndarray, descendant: int, ancestor: int
) -> bool:
    while True:
        if descendant == ancestor:
            return True
        if ancestor_ids[descendant] == descendant:
            return False
        descendant = ancestor_ids[descendant]


def test_graft_chronological_consistency_empty():
    df = alifestd_make_empty()
    df["origin_time"] = pd.Series(dtype=float)
    result = alifestd_graft_chronological_consistency_asexual(df)
    assert result.empty
    assert "ancestor_id" in result


@pytest.mark.parametrize("mutate", [True, False])
def test_graft_chronological_consistency_consistent(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 1],
            "origin_time": [0.0, 1.0, 2.0, 1.0],
        }
    )
    df_ = df.copy()
    result = alifestd_graft_chronological_consistency_asexual(
        df, mutate=mutate
    )
    assert result.equals(df_)
    if not mutate:
        assert df.equals(df_)


@pytest.mark.parametrize("mutate", [True, False])
def test_graft_chronological_consistency_inconsistent(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_id": [0, 0, 1, 2, 3, 4],
            "origin_time": [0, 5, 3, 6, 4, -1],
        }
    )
    df_ = df.copy()
    result = alifestd_graft_chronological_consistency_asexual(
        df, mutate=mutate
    )
    # 2 precedes 1 -> onto 0; 4 precedes 3 -> onto 2; 5 precedes all -> root
    assert result["ancestor_id"].tolist() == [0, 0, 0, 2, 2, 5]
    assert result["origin_time"].tolist() == df_["origin_time"].tolist()
    assert result["id"].tolist() == df_["id"].tolist()
    assert alifestd_find_chronological_inconsistency(result) is None
    if not mutate:
        assert df.equals(df_)


def test_graft_chronological_consistency_ties():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [1.0, 1.0, 1.0],
        }
    )
    result = alifestd_graft_chronological_consistency_asexual(df)
    assert result["ancestor_id"].tolist() == [0, 0, 1]


def test_graft_chronological_consistency_multiple_roots():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 1, 0, 1, 3],
            "origin_time": [10, 0, 5, 1, 0],
        }
    )
    result = alifestd_graft_chronological_consistency_asexual(df)
    assert result["ancestor_id"].tolist() == [0, 1, 2, 1, 1]


def test_graft_chronological_consistency_ancestor_list():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[1]", "[1]"],
            "origin_time": [1, 2, 0, 3],
        }
    )
    result = alifestd_graft_chronological_consistency_asexual(df)
    assert result["ancestor_id"].tolist() == [0, 0, 2, 1]
    # untouched entries keep their original tokens
    assert result["ancestor_list"].tolist() == [
        "[None]",
        "[0]",
        "[none]",
        "[1]",
    ]


def test_graft_chronological_consistency_ancestor_id_dtype():
    df = pd.DataFrame(
        {
            "id": np.array([0, 1, 2], dtype=np.uint32),
            "ancestor_id": np.array([0, 0, 1], dtype=np.uint32),
            "origin_time": [0.0, 2.0, 1.0],
        }
    )
    result = alifestd_graft_chronological_consistency_asexual(df)
    assert result["ancestor_id"].dtype == np.uint32
    assert result["ancestor_id"].tolist() == [0, 0, 0]


def test_graft_chronological_consistency_sexual():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[none]", "[0]", "[0,1]"],
            "origin_time": [0, 1, 2],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_graft_chronological_consistency_asexual(df)


def test_graft_chronological_consistency_noncontiguous():
    df = pd.DataFrame(
        {
            "id": [0, 1, 3],
            "ancestor_id": [0, 0, 1],
            "origin_time": [0, 1, 2],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_graft_chronological_consistency_asexual(df)


def test_graft_chronological_consistency_unsorted():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 2, 0],
            "origin_time": [0, 1, 2],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_graft_chronological_consistency_asexual(df)


@pytest.mark.parametrize(
    "origin_times",
    [
        ["a", "c", "b", "d"],
        pd.to_datetime(["2000", "2002", "2001", "2003"]),
    ],
)
def test_graft_chronological_consistency_non_numeric(origin_times):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 2],
            "origin_time": origin_times,
        }
    )
    result = alifestd_graft_chronological_consistency_asexual(df)
    assert result["ancestor_id"].tolist() == [0, 0, 0, 2]


def test_graft_chronological_consistency_warns():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [0, 1, 2],
            "num_children": [1, 1, 0],
        }
    )
    with pytest.warns(UserWarning):
        alifestd_graft_chronological_consistency_asexual(df)
    result = alifestd_graft_chronological_consistency_asexual(
        df, drop_topological_sensitivity=True
    )
    assert "num_children" not in result


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv"),
        pd.read_csv(f"{assets}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize("seed", range(3))
def test_graft_chronological_consistency_fuzz(
    phylogeny_df: pd.DataFrame, seed: int
):
    phylogeny_df = alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(phylogeny_df), mutate=True
    )
    rng = np.random.default_rng(seed)
    phylogeny_df["origin_time"] = phylogeny_df["origin_time"] + rng.normal(
        scale=phylogeny_df["origin_time"].std() / 4,
        size=len(phylogeny_df),
    )
    original_df = phylogeny_df.copy()
    result = alifestd_graft_chronological_consistency_asexual(
        phylogeny_df, ignore_topological_sensitivity=True
    )
    assert phylogeny_df.equals(original_df)

    assert alifestd_is_topologically_sorted(result)
    assert alifestd_find_chronological_inconsistency(result) is None
    assert result.drop(columns=["ancestor_id", "ancestor_list"]).equals(
        original_df.drop(columns=["ancestor_id", "ancestor_list"])
    )
    for id_, ancestor_list in zip(result["id"], result["ancestor_list"]):
        ancestor_ids = alifestd_parse_ancestor_ids(ancestor_list)
        assert ancestor_ids == (
            []
            if result["ancestor_id"].iloc[id_] == id_
            else [result["ancestor_id"].iloc[id_]]
        )

    original_ancestor_ids = original_df["ancestor_id"].to_numpy()
    grafted_ancestor_ids = result["ancestor_id"].to_numpy()
    expected = _reference_impl(
        original_ancestor_ids,
        original_df["origin_time"].to_numpy(),
    )
    assert np.array_equal(grafted_ancestor_ids, expected)
    for id_, ancestor_id in enumerate(grafted_ancestor_ids):
        assert _is_lineage_ancestor(original_ancestor_ids, id_, ancestor_id)


def _label_parents(phylogeny_df: pd.DataFrame) -> dict:
    """Map each taxon's label to its parent's label."""
    assert alifestd_has_contiguous_ids(phylogeny_df)
    labels = phylogeny_df["label"].to_numpy()
    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    return dict(zip(labels, labels[ancestor_ids]))


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv"),
        pd.read_csv(f"{assets}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize("seed", range(3))
def test_graft_chronological_consistency_row_order_invariant(
    phylogeny_df: pd.DataFrame, seed: int
):
    phylogeny_df = alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(phylogeny_df), mutate=True
    )
    rng = np.random.default_rng(seed)
    phylogeny_df["origin_time"] = phylogeny_df["origin_time"] + rng.normal(
        scale=phylogeny_df["origin_time"].std() / 4,
        size=len(phylogeny_df),
    )
    phylogeny_df["label"] = phylogeny_df["id"]

    shuffled_df = alifestd_shuffle_working_format(phylogeny_df, seed=seed)
    assert alifestd_is_topologically_sorted(shuffled_df)
    assert alifestd_has_contiguous_ids(shuffled_df)
    assert not shuffled_df["label"].equals(phylogeny_df["label"])

    result = alifestd_graft_chronological_consistency_asexual(
        phylogeny_df, ignore_topological_sensitivity=True
    )
    shuffled_result = alifestd_graft_chronological_consistency_asexual(
        shuffled_df, ignore_topological_sensitivity=True
    )
    assert alifestd_find_chronological_inconsistency(shuffled_result) is None
    assert alifestd_test_leaves_isomorphic_asexual(
        result, shuffled_result, taxon_label="label"
    )
    assert _label_parents(result) == _label_parents(shuffled_result)

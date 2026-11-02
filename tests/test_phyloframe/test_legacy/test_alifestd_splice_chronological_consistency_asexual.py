import os

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_coerce_chronological_consistency,
    alifestd_count_root_nodes,
    alifestd_find_chronological_inconsistency,
    alifestd_make_empty,
    alifestd_parse_ancestor_ids,
    alifestd_regraft_chronological_consistency_asexual,
    alifestd_shuffle_working_format,
)
from phyloframe.legacy import (
    alifestd_splice_chronological_consistency_asexual as alifestd_splice_chronological_consistency_asexual_,
)
from phyloframe.legacy import (
    alifestd_test_leaves_isomorphic_asexual,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_splice_chronological_consistency_asexual = (
    enforce_dtype_stability_pandas(
        alifestd_splice_chronological_consistency_asexual_,
    )
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

_assets = pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv"),
        pd.read_csv(f"{assets}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets}/nk_tournamentselection.csv"),
    ],
)


def _perturbed(phylogeny_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    phylogeny_df = alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(phylogeny_df), mutate=True
    )
    rng = np.random.default_rng(seed)
    phylogeny_df["origin_time"] = phylogeny_df["origin_time"] + rng.normal(
        scale=phylogeny_df["origin_time"].std() / 4,
        size=len(phylogeny_df),
    )
    return phylogeny_df


def _nearest_original_ancestors(
    ancestor_ids: np.ndarray, num_taxa: int
) -> np.ndarray:
    """For each original taxon, find nearest original strict ancestor,
    skipping spliced nodes; self if none."""
    result = []
    for idx in range(num_taxa):
        node = ancestor_ids[idx]
        while node >= num_taxa and ancestor_ids[node] != node:
            node = ancestor_ids[node]
        result.append(idx if node >= num_taxa or node == idx else node)
    return np.array(result)


def test_splice_chronological_consistency_empty():
    df = alifestd_make_empty()
    df["origin_time"] = pd.Series(dtype=float)
    result = alifestd_splice_chronological_consistency_asexual(df)
    assert result.empty
    assert "ancestor_id" in result


@pytest.mark.parametrize("mutate", [True, False])
def test_splice_chronological_consistency_consistent(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 1],
            "origin_time": [0.0, 1.0, 2.0, 1.0],
        }
    )
    df_ = df.copy()
    result = alifestd_splice_chronological_consistency_asexual(
        df, mutate=mutate
    )
    assert result.equals(df_)
    if not mutate:
        assert df.equals(df_)


@pytest.mark.parametrize("mutate", [True, False])
def test_splice_chronological_consistency_inconsistent(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_id": [0, 0, 1, 2, 3, 4],
            "origin_time": [0, 5, 3, 6, 4, -1],
            "trait": [1, 2, 3, 4, 5, 6],
        }
    )
    df_ = df.copy()
    result = alifestd_splice_chronological_consistency_asexual(
        df, mutate=mutate, ignore_topological_sensitivity=True
    )
    # 2 precedes 1 -> spliced 6 on edge 0-1
    # 4 precedes 3 -> spliced 7 on edge 2-3
    # 5 precedes all -> spliced 8 as new root, above 0
    assert result["id"].tolist() == [*range(9)]
    assert result["ancestor_id"].tolist() == [8, 6, 6, 7, 7, 8, 0, 2, 8]
    assert result["origin_time"].tolist() == [0, 5, 3, 6, 4, -1, 3, 4, -1]
    assert result["trait"].iloc[:6].tolist() == df_["trait"].tolist()
    assert result["trait"].iloc[6:].isna().all()
    assert alifestd_count_root_nodes(result) == 1
    assert (
        alifestd_find_chronological_inconsistency(
            alifestd_to_working_format(result),
        )
        is None
    )
    if not mutate:
        assert df.equals(df_)


def test_splice_chronological_consistency_shared_edge():
    # 2 and 3 both precede 1, so both splice onto edge 0-1, ordered by time
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 1],
            "origin_time": [0.0, 5.0, 2.0, 1.0],
        }
    )
    result = alifestd_splice_chronological_consistency_asexual(df)
    assert result["ancestor_id"].tolist() == [0, 4, 4, 5, 5, 0]
    assert result["origin_time"].tolist() == [0.0, 5.0, 2.0, 1.0, 2.0, 1.0]


def test_splice_chronological_consistency_ties():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [1.0, 1.0, 1.0],
        }
    )
    result = alifestd_splice_chronological_consistency_asexual(df)
    assert result["ancestor_id"].tolist() == [0, 0, 1]


def test_splice_chronological_consistency_exact_ancestor():
    # 2 precedes 1, but originates exactly with 0; splice below 0
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
            "origin_time": [1.0, 5.0, 1.0, 2.0],
        }
    )
    result = alifestd_splice_chronological_consistency_asexual(df)
    assert result["ancestor_id"].tolist() == [0, 4, 4, 0, 0]
    assert result["origin_time"].tolist() == [1.0, 5.0, 1.0, 2.0, 1.0]


def test_splice_chronological_consistency_ancestor_list():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[1]", "[1]"],
            "origin_time": [1, 2, 0, 3],
        }
    )
    result = alifestd_splice_chronological_consistency_asexual(df)
    assert result["ancestor_id"].tolist() == [4, 0, 4, 1, 4]
    # untouched entries keep their original tokens
    assert result["ancestor_list"].tolist() == [
        "[4]",
        "[0]",
        "[4]",
        "[1]",
        "[none]",
    ]


def test_splice_chronological_consistency_ancestor_id_dtype():
    df = pd.DataFrame(
        {
            "id": np.array([0, 1, 2], dtype=np.uint32),
            "ancestor_id": np.array([0, 0, 1], dtype=np.uint32),
            "origin_time": [0.0, 2.0, 1.0],
        }
    )
    result = alifestd_splice_chronological_consistency_asexual(df)
    assert result["id"].dtype == np.uint32
    assert result["ancestor_id"].dtype == np.uint32
    assert result["ancestor_id"].tolist() == [0, 3, 3, 0]


@pytest.mark.parametrize(
    "origin_times",
    [
        ["a", "c", "b", "d"],
        pd.to_datetime(["2000", "2002", "2001", "2003"]),
    ],
)
def test_splice_chronological_consistency_non_numeric(origin_times):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 2],
            "origin_time": origin_times,
        }
    )
    result = alifestd_splice_chronological_consistency_asexual(df)
    assert result["ancestor_id"].tolist() == [0, 4, 4, 2, 0]
    assert result["origin_time"].iloc[4] == df["origin_time"].iloc[2]


def test_splice_chronological_consistency_sexual():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[none]", "[0]", "[0,1]"],
            "origin_time": [0, 1, 2],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_splice_chronological_consistency_asexual(df)


def test_splice_chronological_consistency_noncontiguous():
    df = pd.DataFrame(
        {
            "id": [0, 1, 3],
            "ancestor_id": [0, 0, 1],
            "origin_time": [0, 1, 2],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_splice_chronological_consistency_asexual(df)


def test_splice_chronological_consistency_unsorted():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 2, 0],
            "origin_time": [0, 1, 2],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_splice_chronological_consistency_asexual(df)


def test_splice_chronological_consistency_warns():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [0, 2, 1],
            "num_children": [1, 1, 0],
        }
    )
    with pytest.warns(UserWarning):
        alifestd_splice_chronological_consistency_asexual(df)
    result = alifestd_splice_chronological_consistency_asexual(
        df, drop_topological_sensitivity=True
    )
    assert "num_children" not in result


@_assets
@pytest.mark.parametrize("seed", range(3))
def test_splice_chronological_consistency_fuzz(
    phylogeny_df: pd.DataFrame, seed: int
):
    phylogeny_df = _perturbed(phylogeny_df, seed)
    original_df = phylogeny_df.copy()
    num_taxa = len(phylogeny_df)

    result = alifestd_splice_chronological_consistency_asexual(
        phylogeny_df, ignore_topological_sensitivity=True
    )
    assert phylogeny_df.equals(original_df)
    assert result["id"].tolist() == [*range(len(result))]
    assert len(result) > num_taxa  # perturbation causes inversions

    # chronologically consistent, with same number of roots
    assert (
        alifestd_find_chronological_inconsistency(
            alifestd_to_working_format(result),
        )
        is None
    )
    assert alifestd_count_root_nodes(result) == alifestd_count_root_nodes(
        original_df
    )

    # original taxa are unchanged, apart from topology (and null upcasting)
    topology_cols = ["ancestor_id", "ancestor_list"]
    pd.testing.assert_frame_equal(
        result.iloc[:num_taxa].drop(columns=topology_cols),
        original_df.drop(columns=topology_cols),
        check_dtype=False,
    )
    for id_, ancestor_id, ancestor_list in zip(
        result["id"], result["ancestor_id"], result["ancestor_list"]
    ):
        assert alifestd_parse_ancestor_ids(ancestor_list) == (
            [] if id_ == ancestor_id else [ancestor_id]
        )

    # spliced nodes are bifurcations, subtending zero-length branch to taxon
    ancestor_ids = result["ancestor_id"].to_numpy()
    origin_times = result["origin_time"].to_numpy()
    for spliced in range(num_taxa, len(result)):
        (children,) = np.nonzero(
            (ancestor_ids == spliced) & (np.arange(len(result)) != spliced),
        )
        assert len(children) == 2
        assert (origin_times[children] == origin_times[spliced]).any()
        assert (
            result.iloc[spliced]
            .drop(["id", "origin_time"] + topology_cols)
            .isna()
            .all()
        )

    # skipping spliced nodes, matches regraft without splicing
    regrafted_df = alifestd_regraft_chronological_consistency_asexual(
        original_df, ignore_topological_sensitivity=True
    )
    assert np.array_equal(
        _nearest_original_ancestors(ancestor_ids, num_taxa),
        regrafted_df["ancestor_id"].to_numpy(),
    )


@_assets
@pytest.mark.parametrize("seed", range(3))
def test_splice_chronological_consistency_row_order_invariant(
    phylogeny_df: pd.DataFrame, seed: int
):
    phylogeny_df = _perturbed(phylogeny_df, seed)
    phylogeny_df["label"] = phylogeny_df["id"]
    shuffled_df = alifestd_shuffle_working_format(phylogeny_df, seed=seed)
    assert not shuffled_df["label"].equals(phylogeny_df["label"])

    result = alifestd_splice_chronological_consistency_asexual(
        phylogeny_df, ignore_topological_sensitivity=True
    )
    shuffled_result = alifestd_splice_chronological_consistency_asexual(
        shuffled_df, ignore_topological_sensitivity=True
    )
    assert len(result) == len(shuffled_result)
    assert alifestd_test_leaves_isomorphic_asexual(
        result, shuffled_result, taxon_label="label"
    )

    def label_parents(df: pd.DataFrame) -> dict:
        labels = df["label"].to_numpy()
        nearest = _nearest_original_ancestors(
            df["ancestor_id"].to_numpy(), len(phylogeny_df)
        )
        return dict(zip(labels[: len(phylogeny_df)], labels[nearest]))

    assert label_parents(result) == label_parents(shuffled_result)


def _consistent_cases(phylogeny_df: pd.DataFrame, seed: int) -> list:
    """Chronologically consistent phylogenies: as-is, coerced after
    perturbation, and spliced after perturbation (in working format)."""
    phylogeny_df = alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(phylogeny_df), mutate=True
    )
    perturbed_df = _perturbed(phylogeny_df, seed)
    coerced_df = alifestd_coerce_chronological_consistency(
        perturbed_df
    ).reset_index(drop=True)
    spliced_df = alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(
            alifestd_splice_chronological_consistency_asexual(
                perturbed_df, ignore_topological_sensitivity=True
            ),
        ),
        mutate=True,
    )
    return [phylogeny_df, coerced_df, spliced_df]


@_assets
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("case", range(3))
def test_splice_chronological_consistency_consistent_unchanged(
    phylogeny_df: pd.DataFrame, seed: int, case: int
):
    phylogeny_df = _consistent_cases(phylogeny_df, seed)[case]
    assert alifestd_find_chronological_inconsistency(phylogeny_df) is None
    original_df = phylogeny_df.copy()

    result = alifestd_splice_chronological_consistency_asexual(phylogeny_df)
    assert phylogeny_df.equals(original_df)
    assert result.equals(original_df)

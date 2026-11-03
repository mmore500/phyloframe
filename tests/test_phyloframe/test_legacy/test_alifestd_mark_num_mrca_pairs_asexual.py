import collections
import itertools
import os

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_find_pair_mrca_id_asexual,
    alifestd_mark_leaves,
)
from phyloframe.legacy import (
    alifestd_mark_num_mrca_pairs_asexual as alifestd_mark_num_mrca_pairs_asexual_,
)
from phyloframe.legacy import (
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_mark_num_mrca_pairs_asexual = enforce_dtype_stability_pandas(
    alifestd_mark_num_mrca_pairs_asexual_,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _make_df() -> pd.DataFrame:
    """Tree:
    0 (root)
    +-- 1
    |   +-- 4
    |   +-- 5
    +-- 2
    |   +-- 6
    +-- 3
    7 (root)
    +-- 8
    +-- 9
    """
    return pd.DataFrame(
        {
            "id": [*range(10)],
            "ancestor_id": [0, 0, 0, 0, 1, 1, 2, 7, 7, 7],
        },
    )


def _reference_num_mrca_pairs(phylogeny_df: pd.DataFrame) -> list:
    phylogeny_df = alifestd_mark_leaves(phylogeny_df)
    leaf_ids = phylogeny_df.loc[phylogeny_df["is_leaf"], "id"]
    counts = collections.Counter(
        alifestd_find_pair_mrca_id_asexual(
            phylogeny_df,
            first,
            second,
            is_topologically_sorted=True,
            has_contiguous_ids=True,
        )
        for first, second in itertools.combinations(leaf_ids, 2)
    )
    return [counts[id_] for id_ in phylogeny_df["id"]]


def _random_tree(seed: int, num_nodes: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ancestor_ids = [
        idx if idx == 0 or rng.random() < 0.05 else rng.integers(0, idx)
        for idx in range(num_nodes)
    ]
    return pd.DataFrame(
        {"id": np.arange(num_nodes), "ancestor_id": ancestor_ids}
    )


@pytest.mark.parametrize("mutate", [True, False])
def test_hand_built_tree(mutate: bool):
    df = _make_df()
    original_df = df.copy()
    res = alifestd_mark_num_mrca_pairs_asexual(df, mutate=mutate)
    # polytomy at 0; unifurcation at 2; leaves 3, 4, 5, 6, 8, 9
    assert res["num_mrca_pairs"].tolist() == [5, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    assert np.issubdtype(res["num_mrca_pairs"].dtype, np.integer)
    assert res["num_mrca_pairs"].tolist() == _reference_num_mrca_pairs(df)
    if not mutate:
        assert df.equals(original_df)


def test_mark_as():
    res = alifestd_mark_num_mrca_pairs_asexual(_make_df(), mark_as="blueberry")
    assert "blueberry" in res.columns
    assert "num_mrca_pairs" not in res.columns


def test_ancestor_list():
    df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_list": ["[none]", "[0]", "[0]"]},
    )
    res = alifestd_mark_num_mrca_pairs_asexual(df)
    assert res["num_mrca_pairs"].tolist() == [1, 0, 0]


def test_singleton():
    df = pd.DataFrame({"id": [0], "ancestor_id": [0]})
    res = alifestd_mark_num_mrca_pairs_asexual(df)
    assert res["num_mrca_pairs"].tolist() == [0]


def test_empty():
    df = pd.DataFrame({"id": [], "ancestor_id": []}, dtype=int)
    res = alifestd_mark_num_mrca_pairs_asexual(df)
    assert "num_mrca_pairs" in res.columns
    assert len(res) == 0


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"id": [0, 2], "ancestor_id": [0, 0]}),  # non-contiguous
        pd.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 2, 0]}),  # unsorted
        pd.DataFrame(  # sexual
            {"id": [0, 1, 2], "ancestor_list": ["[none]", "[0]", "[0,1]"]},
        ),
    ],
)
def test_unsupported(df: pd.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mark_num_mrca_pairs_asexual(df)


@pytest.mark.parametrize("seed", range(5))
def test_matches_pair_mrca_random(seed: int):
    df = _random_tree(seed, num_nodes=60)
    res = alifestd_mark_num_mrca_pairs_asexual(df)
    assert res["num_mrca_pairs"].tolist() == _reference_num_mrca_pairs(df)


def test_matches_pair_mrca_asset():
    df = alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(
            pd.read_csv(
                f"{assets}/example-standard-toy-asexual-phylogeny.csv"
            ),
        ),
    )
    res = alifestd_mark_num_mrca_pairs_asexual(df)
    assert res["num_mrca_pairs"].tolist() == _reference_num_mrca_pairs(df)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv"),
        pd.read_csv(f"{assets}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets}/nk_tournamentselection.csv"),
    ],
)
def test_sums_to_num_leaf_pairs(phylogeny_df: pd.DataFrame):
    df = alifestd_mark_leaves(
        alifestd_try_add_ancestor_id_col(
            alifestd_to_working_format(phylogeny_df)
        )
    )
    res = alifestd_mark_num_mrca_pairs_asexual(df)
    is_root = res["id"] == res["ancestor_id"]
    assert is_root.sum() == 1
    num_leaves = res["is_leaf"].sum()
    assert res["num_mrca_pairs"].sum() == num_leaves * (num_leaves - 1) // 2
    assert (res["num_mrca_pairs"] >= 0).all()
    assert (res.loc[res["is_leaf"], "num_mrca_pairs"] == 0).all()

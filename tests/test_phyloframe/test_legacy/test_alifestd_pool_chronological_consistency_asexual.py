import os
import typing

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_coerce_chronological_consistency,
    alifestd_find_chronological_inconsistency,
    alifestd_find_leaf_ids,
    alifestd_make_empty,
    alifestd_mark_num_mrca_leaf_pairs_asexual,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)
from phyloframe.legacy._alifestd_pool_chronological_consistency_asexual import (
    alifestd_pool_chronological_consistency_asexual as alifestd_pool_chronological_consistency_asexual_,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_pool_chronological_consistency_asexual = (
    enforce_dtype_stability_pandas(
        alifestd_pool_chronological_consistency_asexual_,
    )
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _prepare(phylogeny_df: pd.DataFrame) -> pd.DataFrame:
    return alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(phylogeny_df), mutate=True
    )[["id", "ancestor_id", "origin_time"]].astype({"origin_time": float})


def _make_forest(*phylogeny_dfs: pd.DataFrame) -> pd.DataFrame:
    """Concatenate phylogenies into multi-root forest, with contiguous ids
    and topologically sorted rows."""
    offset = 0
    parts = []
    for phylogeny_df in phylogeny_dfs:
        phylogeny_df = _prepare(phylogeny_df)
        phylogeny_df["id"] += offset
        phylogeny_df["ancestor_id"] += offset
        offset += len(phylogeny_df)
        parts.append(phylogeny_df)
    return pd.concat(parts, ignore_index=True)


_assets = pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv"),
        pd.read_csv(f"{assets}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets}/nk_tournamentselection.csv"),
        pd.read_csv(
            f"{assets}/example-standard-toy-asexual-bifurcating-phylogeny.csv"
        ),
        _make_forest(  # multiple roots
            pd.read_csv(f"{assets}/nk_lexicaseselection.csv"),
            pd.read_csv(f"{assets}/nk_tournamentselection.csv"),
            pd.read_csv(f"{assets}/nk_ecoeaselection.csv"),
        ),
    ],
)


def _add_noise(phylogeny_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    phylogeny_df = phylogeny_df.copy()
    phylogeny_df["origin_time"] += rng.normal(
        scale=phylogeny_df["origin_time"].std() / 4 + 1,
        size=len(phylogeny_df),
    )
    return phylogeny_df


def _pairs_weighted_sum(phylogeny_df: pd.DataFrame) -> float:
    weights = alifestd_mark_num_mrca_leaf_pairs_asexual(phylogeny_df)[
        "num_mrca_leaf_pairs"
    ]
    return (weights * phylogeny_df["origin_time"]).sum()


def _weighted(phylogeny_df: pd.DataFrame, weight_by) -> pd.DataFrame:
    """Add `num_mrca_leaf_pairs` weight column, if requested."""
    if weight_by == "num_mrca_leaf_pairs":
        phylogeny_df = alifestd_mark_num_mrca_leaf_pairs_asexual(
            phylogeny_df
        ).drop(columns="num_leaves")
    return phylogeny_df


def test_pool_chronological_consistency_empty():
    result = alifestd_pool_chronological_consistency_asexual(
        alifestd_make_empty(),
    )
    assert result.empty


@pytest.mark.parametrize("mutate", [True, False])
def test_pool_chronological_consistency_consistent(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            "origin_time": [0.1, 0.1, 0.7, 0.3, 1.0, 0.7, 0.9],
        }
    )
    df_ = df.copy()
    result = alifestd_pool_chronological_consistency_asexual(df, mutate=mutate)
    assert result.equals(df_)
    if not mutate:
        assert df.equals(df_)


@pytest.mark.parametrize("mutate", [True, False])
def test_pool_chronological_consistency_inconsistent(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            "origin_time": [5, 1, 6, 10, 10, 10, 10],
            "trait": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    df = _weighted(df, "num_mrca_leaf_pairs")
    df_ = df.copy()
    result = alifestd_pool_chronological_consistency_asexual(
        df, mutate=mutate, weight_by="num_mrca_leaf_pairs"
    )
    # mrca leaf pair weights are 4, 1, 1 for inner nodes; 1 precedes 0, so
    # 0 and 1 pool to (4 * 5 + 1 * 1) / 5
    assert result["origin_time"].tolist() == [4.2, 4.2, 6, 10, 10, 10, 10]
    assert result["origin_time"].dtype == np.float64
    for col in "id", "ancestor_id", "trait":
        assert result[col].equals(df_[col])
    if not mutate:
        assert df.equals(df_)


def test_pool_chronological_consistency_uniform():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            "origin_time": [5.0, 1.0, 6.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    result = alifestd_pool_chronological_consistency_asexual(df)
    assert result["origin_time"].tolist() == [3, 3, 6, 10, 10, 10, 10]


def test_pool_chronological_consistency_exact():
    # greedy pooling of 2 into 0 first would give mean 16 / 3 for all taxa;
    # optimal pools only 0 and 1
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
            "origin_time": [10.0, 0.0, 6.0],
        }
    )
    result = alifestd_pool_chronological_consistency_asexual(df, pin_by=None)
    assert result["origin_time"].tolist() == [5.0, 5.0, 6.0]


def test_pool_chronological_consistency_pin_by_custom():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [4.0, 2.0, 6.0],
            "is_fixed": [True, False, False],
        }
    )
    result = alifestd_pool_chronological_consistency_asexual(
        df, pin_by="is_fixed"
    )
    # pinned root bounds descendants from below
    assert result["origin_time"].tolist() == [4.0, 4.0, 6.0]
    assert "is_fixed" in result.columns
    assert "is_leaf" not in result.columns


def test_pool_chronological_consistency_pin_by_internal():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [8.0, 5.0, 0.0],
            "is_fixed": [False, True, False],
        }
    )
    result = alifestd_pool_chronological_consistency_asexual(
        df, pin_by="is_fixed"
    )
    # taxa are not pooled across pinned taxon
    assert result["origin_time"].tolist() == [5.0, 5.0, 5.0]


@_assets
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("weight_by", ["num_mrca_leaf_pairs", None])
def test_pool_chronological_consistency_pin_by_fuzz(
    phylogeny_df: pd.DataFrame, seed: int, weight_by: typing.Optional[str]
):
    phylogeny_df = alifestd_coerce_chronological_consistency(
        _prepare(phylogeny_df)
    )
    rng = np.random.default_rng(seed)
    is_fixed = rng.random(len(phylogeny_df)) < 0.2
    noised = _add_noise(phylogeny_df, seed)
    phylogeny_df["origin_time"] = np.where(
        is_fixed, phylogeny_df["origin_time"], noised["origin_time"]
    )
    phylogeny_df["is_fixed"] = is_fixed
    phylogeny_df = _weighted(phylogeny_df, weight_by)

    result = alifestd_pool_chronological_consistency_asexual(
        phylogeny_df, weight_by=weight_by, pin_by="is_fixed"
    )
    assert alifestd_find_chronological_inconsistency(result) is None
    assert (
        result.loc[is_fixed, "origin_time"]
        == phylogeny_df.loc[is_fixed, "origin_time"]
    ).all()


def test_pool_chronological_consistency_pin_by_reuses_is_leaf():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
            "origin_time": [5.0, 3.0, 10.0],
        }
    )
    result = alifestd_pool_chronological_consistency_asexual(df)
    assert result["origin_time"].tolist() == [3.0, 3.0, 10.0]
    assert "is_leaf" not in result.columns

    # precomputed column is used as-is, and retained
    df["is_leaf"] = False
    result = alifestd_pool_chronological_consistency_asexual(df)
    assert result["origin_time"].tolist() == [4.0, 4.0, 10.0]
    assert result["is_leaf"].tolist() == [False] * 3


def test_pool_chronological_consistency_pin_by_non_bool():
    df = pd.DataFrame(
        {
            "id": [0, 1],
            "ancestor_id": [0, 0],
            "origin_time": [5.0, 3.0],
            "pin": [0, 1],
        }
    )
    with pytest.raises(ValueError):
        alifestd_pool_chronological_consistency_asexual(df, pin_by="pin")


def test_pool_chronological_consistency_chain():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 2],
            "origin_time": [4.0, 3.0, 2.0, 1.0],
        }
    )
    result = alifestd_pool_chronological_consistency_asexual(df, pin_by=None)
    assert result["origin_time"].tolist() == [2.5] * 4


def test_pool_chronological_consistency_unifurcation():
    # 0: root, bifurcating into unifurcation 1 and leaf 5
    # 1: unifurcation onto 2, which bifurcates into leaves 3 and 4
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_id": [0, 0, 1, 2, 2, 0],
            "origin_time": [5.0, 7.0, 2.0, 10.0, 10.0, 10.0],
        }
    )
    result = alifestd_pool_chronological_consistency_asexual(
        _weighted(df, "num_mrca_leaf_pairs"), weight_by="num_mrca_leaf_pairs"
    )
    # 0 (weight 2) pools with 2 (weight 1) to (2 * 5 + 2) / 3, across
    # unifurcation 1 (weight 0), which is clamped between them
    assert result["origin_time"].tolist() == [4, 4, 4, 10, 10, 10]


def test_pool_chronological_consistency_unifurcation_clamped():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_id": [0, 0, 1, 2, 2, 0],
            "origin_time": [5.0, 1.0, 8.0, 10.0, 10.0, 10.0],
        }
    )
    result = alifestd_pool_chronological_consistency_asexual(
        _weighted(df, "num_mrca_leaf_pairs"), weight_by="num_mrca_leaf_pairs"
    )
    assert result["origin_time"].tolist() == [5, 5, 8, 10, 10, 10]

    df.loc[1, "origin_time"] = 9.0
    result = alifestd_pool_chronological_consistency_asexual(
        _weighted(df, "num_mrca_leaf_pairs"), weight_by="num_mrca_leaf_pairs"
    )
    assert result["origin_time"].tolist() == [5, 8, 8, 10, 10, 10]


def test_pool_chronological_consistency_polytomy():
    # 0: root polytomy with leaves 1, 2 and inner 3 (weight 5 at root)
    # 3: bifurcation with leaves 4, 5 (weight 1)
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_id": [0, 0, 0, 0, 3, 3],
            "origin_time": [3.0, 9.0, 9.0, 0.0, 9.0, 9.0],
        }
    )
    result = alifestd_pool_chronological_consistency_asexual(
        _weighted(df, "num_mrca_leaf_pairs"), weight_by="num_mrca_leaf_pairs"
    )
    assert result["origin_time"].tolist() == [2.5, 9, 9, 2.5, 9, 9]


def test_pool_chronological_consistency_multiple_roots():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "ancestor_id": [0, 1, 0, 1, 2, 2, 3, 3, 0, 1],
            "origin_time": [5.0, 0.0, 1.0, 2.0, 9, 9, 9, 9, 9, 9],
        }
    )
    result = alifestd_pool_chronological_consistency_asexual(
        _weighted(df, "num_mrca_leaf_pairs"), weight_by="num_mrca_leaf_pairs"
    )
    # tree rooted at 0: weights 2 (root), 1 (2); pools to (2 * 5 + 1) / 3
    # tree rooted at 1: consistent, unchanged
    expected = [11 / 3, 0, 11 / 3, 2, 9, 9, 9, 9, 9, 9]
    assert result["origin_time"].tolist() == pytest.approx(expected)


@pytest.mark.parametrize("weight_by", ["num_mrca_leaf_pairs", None])
def test_pool_chronological_consistency_no_pin(weight_by):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
            "origin_time": [5.0, 3.0, 10.0],
        }
    )
    df = _weighted(df, weight_by)
    result = alifestd_pool_chronological_consistency_asexual(
        df, weight_by=weight_by
    )
    assert result["origin_time"].tolist() == [3.0, 3.0, 10.0]

    result = alifestd_pool_chronological_consistency_asexual(
        df, weight_by=weight_by, pin_by=None
    )
    if weight_by == "num_mrca_leaf_pairs":  # leaves have zero weight
        assert result["origin_time"].tolist() == [5.0, 5.0, 10.0]
    else:
        assert result["origin_time"].tolist() == [4.0, 4.0, 10.0]


def test_pool_chronological_consistency_custom_weights():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 1],
            "origin_time": [6.0, 0.0, 9.0, 9.0],
            "w": [1.0, 2.0, 0.0, 0.0],
        }
    )
    result = alifestd_pool_chronological_consistency_asexual(df, weight_by="w")
    assert result["origin_time"].tolist() == [2.0, 2.0, 9.0, 9.0]


@pytest.mark.parametrize(
    "weight_by",
    [
        np.array([1.0, -1.0, 0.0]),
        np.array([1.0, np.nan, 0.0]),
        np.array([1.0, np.inf, 0.0]),
    ],
)
def test_pool_chronological_consistency_invalid_weights(
    weight_by: np.ndarray,
):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [1.0, 0.0, 2.0],
            "w": weight_by,
        }
    )
    with pytest.raises(ValueError):
        alifestd_pool_chronological_consistency_asexual(df, weight_by="w")


def test_pool_chronological_consistency_nan_origin_time():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [1.0, np.nan, 2.0],
        }
    )
    with pytest.raises(ValueError):
        alifestd_pool_chronological_consistency_asexual(df)


@pytest.mark.parametrize(
    "dtype, expected_dtype",
    [
        (np.int64, np.float64),
        (np.uint32, np.float64),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ],
)
def test_pool_chronological_consistency_dtype(
    dtype: type, expected_dtype: type
):
    df = pd.DataFrame(
        {
            "id": np.array([0, 1, 2, 3], dtype=np.uint32),
            "ancestor_id": np.array([0, 0, 1, 1], dtype=np.uint32),
            "origin_time": np.array([2, 1, 3, 3], dtype=dtype),
        }
    )
    result = alifestd_pool_chronological_consistency_asexual(df)
    assert result["origin_time"].dtype == expected_dtype
    assert result["origin_time"].tolist() == [1.5, 1.5, 3, 3]


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(  # non-contiguous
            {
                "id": [0, 1, 3],
                "ancestor_id": [0, 0, 1],
                "origin_time": [0, 1, 2],
            },
        ),
        pd.DataFrame(  # unsorted
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
                "origin_time": [0, 1, 2],
            },
        ),
        pd.DataFrame(  # sexual
            {
                "id": [0, 1, 2],
                "ancestor_list": ["[none]", "[0]", "[0,1]"],
                "origin_time": [0, 1, 2],
            },
        ),
        pd.DataFrame(  # non-numeric
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": ["a", "c", "b"],
            },
        ),
    ],
)
def test_pool_chronological_consistency_unsupported(df: pd.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_pool_chronological_consistency_asexual(df)


@_assets
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("weight_by", ["num_mrca_leaf_pairs", None])
@pytest.mark.parametrize("pin_by", ["is_leaf", None])
def test_pool_chronological_consistency_fuzz(
    phylogeny_df: pd.DataFrame,
    seed: int,
    weight_by,
    pin_by: typing.Optional[str],
):
    phylogeny_df = _add_noise(_prepare(phylogeny_df), seed)
    phylogeny_df = _weighted(phylogeny_df, weight_by)
    phylogeny_df["trait"] = np.arange(len(phylogeny_df))
    original = phylogeny_df.copy()

    result = alifestd_pool_chronological_consistency_asexual(
        phylogeny_df, weight_by=weight_by, pin_by=pin_by
    )
    assert original.equals(phylogeny_df)
    assert alifestd_find_chronological_inconsistency(result) is None
    for col in "id", "ancestor_id", "trait":
        assert result[col].equals(original[col])

    leaf_ids = [*alifestd_find_leaf_ids(original)]
    if pin_by is not None:
        assert (
            result.loc[leaf_ids, "origin_time"]
            == original.loc[leaf_ids, "origin_time"]
        ).all()


@_assets
@pytest.mark.parametrize("seed", range(3))
def test_pool_chronological_consistency_conserves_mrca_leaf_pairs(
    phylogeny_df: pd.DataFrame, seed: int
):
    phylogeny_df = _add_noise(_prepare(phylogeny_df), seed)
    # ensure leaves do not precede ancestors, so no clamping is needed
    leaf_ids = [*alifestd_find_leaf_ids(phylogeny_df)]
    phylogeny_df.loc[leaf_ids, "origin_time"] += (
        phylogeny_df["origin_time"].max()
        - phylogeny_df["origin_time"].min()
        + 1
    )

    result = alifestd_pool_chronological_consistency_asexual(
        _weighted(phylogeny_df, "num_mrca_leaf_pairs"),
        weight_by="num_mrca_leaf_pairs",
    )
    assert alifestd_find_chronological_inconsistency(result) is None
    assert _pairs_weighted_sum(result) == pytest.approx(
        _pairs_weighted_sum(phylogeny_df), rel=1e-12
    )


@_assets
@pytest.mark.parametrize("seed", range(3))
def test_pool_chronological_consistency_conserves_uniform(
    phylogeny_df: pd.DataFrame, seed: int
):
    phylogeny_df = _add_noise(_prepare(phylogeny_df), seed)

    result = alifestd_pool_chronological_consistency_asexual(
        phylogeny_df, pin_by=None
    )
    assert alifestd_find_chronological_inconsistency(result) is None
    assert result["origin_time"].sum() == pytest.approx(
        phylogeny_df["origin_time"].sum(), rel=1e-12
    )

    # pooling is least-squares optimal, so moves less than coercion
    coerced = alifestd_coerce_chronological_consistency(phylogeny_df)
    pooled_change = (
        (result["origin_time"] - phylogeny_df["origin_time"]) ** 2
    ).sum()
    coerced_change = (
        (coerced["origin_time"].to_numpy() - phylogeny_df["origin_time"]) ** 2
    ).sum()
    assert pooled_change <= coerced_change


@_assets
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("case", ["as-is", "coerced"])
@pytest.mark.parametrize("weight_by", ["num_mrca_leaf_pairs", None])
@pytest.mark.parametrize("pin_by", ["is_leaf", None])
def test_pool_chronological_consistency_consistent_unchanged(
    phylogeny_df: pd.DataFrame,
    seed: int,
    case: str,
    weight_by,
    pin_by: typing.Optional[str],
):
    phylogeny_df = _prepare(phylogeny_df)
    if case == "coerced":
        phylogeny_df = alifestd_coerce_chronological_consistency(
            _add_noise(phylogeny_df, seed)
        ).reset_index(drop=True)
    assert alifestd_find_chronological_inconsistency(phylogeny_df) is None
    phylogeny_df = _weighted(phylogeny_df, weight_by)

    result = alifestd_pool_chronological_consistency_asexual(
        phylogeny_df, weight_by=weight_by, pin_by=pin_by
    )
    assert result.equals(phylogeny_df)

import itertools as it
import os

import numpy as np
import pandas as pd
import pytest
from tqdm import tqdm

from phyloframe.legacy import (
    alifestd_is_chronologically_ordered,
    alifestd_mark_root_id,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_calc_distance_matrix_asexual import (
    alifestd_calc_distance_matrix_asexual as alifestd_calc_distance_matrix_asexual_,
)
from phyloframe.legacy._alifestd_find_pair_distance_asexual import (
    alifestd_find_pair_distance_asexual,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_calc_distance_matrix_asexual = enforce_dtype_stability_pandas(
    alifestd_calc_distance_matrix_asexual_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def make_expected(
    phylogeny_df: pd.DataFrame, criterion: str = "origin_time"
) -> np.ndarray:
    n = len(phylogeny_df)
    result = np.full((n, n), np.nan, dtype=np.float64)
    if n == 0:
        return result

    phylogeny_df = alifestd_mark_root_id(phylogeny_df, mutate=True)

    for (i, id1), (j, id2) in tqdm(
        it.product(enumerate(phylogeny_df["id"]), repeat=2),
    ):
        assert i == id1 and j == id2
        if phylogeny_df["root_id"].iat[i] == phylogeny_df["root_id"].iat[j]:
            dist = alifestd_find_pair_distance_asexual(
                phylogeny_df, id1, id2, criterion=criterion, mutate=True
            )
            result[i, j] = dist if dist is not None else np.nan
        else:
            result[i, j] = np.nan

    return result


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pytest.param(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
            marks=pytest.mark.heavy,
        ),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize(
    "mutate",
    [True, False],
)
def test_big1(phylogeny_df: pd.DataFrame, mutate: bool):
    phylogeny_df = phylogeny_df.copy()
    assert alifestd_is_chronologically_ordered(phylogeny_df)
    phylogeny_df = alifestd_to_working_format(phylogeny_df)
    original_df = phylogeny_df.copy()

    expected = make_expected(phylogeny_df.copy())
    actual = alifestd_calc_distance_matrix_asexual(
        phylogeny_df,
        mutate=mutate,
        progress_wrap=tqdm,
    )

    np.testing.assert_array_equal(
        np.isnan(expected), np.isnan(actual), err_msg="NaN pattern mismatch"
    )
    np.testing.assert_allclose(np.nan_to_num(actual), np.nan_to_num(expected))
    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_simple1(mutate: bool):
    # Tree:  0 -> 1 -> 2, 0 -> 3
    # origin_times: 0=0, 1=10, 2=20, 3=15
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
            "origin_time": [0, 10, 20, 15],
        }
    )
    original_df = phylogeny_df.copy()

    # distance[i,j] = origin_time[i] + origin_time[j] - 2 * origin_time[mrca]
    # MRCA matrix:
    #   [0,0,0,0]
    #   [0,1,1,0]
    #   [0,1,2,0]
    #   [0,0,0,3]
    # distance[0,0]: (0+0 - 2*0) = 0
    # distance[0,1]: (0+10 - 2*0) = 10
    # distance[2,3]: mrca=0; (20+15 - 2*0) = 35
    # distance[1,2]: mrca=1; (10+20 - 2*10) = 10
    expected = np.array(
        [
            [0.0, 10.0, 20.0, 15.0],
            [10.0, 0.0, 10.0, 25.0],
            [20.0, 10.0, 0.0, 35.0],
            [15.0, 25.0, 35.0, 0.0],
        ],
        dtype=np.float64,
    )
    res = alifestd_calc_distance_matrix_asexual(phylogeny_df, mutate=mutate)
    np.testing.assert_allclose(res, expected)

    # ensure idempotency
    res = alifestd_calc_distance_matrix_asexual(phylogeny_df, mutate=mutate)
    np.testing.assert_allclose(res, expected)

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_simple2_ancestor_list(mutate: bool):
    # Tree: 0 -> 1 -> 2, 0 -> 3
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[1]", "[0]"],
            "origin_time": [0, 10, 20, 15],
        }
    )
    original_df = phylogeny_df.copy()

    expected = np.array(
        [
            [0.0, 10.0, 20.0, 15.0],
            [10.0, 0.0, 10.0, 25.0],
            [20.0, 10.0, 0.0, 35.0],
            [15.0, 25.0, 35.0, 0.0],
        ],
        dtype=np.float64,
    )
    res = alifestd_calc_distance_matrix_asexual(phylogeny_df, mutate=mutate)
    np.testing.assert_allclose(res, expected)

    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_custom_criterion():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
            "origin_time": [0, 10, 20, 15],
            "depth": [0.0, 1.0, 2.0, 1.0],
        }
    )
    # MRCA of 2 and 3 is 0; depth distance = (2.0 + 1.0 - 2*0.0) = 3.0
    res = alifestd_calc_distance_matrix_asexual(
        phylogeny_df, criterion="depth"
    )
    assert res[2, 3] == pytest.approx(3.0)
    assert res[3, 2] == pytest.approx(3.0)
    # Self-distances are 0
    for i in range(4):
        assert res[i, i] == pytest.approx(0.0)


def test_symmetric():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
            "origin_time": [0, 10, 20, 15],
        }
    )
    res = alifestd_calc_distance_matrix_asexual(phylogeny_df)
    np.testing.assert_allclose(res, res.T)


def test_self_distance_zero():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
            "origin_time": [0, 10, 20, 15],
        }
    )
    res = alifestd_calc_distance_matrix_asexual(phylogeny_df)
    for i in range(4):
        assert res[i, i] == pytest.approx(0.0)


@pytest.mark.parametrize("mutate", [True, False])
@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.DataFrame(
            {
                "id": [],
                "ancestor_list": [],
                "origin_time": [],
            }
        ),
        pd.DataFrame(
            {
                "id": [],
                "ancestor_id": [],
                "origin_time": [],
            }
        ),
        pd.DataFrame(
            {
                "id": [0],
                "ancestor_list": ["[None]"],
                "origin_time": [5.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "ancestor_list": ["[None]"],
                "origin_time": [5.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1],
                "ancestor_list": ["[None]", "[0]"],
                "origin_time": [0.0, 10.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 1, 2],
                "origin_time": [0.0, 5.0, 10.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
                "ancestor_list": ["[None]", "[0]", "[0]"],
                "origin_time": [0.0, 5.0, 10.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_list": ["[None]", "[None]", "[1]"],
                "origin_time": [0.0, 5.0, 10.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 2, 1],
                "origin_time": [0.0, 5.0, 3.0, 10.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 0, 0],
                "origin_time": [0.0, 5.0, 8.0, 12.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_list": ["[None]", "[0]", "[0]", "[0]"],
                "origin_time": [0.0, 5.0, 8.0, 12.0],
            }
        ),
    ],
)
def test_edge_cases(phylogeny_df: pd.DataFrame, mutate: bool):
    phylogeny_df = phylogeny_df.copy()
    original_df = phylogeny_df.copy()

    res = alifestd_calc_distance_matrix_asexual(phylogeny_df, mutate=mutate)
    expected = make_expected(phylogeny_df.copy())

    np.testing.assert_array_equal(
        np.isnan(res), np.isnan(expected), err_msg="NaN pattern mismatch"
    )
    np.testing.assert_allclose(np.nan_to_num(res), np.nan_to_num(expected))

    # ensure idempotency
    res2 = alifestd_calc_distance_matrix_asexual(phylogeny_df, mutate=mutate)
    np.testing.assert_array_equal(
        np.isnan(res2), np.isnan(expected), err_msg="NaN pattern mismatch"
    )
    np.testing.assert_allclose(np.nan_to_num(res2), np.nan_to_num(expected))

    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_disjoint_trees_nan():
    """Pairs from different trees have NaN distance."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 2, 2],
            "origin_time": [0.0, 5.0, 10.0, 15.0],
        }
    )
    res = alifestd_calc_distance_matrix_asexual(phylogeny_df)
    # Cross-tree pairs should be NaN
    assert np.isnan(res[0, 2])
    assert np.isnan(res[0, 3])
    assert np.isnan(res[1, 2])
    assert np.isnan(res[1, 3])
    # Same-tree pairs should be finite
    assert np.isfinite(res[0, 1])
    assert np.isfinite(res[2, 3])
    assert res[0, 1] == pytest.approx(5.0)
    assert res[2, 3] == pytest.approx(5.0)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
        ),
    ],
)
def test_fuzz_matches_pairwise(phylogeny_df: pd.DataFrame):
    """Matrix entries should match pairwise distance function."""
    res = alifestd_calc_distance_matrix_asexual(phylogeny_df)
    ids = phylogeny_df["id"].tolist()
    for i in ids[:5]:
        for j in ids[:5]:
            expected = alifestd_find_pair_distance_asexual(
                phylogeny_df, i, j, mutate=False
            )
            if expected is None:
                assert np.isnan(res[i, j])
            else:
                assert res[i, j] == pytest.approx(expected)

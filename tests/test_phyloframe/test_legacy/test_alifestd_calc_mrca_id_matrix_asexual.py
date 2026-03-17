import itertools as it
import os
import typing

import numpy as np
import pandas as pd
import pytest
from tqdm import tqdm

from phyloframe.legacy import (
    alifestd_calc_mrca_id_matrix_asexual as alifestd_calc_mrca_id_matrix_asexual_,
)
from phyloframe.legacy import (
    alifestd_find_mrca_id_asexual,
    alifestd_is_chronologically_ordered,
    alifestd_mark_csr_children_asexual,
    alifestd_mark_csr_offsets_asexual,
    alifestd_mark_first_child_id_asexual,
    alifestd_mark_next_sibling_id_asexual,
    alifestd_mark_num_children_asexual,
    alifestd_mark_root_id,
    alifestd_to_working_format,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_calc_mrca_id_matrix_asexual = enforce_dtype_stability_pandas(
    alifestd_calc_mrca_id_matrix_asexual_
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


def make_expected(phylogeny_df: pd.DataFrame) -> np.ndarray:
    n = len(phylogeny_df)
    result = np.zeros((n, n), dtype=np.int64)
    if n == 0:
        return result

    phylogeny_df = alifestd_mark_root_id(phylogeny_df, mutate=True)

    for (i, id1), (j, id2) in tqdm(
        it.product(enumerate(phylogeny_df["id"]), repeat=2),
    ):
        assert i == id1 and j == id2
        if phylogeny_df["root_id"].iat[i] == phylogeny_df["root_id"].iat[j]:
            result[i, j] = alifestd_find_mrca_id_asexual(
                phylogeny_df, (id1, id2), mutate=True
            )
        else:
            result[i, j] = -1

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
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(_prep_noop, id="no-precompute"),
        pytest.param(_prep_csr, id="csr"),
        pytest.param(_prep_linked_list, id="linked-list"),
    ],
)
def test_big1(
    phylogeny_df: pd.DataFrame, mutate: bool, apply: typing.Callable
):
    phylogeny_df = phylogeny_df.copy()
    assert alifestd_is_chronologically_ordered(phylogeny_df)
    phylogeny_df = alifestd_to_working_format(phylogeny_df)
    original_df = phylogeny_df.copy()

    expected = make_expected(phylogeny_df.copy())
    actual = alifestd_calc_mrca_id_matrix_asexual(
        apply(phylogeny_df.copy()),
        mutate=mutate,
        progress_wrap=tqdm,
    )

    np.testing.assert_array_equal(expected, actual)
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
def test_simple1(mutate: bool, apply: typing.Callable):
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[1]", "[0]"],
        }
    )
    original_df = phylogeny_df.copy()

    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 2, 0],
            [0, 0, 0, 3],
        ],
        dtype=np.int64,
    )
    res = alifestd_calc_mrca_id_matrix_asexual(
        apply(phylogeny_df.copy()), mutate=mutate
    )
    np.testing.assert_array_equal(res, expected)

    # ensure idempotency
    res = alifestd_calc_mrca_id_matrix_asexual(
        apply(phylogeny_df.copy()), mutate=mutate
    )
    np.testing.assert_array_equal(res, expected)

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.DataFrame(
            {
                "id": [],
                "ancestor_list": [],
            }
        ),
        pd.DataFrame(
            {
                "id": [],
                "ancestor_id": [],
            }
        ),
        pd.DataFrame(
            {
                "id": [0],
                "ancestor_list": ["[None]"],
            }
        ),
        pd.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "ancestor_list": ["[None]"],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1],
                "ancestor_list": ["[None]", "[0]"],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 1, 2],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
                "ancestor_list": ["[None]", "[0]", "[0]"],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_list": ["[None]", "[None]", "[1]"],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 2, 1],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 0, 0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_list": ["[None]", "[0]", "[0]", "[0]"],
            }
        ),
    ],
)
def test_edge_cases(phylogeny_df: pd.DataFrame, mutate: bool):
    phylogeny_df = phylogeny_df.copy()
    original_df = phylogeny_df.copy()

    res = alifestd_calc_mrca_id_matrix_asexual(phylogeny_df, mutate=mutate)
    expected = make_expected(phylogeny_df.copy())
    np.testing.assert_array_equal(res, expected)

    # ensure idempotency
    res = alifestd_calc_mrca_id_matrix_asexual(phylogeny_df, mutate=mutate)
    np.testing.assert_array_equal(res, expected)

    if not mutate:
        assert original_df.equals(phylogeny_df)

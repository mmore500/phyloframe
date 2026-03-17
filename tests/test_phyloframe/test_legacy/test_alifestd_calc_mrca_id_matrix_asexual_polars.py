import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tqdm import tqdm

from phyloframe.legacy import (
    alifestd_calc_mrca_id_matrix_asexual,
    alifestd_is_chronologically_ordered,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_calc_mrca_id_matrix_asexual_polars import (
    alifestd_calc_mrca_id_matrix_asexual_polars as alifestd_calc_mrca_id_matrix_asexual_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_calc_mrca_id_matrix_asexual_polars = enforce_dtype_stability_polars(
    alifestd_calc_mrca_id_matrix_asexual_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_big1_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify polars result matches pandas MRCA matrix on real datasets."""
    phylogeny_df = phylogeny_df.copy()
    assert alifestd_is_chronologically_ordered(phylogeny_df)
    phylogeny_df = alifestd_to_working_format(phylogeny_df)
    df_pl = pl.from_pandas(phylogeny_df)

    expected = alifestd_calc_mrca_id_matrix_asexual(phylogeny_df)
    actual = alifestd_calc_mrca_id_matrix_asexual_polars(
        apply(df_pl), progress_wrap=tqdm
    )

    np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple1(apply: typing.Callable):
    """Test a simple 4-node tree matches known expected matrix."""
    phylogeny_df_pd = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[1]", "[0]"],
        }
    )
    phylogeny_df_pd = alifestd_to_working_format(phylogeny_df_pd)
    df_pl = pl.from_pandas(phylogeny_df_pd)

    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 2, 0],
            [0, 0, 0, 3],
        ],
        dtype=np.int64,
    )
    actual = alifestd_calc_mrca_id_matrix_asexual_polars(apply(df_pl))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
@pytest.mark.parametrize(
    "phylogeny_df_pd",
    [
        pd.DataFrame(
            {
                "id": pd.array([], dtype="int64"),
                "ancestor_list": pd.array([], dtype="object"),
            }
        ),
        pd.DataFrame(
            {
                "id": pd.array([], dtype="int64"),
                "ancestor_id": pd.array([], dtype="int64"),
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
def test_edge_cases(phylogeny_df_pd: pd.DataFrame, apply: typing.Callable):
    """Test edge cases match the pandas matrix implementation."""
    phylogeny_df_pd = phylogeny_df_pd.copy()

    expected = alifestd_calc_mrca_id_matrix_asexual(phylogeny_df_pd)

    phylogeny_df_pd_wf = alifestd_to_working_format(phylogeny_df_pd.copy())
    df_pl = pl.from_pandas(phylogeny_df_pd_wf)

    actual = alifestd_calc_mrca_id_matrix_asexual_polars(apply(df_pl))
    np.testing.assert_array_equal(actual, expected)

    # idempotency check
    actual2 = alifestd_calc_mrca_id_matrix_asexual_polars(apply(df_pl))
    np.testing.assert_array_equal(actual2, expected)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_does_not_mutate_input(apply: typing.Callable):
    """Verify the input dataframe is not mutated."""
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
        }
    )
    original_columns = df_pl.columns[:]
    original_data = df_pl.clone()

    alifestd_calc_mrca_id_matrix_asexual_polars(apply(df_pl))

    assert df_pl.columns == original_columns
    assert df_pl.equals(original_data)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_single_node(apply: typing.Callable):
    """Test single-node phylogeny: MRCA with self is self."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        )
    )
    result = alifestd_calc_mrca_id_matrix_asexual_polars(df_pl)
    np.testing.assert_array_equal(result, np.array([[0]], dtype=np.int64))


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_multiple_roots(apply: typing.Callable):
    """Test phylogeny with multiple independent roots."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 1, 2],
            }
        )
    )
    result = alifestd_calc_mrca_id_matrix_asexual_polars(df_pl)
    expected = np.array(
        [
            [0, -1, -1],
            [-1, 1, -1],
            [-1, -1, 2],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_uint64_ancestor_id(apply: typing.Callable):
    """Regression: UInt64 ancestor_id should be handled correctly."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": pl.Series([0, 1, 2, 3], dtype=pl.UInt64),
                "ancestor_id": pl.Series([0, 0, 1, 0], dtype=pl.UInt64),
            }
        )
    )
    result = alifestd_calc_mrca_id_matrix_asexual_polars(df_pl)
    assert result.dtype == np.int64
    assert result.shape == (4, 4)
    # Symmetry check
    np.testing.assert_array_equal(result, result.T)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_with_ancestor_list_col(apply: typing.Callable):
    """Test that ancestor_list is correctly converted to ancestor_id."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_list": ["[None]", "[0]", "[0]", "[1]"],
            }
        )
    )
    result = alifestd_calc_mrca_id_matrix_asexual_polars(df_pl)
    # Tree: 0->1->3, 0->2.
    # MRCA matrix:
    #   0: [0,0,0,0]
    #   1: [0,1,0,1]
    #   2: [0,0,2,0]
    #   3: [0,1,0,3]
    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 2, 0],
            [0, 1, 0, 3],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(result, expected)

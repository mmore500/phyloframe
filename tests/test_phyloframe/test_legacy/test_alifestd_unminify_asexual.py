import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_has_contiguous_ids,
    alifestd_is_topologically_sorted,
    alifestd_try_add_ancestor_list_col,
    alifestd_unminify_asexual,
    alifestd_validate,
)


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "ancestor_offset": [0, 1, 1, 3, 1],
            "name": ["a", "b", "d", "c", "e"],
        }
    )
    original = phylogeny_df.copy()

    result = alifestd_unminify_asexual(phylogeny_df, mutate=mutate)

    if not mutate:
        assert original.equals(phylogeny_df)
    assert result.columns.tolist() == ["id", "ancestor_id", "name"]
    assert result["id"].tolist() == [0, 1, 2, 3, 4]
    assert result["ancestor_id"].tolist() == [0, 0, 1, 0, 3]
    assert result["name"].tolist() == ["a", "b", "d", "c", "e"]
    assert alifestd_has_contiguous_ids(result)
    assert alifestd_is_topologically_sorted(result)

    result = alifestd_try_add_ancestor_list_col(result)
    assert result["ancestor_list"].tolist() == [
        "[none]",
        "[0]",
        "[1]",
        "[0]",
        "[3]",
    ]
    assert alifestd_validate(result)


def test_non_default_index():
    phylogeny_df = pd.DataFrame(
        {"ancestor_offset": [0, 1, 2]},
        index=[7, 3, 5],
    )

    result = alifestd_unminify_asexual(phylogeny_df)

    assert result["id"].tolist() == [0, 1, 2]
    assert result["ancestor_id"].tolist() == [0, 0, 0]


@pytest.mark.parametrize("offset_dtype", [np.int8, np.int64, np.uint32])
def test_dtype_default(offset_dtype: type):
    # offsets fit in int8, but ids do not
    phylogeny_df = pd.DataFrame(
        {"ancestor_offset": np.ones(300, dtype=offset_dtype)},
    )
    phylogeny_df.loc[0, "ancestor_offset"] = 0

    result = alifestd_unminify_asexual(phylogeny_df)

    assert result["id"].dtype == np.int64
    assert result["ancestor_id"].dtype == np.int64
    assert result["id"].tolist() == list(range(300))
    assert result["ancestor_id"].tolist() == [0, *range(299)]


@pytest.mark.parametrize("dtype_id", [np.int32, np.uint64])
def test_dtype_explicit(dtype_id: type):
    phylogeny_df = pd.DataFrame(
        {"ancestor_offset": np.array([0, 1, 1], dtype=np.int8)},
    )

    result = alifestd_unminify_asexual(phylogeny_df, dtype_id=dtype_id)

    assert result["id"].dtype == dtype_id
    assert result["ancestor_id"].dtype == dtype_id
    assert result["ancestor_id"].tolist() == [0, 0, 1]


@pytest.mark.parametrize("column", ["id", "ancestor_id"])
def test_warns_existing_id_columns(column: str):
    phylogeny_df = pd.DataFrame(
        {"ancestor_offset": [0, 1, 1], column: [9, 9, 9]},
    )

    with pytest.warns(UserWarning, match=column):
        result = alifestd_unminify_asexual(phylogeny_df)

    assert result.columns.tolist() == ["id", "ancestor_id"]
    assert result["id"].tolist() == [0, 1, 2]
    assert result["ancestor_id"].tolist() == [0, 0, 1]


def test_empty():
    phylogeny_df = pd.DataFrame(
        {"ancestor_offset": pd.Series([], dtype=np.int64)},
    )

    result = alifestd_unminify_asexual(phylogeny_df)

    assert len(result) == 0
    assert result.columns.tolist() == ["id", "ancestor_id"]

import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_consolidate_roots_asexual as alifestd_consolidate_roots_asexual_,
)
from phyloframe.legacy import (
    alifestd_try_add_ancestor_list_col,
    alifestd_validate,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_consolidate_roots_asexual = enforce_dtype_stability_pandas(
    alifestd_consolidate_roots_asexual_
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _consolidate(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return alifestd_consolidate_roots_asexual(
        df, ignore_topological_sensitivity=True, **kwargs
    )


@pytest.mark.parametrize("mutate", [True, False])
def test_empty(mutate: bool):
    df = pd.DataFrame({"id": [], "ancestor_id": []}, dtype=int)
    assert _consolidate(df, mutate=mutate).empty


@pytest.mark.parametrize("mutate", [True, False])
def test_single_root_unchanged(mutate: bool):
    df = pd.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 0, 1]})
    original_df = df.copy()
    result = _consolidate(df, mutate=mutate)
    pd.testing.assert_frame_equal(result, original_df)
    if not mutate:
        assert original_df.equals(df)


@pytest.mark.parametrize("mutate", [True, False])
def test_oldest_by_lowest_origin_time(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 1, 0, 1, 1, 5, 6],
            "origin_time": [3.0, 1.0, 4.0, 2.0, 5.0, 0.5, 9.0],
        },
    )
    original_df = df.copy()
    result = _consolidate(df, mutate=mutate)
    # root 5 is oldest; roots 0, 1, and childless 6 are deleted
    assert result["id"].tolist() == [2, 3, 4, 5]
    assert result["ancestor_id"].tolist() == [5, 5, 5, 5]
    if not mutate:
        assert original_df.equals(df)


def test_oldest_by_lowest_id():
    df = pd.DataFrame({"id": [0, 1, 2, 3, 4], "ancestor_id": [0, 0, 2, 2, 3]})
    result = _consolidate(df)
    assert result["id"].tolist() == [0, 1, 3, 4]
    assert result["ancestor_id"].tolist() == [0, 0, 0, 3]


@pytest.mark.parametrize("root_ancestor_token", ["none", ""])
def test_ancestor_list_updated(root_ancestor_token: str):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[none]", "[none]", "[1]", "[2]"],
        },
    )
    result = _consolidate(df, root_ancestor_token=root_ancestor_token)
    assert result["id"].tolist() == [0, 2, 3]
    assert result["ancestor_list"].tolist() == [
        f"[{root_ancestor_token}]",
        "[0]",
        "[2]",
    ]
    assert alifestd_validate(result)


def test_asset_tworoots():
    df = pd.read_csv(
        f"{assets}/example-standard-toy-asexual-phylogeny-tworoots.csv"
    )
    result = _consolidate(df)
    assert alifestd_validate(alifestd_try_add_ancestor_list_col(result))
    assert (result["id"] == result["ancestor_id"]).sum() == 1
    assert len(result) == len(df) - 1


def test_sexual_raises():
    df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_list": ["[]", "[]", "[0,1]"]}
    )
    with pytest.raises(ValueError, match="asexual"):
        _consolidate(df)


@pytest.mark.parametrize("column", ["num_children", "is_leaf"])
def test_warns_topological_sensitivity(column: str):
    # root 0 is a leaf until consolidation gives it child 2
    df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 1, 1], column: [0, 1, 0]},
    )
    with pytest.warns(UserWarning, match=column):
        alifestd_consolidate_roots_asexual_(df)


@pytest.mark.parametrize(
    "origin_time", [None, float("nan")], ids=["null", "nan"]
)
def test_null_or_nan_root_origin_time_raises(origin_time):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 1, 0, 1],
            "origin_time": [origin_time, 0.0, 1.0, 2.0],
        },
    )
    with pytest.raises(NotImplementedError, match="origin_time"):
        _consolidate(df)

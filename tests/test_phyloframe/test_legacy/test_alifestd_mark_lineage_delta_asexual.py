import os
import warnings

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
    alifestd_mark_lineage_cumsum_asexual,
    alifestd_mark_lineage_delta_asexual,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _make_tree() -> pd.DataFrame:
    #       0
    #     1   4
    #    2 3
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 1, 1, 0],
            "v": [5, 7, 4, 7, 9],
        },
    )


def test_empty():
    mt = alifestd_make_empty()
    mt["origin_time"] = pd.Series(dtype=float)
    res = alifestd_mark_lineage_delta_asexual(mt, "origin_time")
    assert "lineage_delta" in res.columns
    assert len(res) == 0


def test_deltas():
    res = alifestd_mark_lineage_delta_asexual(_make_tree(), "v")
    # a root's delta is zero by default
    assert res["lineage_delta"].tolist() == [0, 2, -3, 0, 4]
    assert res["lineage_delta"].dtype == np.int64


@pytest.mark.parametrize("basal_value, root_delta", [(0, 5), (3, 2), (5, 0)])
def test_basal_value(basal_value: int, root_delta: int):
    res = alifestd_mark_lineage_delta_asexual(
        _make_tree(), "v", basal_value=basal_value
    )
    assert res["lineage_delta"].tolist() == [root_delta, 2, -3, 0, 4]
    assert res["lineage_delta"].dtype == np.int64


def test_binary_trait_transitions():
    df = _make_tree().assign(trait=[False, True, True, False, False])
    res = alifestd_mark_lineage_delta_asexual(
        df, "trait", mark_as="transitions"
    )
    assert res["transitions"].tolist() == [0, 1, 0, -1, 0]
    assert res["transitions"].dtype == np.int8


def test_basal_value_gives_root_gain():
    df = _make_tree().assign(trait=[True, True, True, True, False])
    res = alifestd_mark_lineage_delta_asexual(df, "trait")
    assert res["lineage_delta"].tolist() == [0, 0, 0, 0, -1]
    res = alifestd_mark_lineage_delta_asexual(df, "trait", basal_value=False)
    assert res["lineage_delta"].tolist() == [1, 0, 0, 0, -1]
    assert res["lineage_delta"].dtype == np.int8


def test_keeps_input_dtype():
    df = _make_tree().assign(v=np.array([5, 7, 4, 7, 9], dtype=np.uint8))
    with pytest.warns(UserWarning, match="wrap around"):
        res = alifestd_mark_lineage_delta_asexual(df, "v")
    # negative deltas of unsigned values wrap around
    assert res["lineage_delta"].tolist() == [0, 2, 253, 0, 4]
    assert res["lineage_delta"].dtype == np.uint8


def test_unsigned_without_negative_deltas_does_not_warn():
    df = _make_tree().assign(v=np.array([5, 7, 8, 7, 9], dtype=np.uint8))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = alifestd_mark_lineage_delta_asexual(df, "v")
    assert res["lineage_delta"].tolist() == [0, 2, 1, 0, 4]
    # a basal value above the root's value gives a negative root delta
    with pytest.warns(UserWarning, match="wrap around"):
        alifestd_mark_lineage_delta_asexual(df, "v", basal_value=6)


def test_nan_propagates():
    df = _make_tree().assign(v=[1.0, np.nan, 2.0, 3.0, 4.0])
    res = alifestd_mark_lineage_delta_asexual(df, "v")
    assert np.isnan(res["lineage_delta"].to_numpy()[[1, 2, 3]]).all()
    assert res["lineage_delta"].to_numpy()[[0, 4]].tolist() == [0.0, 3.0]


def test_inverts_cumsum():
    df = alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(
            pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv")
        ),
    )
    rng = np.random.default_rng(1)
    df["v"] = rng.integers(-100, 100, len(df))
    deltas = alifestd_mark_lineage_delta_asexual(
        df, "v", basal_value=0, mark_as="d"
    )
    back = alifestd_mark_lineage_cumsum_asexual(deltas, "d", mark_as="back")
    assert (back["back"] == df["v"]).all()
    sums = alifestd_mark_lineage_cumsum_asexual(df, "v", mark_as="s")
    forth = alifestd_mark_lineage_delta_asexual(
        sums, "s", basal_value=0, mark_as="forth"
    )
    assert (forth["forth"] == df["v"]).all()


@pytest.mark.parametrize("mutate", [False, True])
def test_mutate(mutate: bool):
    df = _make_tree()
    original = df.copy()
    res = alifestd_mark_lineage_delta_asexual(df, "v", mutate=mutate)
    assert "lineage_delta" in res.columns
    if not mutate:
        pd.testing.assert_frame_equal(df, original)


def test_missing_values_column_raises():
    with pytest.raises(ValueError):
        alifestd_mark_lineage_delta_asexual(_make_tree(), "absent")


def test_non_contiguous_raises():
    df = _make_tree().assign(id=[0, 1, 2, 3, 5], ancestor_id=[0, 0, 1, 1, 0])
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_delta_asexual(df, "v")


def test_ancestor_list_column():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[none]", "[0]", "[1]"],
            "v": [1, 3, 2],
        },
    )
    res = alifestd_mark_lineage_delta_asexual(df, "v")
    assert res["lineage_delta"].tolist() == [0, 2, -1]

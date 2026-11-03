import os

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_mark_lineage_cummax_asexual,
    alifestd_mark_lineage_ffill_asexual,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _make_df() -> pd.DataFrame:
    """Tree:
    0 (root, missing)
    +-- 1 (5)
    |   +-- 3 (missing)
    |   |   +-- 5 (missing)
    |   +-- 4 (7)
    +-- 2 (missing)
    6 (root, 2)
    +-- 7 (missing)
    """
    return pd.DataFrame(
        {
            "id": [*range(8)],
            "ancestor_id": [0, 0, 0, 1, 1, 3, 6, 6],
            "v": [np.nan, 5.0, np.nan, np.nan, 7.0, np.nan, 2.0, np.nan],
        },
    )


def _reference_ffill(ancestor_ids: np.ndarray, values: pd.Series) -> list:
    result = []
    for idx in range(len(ancestor_ids)):
        node = idx
        while pd.isna(values.iloc[node]) and ancestor_ids[node] != node:
            node = ancestor_ids[node]
        result.append(values.iloc[node])
    return result


@pytest.mark.parametrize("mutate", [True, False])
def test_hand_built_tree(mutate: bool):
    df = _make_df()
    original_df = df.copy()
    res = alifestd_mark_lineage_ffill_asexual(df, "v", mutate=mutate)
    np.testing.assert_array_equal(
        res["lineage_ffill"],
        [np.nan, 5.0, np.nan, 5.0, 7.0, 5.0, 2.0, 2.0],
    )
    assert res["v"].equals(original_df["v"])
    if not mutate:
        assert df.equals(original_df)


@pytest.mark.parametrize(
    "values, expected",
    [
        (
            pd.array([None, 5, None, None, 7, None, 2, None], dtype="Int64"),
            pd.array([None, 5, None, 5, 7, 5, 2, 2], dtype="Int64"),
        ),
        (
            pd.Series([None, "a", None, None, "b", None, "c", None]),
            pd.Series([None, "a", None, "a", "b", "a", "c", "c"]),
        ),
        (
            pd.to_datetime(
                [None, "2001", None, None, "2002", None, "2003", None],
            ),
            pd.to_datetime(
                [None, "2001", None, "2001", "2002", "2001", "2003", "2003"],
            ),
        ),
    ],
)
def test_dtypes(values, expected):
    df = _make_df().assign(v=values)
    res = alifestd_mark_lineage_ffill_asexual(df, "v")
    assert res["lineage_ffill"].dtype == df["v"].dtype
    pd.testing.assert_series_equal(
        res["lineage_ffill"],
        pd.Series(expected, name="lineage_ffill"),
        check_dtype=False,
    )


def test_no_missing_unchanged():
    df = _make_df().fillna(1.0)
    res = alifestd_mark_lineage_ffill_asexual(df, "v")
    assert res["lineage_ffill"].tolist() == df["v"].tolist()


def test_mark_as():
    res = alifestd_mark_lineage_ffill_asexual(
        _make_df(), "v", mark_as="blueberry"
    )
    assert "blueberry" in res.columns
    assert "lineage_ffill" not in res.columns


def test_empty():
    df = pd.DataFrame({"id": [], "ancestor_id": [], "v": []}, dtype=float)
    res = alifestd_mark_lineage_ffill_asexual(df, "v")
    assert "lineage_ffill" in res.columns
    assert len(res) == 0


def test_missing_column():
    with pytest.raises(ValueError):
        alifestd_mark_lineage_ffill_asexual(_make_df(), "nope")


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(  # non-contiguous
            {"id": [0, 2], "ancestor_id": [0, 0], "v": [1.0, np.nan]},
        ),
        pd.DataFrame(  # unsorted
            {"id": [0, 1, 2], "ancestor_id": [0, 2, 0], "v": [1.0] * 3},
        ),
        pd.DataFrame(  # sexual
            {
                "id": [0, 1, 2],
                "ancestor_list": ["[none]", "[0]", "[0,1]"],
                "v": [1.0] * 3,
            },
        ),
    ],
)
def test_unsupported(df: pd.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_ffill_asexual(df, "v")


@pytest.mark.parametrize("seed", range(5))
def test_matches_reference_random(seed: int):
    rng = np.random.default_rng(seed)
    num_nodes = 200
    df = pd.DataFrame(
        {
            "id": np.arange(num_nodes),
            "ancestor_id": [
                idx
                if idx == 0 or rng.random() < 0.05
                else rng.integers(0, idx)
                for idx in range(num_nodes)
            ],
            "v": np.where(
                rng.random(num_nodes) < 0.6, np.nan, rng.random(num_nodes)
            ),
        },
    )
    res = alifestd_mark_lineage_ffill_asexual(df, "v")
    np.testing.assert_array_equal(
        res["lineage_ffill"],
        _reference_ffill(df["ancestor_id"].to_numpy(), df["v"]),
    )


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv"),
        pd.read_csv(f"{assets}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize("seed", range(3))
def test_matches_cummax_for_increasing_values(
    phylogeny_df: pd.DataFrame, seed: int
):
    # e.g., ranks increasing down lineages, as filled after splicing
    df = alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(phylogeny_df)
    )
    rng = np.random.default_rng(seed)
    df["v"] = np.where(
        rng.random(len(df)) < 0.5, np.nan, df["origin_time"].astype(float)
    )
    res = alifestd_mark_lineage_ffill_asexual(df, "v")
    res = alifestd_mark_lineage_cummax_asexual(res, "v")
    has_value = res["lineage_ffill"].notna()
    assert has_value.any()
    np.testing.assert_array_equal(
        res.loc[has_value, "lineage_ffill"],
        res.loc[has_value, "lineage_cummax"],
    )

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_find_pair_distance_asexual,
    alifestd_make_empty,
)
from phyloframe.legacy import (
    alifestd_mask_closest_relatives_asexual as alifestd_mask_closest_relatives_asexual_,
)
from phyloframe.legacy import (
    alifestd_prune_extinct_lineages_asexual,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_mask_closest_relatives_asexual = enforce_dtype_stability_pandas(
    alifestd_mask_closest_relatives_asexual_
)


def _make_tree() -> pd.DataFrame:
    #        0
    #     1     2
    #    3 4   5 6
    #   7 8
    return pd.DataFrame(
        {
            "id": range(9),
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2, 3, 3],
            "origin_time": [0, 1, 1, 2, 3, 3, 9, 9, 4],
        },
    )


def _mask(phylogeny_df: pd.DataFrame, ids: list) -> np.ndarray:
    return phylogeny_df["id"].isin(ids).to_numpy(copy=True)


def test_empty():
    phylogeny_df = alifestd_make_empty()
    phylogeny_df["origin_time"] = pd.Series([], dtype=float)
    res = alifestd_mask_closest_relatives_asexual(
        phylogeny_df,
        candidate_mask=np.array([], dtype=bool),
        extant_mask=np.array([], dtype=bool),
    )
    assert "alifestd_mask_closest_relatives_asexual" in res.columns
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [False, True])
def test_simple(mutate: bool):
    phylogeny_df = _make_tree()
    original_df = phylogeny_df.copy()
    res = alifestd_mask_closest_relatives_asexual(
        phylogeny_df,
        mutate=mutate,
        candidate_mask=_mask(phylogeny_df, [4, 5, 8]),
        extant_mask=_mask(phylogeny_df, [6, 7]),
    )
    # leaf 7 is closest to its sibling 8, and leaf 6 to its sibling 5; leaf
    # 4 is further from leaf 7 than leaf 8 is, so is not marked
    assert np.flatnonzero(
        res["alifestd_mask_closest_relatives_asexual"]
    ).tolist() == [5, 8]
    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_mark_as():
    phylogeny_df = _make_tree()
    res = alifestd_mask_closest_relatives_asexual(
        phylogeny_df,
        candidate_mask=_mask(phylogeny_df, [4, 5, 8]),
        extant_mask=_mask(phylogeny_df, [6]),
        mark_as="m",
    )
    assert np.flatnonzero(res["m"]).tolist() == [5]
    assert "alifestd_mask_closest_relatives_asexual" not in res.columns


def test_closer_branch_beats_later_mrca():
    phylogeny_df = _make_tree()
    # leaf 7 shares a later mrca with leaf 8 than with leaf 4, but leaf 8's
    # long branch puts it further away
    phylogeny_df.loc[8, "origin_time"] = 20
    res = alifestd_mask_closest_relatives_asexual(
        phylogeny_df,
        candidate_mask=_mask(phylogeny_df, [4, 8]),
        extant_mask=_mask(phylogeny_df, [7]),
        mark_as="m",
    )
    assert np.flatnonzero(res["m"]).tolist() == [4]


def test_criterion():
    phylogeny_df = _make_tree()
    phylogeny_df["depth"] = [0, 1, 1, 2, 2, 2, 2, 3, 3]
    phylogeny_df.loc[8, "origin_time"] = 20
    res = alifestd_mask_closest_relatives_asexual(
        phylogeny_df,
        candidate_mask=_mask(phylogeny_df, [4, 8]),
        criterion="depth",
        extant_mask=_mask(phylogeny_df, [7]),
        mark_as="m",
    )
    assert np.flatnonzero(res["m"]).tolist() == [8]


def test_ties_all_marked():
    phylogeny_df = _make_tree()
    phylogeny_df["origin_time"] = [0, 1, 1, 2, 2, 2, 2, 3, 3]
    res = alifestd_mask_closest_relatives_asexual(
        phylogeny_df,
        candidate_mask=_mask(phylogeny_df, [4, 7, 8]),
        extant_mask=_mask(phylogeny_df, [5]),
        mark_as="m",
    )
    # leaf 4 is closer than tied leaves 7 and 8
    assert np.flatnonzero(res["m"]).tolist() == [4]
    res = alifestd_mask_closest_relatives_asexual(
        phylogeny_df,
        candidate_mask=_mask(phylogeny_df, [7, 8]),
        extant_mask=_mask(phylogeny_df, [5]),
        mark_as="m",
    )
    assert np.flatnonzero(res["m"]).tolist() == [7, 8]


def test_inner_nodes():
    phylogeny_df = _make_tree()
    # candidate ancestors (1) and descendants (7) of extant node 3
    res = alifestd_mask_closest_relatives_asexual(
        phylogeny_df,
        candidate_mask=_mask(phylogeny_df, [0, 1, 7]),
        extant_mask=_mask(phylogeny_df, [3]),
        mark_as="m",
    )
    assert np.flatnonzero(res["m"]).tolist() == [1]


def test_extant_candidate_is_own_closest():
    phylogeny_df = _make_tree()
    res = alifestd_mask_closest_relatives_asexual(
        phylogeny_df,
        candidate_mask=_mask(phylogeny_df, [7, 8]),
        extant_mask=_mask(phylogeny_df, [7]),
        mark_as="m",
    )
    assert np.flatnonzero(res["m"]).tolist() == [7]


def test_no_extant_or_no_candidates():
    phylogeny_df = _make_tree()
    for candidate_ids, extant_ids in ([], [6, 7]), ([4, 5, 8], []):
        res = alifestd_mask_closest_relatives_asexual(
            phylogeny_df,
            candidate_mask=_mask(phylogeny_df, candidate_ids),
            extant_mask=_mask(phylogeny_df, extant_ids),
            mark_as="m",
        )
        assert not res["m"].any()


def test_multiple_roots():
    phylogeny_df = pd.DataFrame(
        {
            "id": range(6),
            "ancestor_id": [0, 1, 0, 0, 1, 1],
            "origin_time": [0, 0, 1, 2, 1, 2],
        },
    )
    # extant leaf 5 has no candidate in its tree
    res = alifestd_mask_closest_relatives_asexual(
        phylogeny_df,
        candidate_mask=_mask(phylogeny_df, [2, 3]),
        extant_mask=_mask(phylogeny_df, [5]),
        mark_as="m",
    )
    assert not res["m"].any()
    res = alifestd_mask_closest_relatives_asexual(
        phylogeny_df,
        candidate_mask=_mask(phylogeny_df, [2, 4]),
        extant_mask=_mask(phylogeny_df, [3, 5]),
        mark_as="m",
    )
    assert np.flatnonzero(res["m"]).tolist() == [2, 4]


@pytest.mark.parametrize("seed", range(5))
def test_matches_pair_distances(seed: int):
    rng = np.random.default_rng(seed)
    n = 40
    ancestor_ids = np.concatenate([[0], rng.integers(0, np.arange(1, n))])
    ancestor_ids[n // 2] = n // 2  # second root
    origin_times = np.zeros(n)
    for id_, ancestor_id in enumerate(ancestor_ids):  # integers, so ties
        origin_times[id_] = origin_times[ancestor_id] + rng.integers(3)
    phylogeny_df = pd.DataFrame(
        {
            "id": range(n),
            "ancestor_list": [
                f"[{'none' if id_ == ancestor_id else ancestor_id}]"
                for id_, ancestor_id in enumerate(ancestor_ids)
            ],
            "origin_time": origin_times,
        },
    )
    extant_mask = rng.random(n) < 0.3
    candidate_mask = rng.random(n) < 0.3

    expected = np.zeros(n, dtype=bool)
    for extant_id in np.flatnonzero(extant_mask):
        distances = {
            candidate_id: alifestd_find_pair_distance_asexual(
                phylogeny_df, int(extant_id), int(candidate_id)
            )
            for candidate_id in np.flatnonzero(candidate_mask)
        }
        distances = {k: v for k, v in distances.items() if v is not None}
        for candidate_id, distance in distances.items():
            expected[candidate_id] |= distance == min(distances.values())

    res = alifestd_mask_closest_relatives_asexual(
        phylogeny_df,
        candidate_mask=candidate_mask,
        extant_mask=extant_mask,
        mark_as="m",
    )
    assert res["m"].tolist() == expected.tolist()
    assert expected.any()


def test_prune_extinct_lineages():
    phylogeny_df = _make_tree()
    res = alifestd_mask_closest_relatives_asexual(
        phylogeny_df,
        candidate_mask=_mask(phylogeny_df, [4, 5, 8]),
        extant_mask=_mask(phylogeny_df, [6, 7]),
        mark_as="is_persistent",
    )
    res = alifestd_prune_extinct_lineages_asexual(
        res, criterion="is_persistent"
    )
    assert res["id"].tolist() == [0, 1, 2, 3, 5, 8]


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "id": [3, 13, 23],
                    "ancestor_id": [3, 3, 3],
                    "origin_time": [0, 1, 1],
                }
            ),
            id="noncontiguous",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "id": [0, 1, 2],
                    "ancestor_id": [1, 1, 1],
                    "origin_time": [1, 0, 1],
                }
            ),
            id="unsorted",
        ),
    ],
)
def test_not_working_format_raises(phylogeny_df: pd.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mask_closest_relatives_asexual(
            phylogeny_df,
            candidate_mask=np.ones(3, dtype=bool),
            extant_mask=np.ones(3, dtype=bool),
        )


@pytest.mark.parametrize(
    "masks",
    [
        dict(candidate_mask=[False, True, True], extant_mask=[0, 1, 0]),
        dict(candidate_mask=[0, 1, 1], extant_mask=[False, True, False]),
    ],
)
def test_non_bool_mask_raises(masks: dict):
    phylogeny_df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 0, 0], "origin_time": [0, 1, 1]}
    )
    with pytest.raises(NotImplementedError):
        alifestd_mask_closest_relatives_asexual(phylogeny_df, **masks)


def test_missing_criterion_raises():
    phylogeny_df = pd.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 0, 0]})
    with pytest.raises(ValueError):
        alifestd_mask_closest_relatives_asexual(
            phylogeny_df,
            candidate_mask=np.ones(3, dtype=bool),
            extant_mask=np.ones(3, dtype=bool),
        )


@pytest.mark.parametrize(
    "origin_time",
    [
        pytest.param([1.0, 0.0, 2.0], id="decreasing"),
        pytest.param([0.0, np.nan, 1.0], id="nan"),
        pytest.param([0.0, np.inf, 1.0], id="inf"),
    ],
)
def test_unsupported_criterion_raises(origin_time: list):
    phylogeny_df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 0, 0], "origin_time": origin_time}
    )
    with pytest.raises(NotImplementedError):
        alifestd_mask_closest_relatives_asexual(
            phylogeny_df,
            candidate_mask=np.ones(3, dtype=bool),
            extant_mask=np.ones(3, dtype=bool),
        )

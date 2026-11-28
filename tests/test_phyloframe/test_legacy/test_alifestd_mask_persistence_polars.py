import typing

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_find_pair_distance_polars,
    alifestd_mark_leaves_polars,
    alifestd_mark_node_depth_polars,
    alifestd_mask_persistence_polars,
    alifestd_prune_extinct_lineages_polars,
)

apply_params = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


def _make_tree() -> pl.DataFrame:
    #        0
    #     1     2
    #    3 4   5 6
    #   7 8
    return pl.DataFrame(
        {
            "id": range(9),
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2, 3, 3],
            "origin_time": [0, 1, 1, 2, 3, 3, 9, 9, 4],
            "is_leaf": [False] * 4 + [True] * 5,
        },
    )


def _make_comb() -> pl.DataFrame:
    # 0 -- 1 -- 2 -- 3 -- 4
    #      |    |    +--- 5
    #      |    +-------- 6
    #      +------------- 7
    return pl.DataFrame(
        {
            "id": range(8),
            "ancestor_id": [0, 0, 1, 2, 3, 3, 2, 1],
            "origin_time": [0, 39, 48, 50, 100, 60, 50, 40],
        },
    )


def _make_random_tree(seed: int, n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    ancestor_ids = np.concatenate([[0], rng.integers(0, np.arange(1, n))])
    ancestor_ids[n // 2] = n // 2  # second root
    origin_times, mutations = np.zeros(n), np.zeros(n)
    for id_, ancestor_id in enumerate(ancestor_ids):  # integers, so ties
        origin_times[id_] = origin_times[ancestor_id] + rng.integers(3)
        mutations[id_] = mutations[ancestor_id] + rng.integers(3)
    return pl.DataFrame(
        {
            "id": range(n),
            "ancestor_id": ancestor_ids,
            "origin_time": origin_times,
            "mutations": mutations,
            "priority": rng.integers(4, size=n),
            "is_extant": rng.random(n) < 0.3,
            "is_candidate": rng.random(n) < 0.3,
        },
    )


@apply_params
def test_empty(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Int64),
                "ancestor_id": pl.Series([], dtype=pl.Int64),
                "origin_time": pl.Series([], dtype=pl.Float64),
                "is_leaf": pl.Series([], dtype=pl.Boolean),
            }
        ),
    )
    res = (
        alifestd_mask_persistence_polars(
            df_pl, candidate_mask="is_leaf", extant_mask="is_leaf"
        )
        .lazy()
        .collect()
    )
    assert res.is_empty()
    assert "alifestd_mask_persistence_polars" in res.columns


@apply_params
def test_simple(apply: typing.Callable):
    df_pl = _make_tree()
    res = (
        alifestd_mask_persistence_polars(
            apply(df_pl),
            candidate_mask=pl.col("is_leaf")
            & pl.col("origin_time").is_between(3, 4),
            extant_mask=pl.col("is_leaf") & (pl.col("origin_time") > 8),
            mark_as="m",
        )
        .lazy()
        .collect()
    )
    # leaf 7 is closest to its sibling 8, and leaf 6 to its sibling 5; leaf
    # 4 is further from leaf 7 than leaf 8 is, so is not marked
    assert res["m"].arg_true().to_list() == [5, 8]
    assert res.drop("m").equals(df_pl)


def test_mask_column_names():
    df_pl = _make_tree().with_columns(
        is_candidate=pl.col("id").is_in([4, 5, 8]),
        is_extant=pl.col("id").is_in([6, 7]),
    )
    res = alifestd_mask_persistence_polars(
        df_pl, candidate_mask="is_candidate", extant_mask="is_extant"
    )
    assert res["alifestd_mask_persistence_polars"].arg_true().to_list() == [
        5,
        8,
    ]


def test_closer_branch_beats_later_mrca():
    # leaf 7 shares a later mrca with leaf 8 than with leaf 4, but leaf 8's
    # long branch puts it further away
    df_pl = _make_tree().with_columns(
        origin_time=pl.Series([0, 1, 1, 2, 3, 3, 9, 9, 20]),
    )
    res = alifestd_mask_persistence_polars(
        df_pl,
        candidate_mask=pl.col("id").is_in([4, 8]),
        extant_mask=pl.col("id") == 7,
        mark_as="m",
    )
    assert res["m"].arg_true().to_list() == [4]


@pytest.mark.parametrize(
    "criteria, expected",
    [
        # by elapsed time, leaf 6 is closest: (100 - 48) + (50 - 48)
        pytest.param({}, [6], id="default"),
        # by number of edges, sibling leaf 5 is closest
        pytest.param(
            dict(criterion_cherry="node_depth", criterion_trunk="node_depth"),
            [5],
            id="node-depth",
        ),
        pytest.param(
            dict(
                criterion_cherry=pl.col("node_depth"),
                criterion_trunk=pl.col("node_depth") * 2,
            ),
            [5],
            id="expr",
        ),
        # leaf 5 shares the most recent mrca with leaf 4
        pytest.param(dict(criterion_cherry=pl.lit(0)), [5], id="trunk-only"),
        # leaf 7 sits closest to the lineage of leaf 4: 40 - 39
        pytest.param(dict(criterion_trunk=pl.lit(0)), [7], id="cherry-only"),
    ],
)
def test_criterion_cherry_trunk(criteria: dict, expected: list):
    res = alifestd_mark_node_depth_polars(_make_comb()).pipe(
        alifestd_mask_persistence_polars,
        candidate_mask=pl.col("id").is_in([5, 6, 7]),
        extant_mask=pl.col("id") == 4,
        mark_as="m",
        **criteria,
    )
    assert res["m"].arg_true().to_list() == expected


@pytest.mark.parametrize(
    "criterion_tiebreak, expected",
    [
        # default: latest, then lowest id
        pytest.param(
            pl.struct(
                pl.col("origin_time").rank("dense", descending=True), "id"
            ).rank("ordinal"),
            [7],
            id="default",
        ),
        pytest.param(pl.lit(0), [7, 8], id="constant"),
        pytest.param("priority", [8], id="column"),
        pytest.param(-pl.col("id"), [8], id="expr"),
        # candidates also tied by tiebreak value are all marked
        pytest.param("origin_time", [7, 8], id="tied"),
    ],
)
def test_criterion_tiebreak(
    criterion_tiebreak: typing.Union[str, pl.Expr], expected: list
):
    # leaves 7 and 8 are equally close to leaf 5; leaf 4 is closer to
    # leaf 6, whatever its tiebreak value
    df_pl = _make_tree().with_columns(
        origin_time=pl.Series([0, 1, 1, 2, 2, 2, 2, 3, 3]),
        priority=pl.Series([0, 0, 0, 0, 9, 0, 0, 2, 1]),
    )
    kwargs = dict(
        extant_mask=pl.col("id") == 5,
        mark_as="m",
        criterion_tiebreak=criterion_tiebreak,
    )
    res = alifestd_mask_persistence_polars(
        df_pl, candidate_mask=pl.col("id").is_in([7, 8]), **kwargs
    )
    assert res["m"].arg_true().to_list() == expected
    res = alifestd_mask_persistence_polars(
        df_pl, candidate_mask=pl.col("id").is_in([4, 7, 8]), **kwargs
    )
    assert res["m"].arg_true().to_list() == [4]


def test_default_tiebreak_takes_latest_sharing_most_recent_mrca():
    # without cherry distance, leaves 7 and 8 tie as closest to leaf 5
    df_pl = _make_tree().with_columns(
        origin_time=pl.Series([0, 1, 1, 2, 3, 3, 9, 20, 4]),
    )
    kwargs = dict(
        candidate_mask=pl.col("id").is_in([7, 8]),
        extant_mask=pl.col("id") == 5,
        criterion_cherry=pl.lit(0),
        mark_as="m",
    )
    res = alifestd_mask_persistence_polars(df_pl, **kwargs)
    assert res["m"].arg_true().to_list() == [7]
    res = alifestd_mask_persistence_polars(
        df_pl, criterion_tiebreak=pl.lit(0), **kwargs
    )
    assert res["m"].arg_true().to_list() == [7, 8]


def test_inner_nodes():
    # candidate ancestors (1) and descendants (7) of extant node 3
    res = alifestd_mask_persistence_polars(
        _make_tree(),
        candidate_mask=pl.col("id").is_in([0, 1, 7]),
        extant_mask=pl.col("id") == 3,
        mark_as="m",
    )
    assert res["m"].arg_true().to_list() == [1]


def test_inner_node_candidates_before_time():
    # among all taxa before time 49, node 2 is the latest ancestor of leaf 4
    res = alifestd_mask_persistence_polars(
        _make_comb(),
        candidate_mask=pl.col("origin_time") < 49,
        extant_mask=pl.col("id") == 4,
        mark_as="m",
    )
    assert res["m"].arg_true().to_list() == [2]


def test_inner_node_candidates_before_time_tied():
    #   0
    #   1
    #  2 5
    # 3 4
    # nodes 1, 2, 3, and 5 are at the same time, so are equally close to
    # leaf 4 in elapsed time; of these, node 2 is fewest edges away
    df_pl = pl.DataFrame(
        {
            "id": range(6),
            "ancestor_id": [0, 0, 1, 2, 2, 1],
            "origin_time": [0, 5, 5, 5, 20, 5],
        },
    )
    kwargs = dict(
        candidate_mask=pl.col("origin_time") < 10,
        extant_mask=pl.col("id") == 4,
        mark_as="m",
        criterion_tiebreak=pl.lit(0),
    )
    res = alifestd_mask_persistence_polars(df_pl, **kwargs)
    assert res["m"].arg_true().to_list() == [2]
    res = alifestd_mask_persistence_polars(
        df_pl,
        criterion_cherry="origin_time",
        criterion_trunk="origin_time",
        **kwargs,
    )
    assert res["m"].arg_true().to_list() == [1, 2, 3, 5]


@pytest.mark.parametrize(
    "node_depth, expected",
    [
        # leaves 4 and 8 are equally close to leaf 7 in elapsed time, but
        # leaf 8 is fewer edges away
        pytest.param(None, [8], id="calculated"),
        pytest.param([0, 1, 1, 2, 2, 2, 2, 3, 3], [8], id="existing"),
        pytest.param([0] * 9, [4, 8], id="existing-constant"),
    ],
)
def test_default_node_depth(
    node_depth: typing.Optional[typing.List[int]], expected: list
):
    df_pl = _make_tree().with_columns(
        origin_time=pl.Series([0, 1, 1, 1, 3, 3, 9, 9, 3]),
    )
    if node_depth is not None:
        df_pl = df_pl.with_columns(node_depth=pl.Series(node_depth))
    res = alifestd_mask_persistence_polars(
        df_pl,
        candidate_mask=pl.col("id").is_in([4, 8]),
        extant_mask=pl.col("id") == 7,
        mark_as="m",
        criterion_tiebreak=pl.lit(0),
    )
    assert res["m"].arg_true().to_list() == expected
    assert res.drop("m").equals(df_pl)  # no node depth column added


def test_large_integer_origin_time():
    # leaf 7 is closer to leaf 4 than leaf 8 is, by less than floats resolve
    df_pl = _make_tree().with_columns(
        origin_time=pl.Series(
            [0, 1, 1, 2, 2**53, 3, 9, 2**53, 2**53 + 1]
        ),
    )
    kwargs = dict(
        candidate_mask=pl.col("id").is_in([7, 8]),
        extant_mask=pl.col("id") == 4,
        mark_as="m",
        criterion_tiebreak=pl.lit(0),
    )
    res = alifestd_mask_persistence_polars(df_pl, **kwargs)
    assert res["m"].arg_true().to_list() == [7]
    res = alifestd_mask_persistence_polars(
        df_pl.with_columns(pl.col("origin_time").cast(pl.Float64)), **kwargs
    )
    assert res["m"].arg_true().to_list() == [7, 8]


def test_datetime_origin_time():
    df_pl = _make_tree().with_columns(
        origin_time=pl.datetime(2020, 1, 1)
        + pl.duration(days=pl.col("origin_time")),
    )
    kwargs = dict(
        candidate_mask=pl.col("id").is_in([4, 5, 8]),
        extant_mask=pl.col("id").is_in([6, 7]),
        mark_as="m",
    )
    res = alifestd_mask_persistence_polars(df_pl, **kwargs)
    assert res["m"].arg_true().to_list() == [5, 8]
    with pytest.raises(NotImplementedError):  # not as struct field
        alifestd_mask_persistence_polars(
            df_pl, criterion_cherry="origin_time", **kwargs
        )


@pytest.mark.parametrize(
    "name", ["criterion_cherry", "criterion_tiebreak", "criterion_trunk"]
)
@pytest.mark.parametrize("criterion", [pl.lit("a"), pl.lit(True)])
def test_non_numeric_criterion_raises(criterion: pl.Expr, name: str):
    with pytest.raises(NotImplementedError):
        alifestd_mask_persistence_polars(
            _make_tree(),
            candidate_mask=pl.lit(True),
            extant_mask=pl.lit(True),
            **{name: criterion},
        )


@pytest.mark.parametrize(
    "criteria",
    [{}, dict(criterion_cherry="origin_time", criterion_trunk="origin_time")],
)
def test_unsigned_origin_time(criteria: dict):
    df_pl = _make_tree().with_columns(pl.col("origin_time").cast(pl.UInt32))
    res = alifestd_mask_persistence_polars(
        df_pl,
        candidate_mask=pl.col("id").is_in([4, 5, 8]),
        extant_mask=pl.col("id").is_in([6, 7]),
        mark_as="m",
        **criteria,
    )
    assert res["m"].arg_true().to_list() == [5, 8]


@pytest.mark.parametrize("seed", range(5))
def test_inner_node_candidates_before_time_marks_latest_ancestors(seed: int):
    df_pl = alifestd_mark_leaves_polars(_make_random_tree(seed, n=300))
    ancestor_ids = df_pl["ancestor_id"].to_list()
    origin_times = df_pl["origin_time"].to_list()
    extant_mask = pl.col("is_leaf") & (pl.col("origin_time") > 7)
    extant_ids = df_pl.select(extant_mask).to_series().arg_true().to_list()
    for time in 5, 7:
        expected = set()
        for id_ in extant_ids:
            while origin_times[id_] >= time and ancestor_ids[id_] != id_:
                id_ = ancestor_ids[id_]
            if origin_times[id_] < time:
                expected.add(id_)

        res = alifestd_mask_persistence_polars(
            df_pl,
            candidate_mask=pl.col("origin_time") < time,
            extant_mask=extant_mask,
            mark_as="m",
        )
        assert res["m"].arg_true().to_list() == sorted(expected)
        assert 1 < len(expected) < len(extant_ids)


def test_extant_candidate_is_own_closest():
    res = alifestd_mask_persistence_polars(
        _make_tree(),
        candidate_mask=pl.col("id").is_in([7, 8]),
        extant_mask=pl.col("id") == 7,
        mark_as="m",
    )
    assert res["m"].arg_true().to_list() == [7]


@pytest.mark.parametrize(
    "candidate_ids, extant_ids", [([], [6, 7]), ([4, 5, 8], [])]
)
def test_no_extant_or_no_candidates(candidate_ids: list, extant_ids: list):
    res = alifestd_mask_persistence_polars(
        _make_tree(),
        candidate_mask=pl.col("id").is_in(candidate_ids),
        extant_mask=pl.col("id").is_in(extant_ids),
        mark_as="m",
    )
    assert not res["m"].any()


def test_multiple_roots():
    df_pl = pl.DataFrame(
        {
            "id": range(6),
            "ancestor_id": [0, 1, 0, 0, 1, 1],
            "origin_time": [0, 0, 1, 2, 1, 2],
        },
    )
    # extant leaf 5 has no candidate in its tree
    res = alifestd_mask_persistence_polars(
        df_pl,
        candidate_mask=pl.col("id").is_in([2, 3]),
        extant_mask=pl.col("id") == 5,
        mark_as="m",
    )
    assert not res["m"].any()
    res = alifestd_mask_persistence_polars(
        df_pl,
        candidate_mask=pl.col("id").is_in([2, 4]),
        extant_mask=pl.col("id").is_in([3, 5]),
        mark_as="m",
    )
    assert res["m"].arg_true().to_list() == [2, 4]


@pytest.mark.parametrize("seed", range(3))
def test_marked_candidates_minimize_pair_distance(seed: int):
    df_pl = _make_random_tree(seed)
    candidate_ids = df_pl["is_candidate"].arg_true().to_list()
    for extant_id in df_pl["is_extant"].arg_true():
        distances = {
            candidate_id: alifestd_find_pair_distance_polars(
                df_pl, extant_id, candidate_id
            )
            for candidate_id in candidate_ids
        }
        distances = {k: v for k, v in distances.items() if v is not None}
        res = alifestd_mask_persistence_polars(
            df_pl,
            candidate_mask="is_candidate",
            extant_mask=pl.col("id") == extant_id,
            mark_as="m",
        )
        marked_ids = res["m"].arg_true().to_list()
        assert len(marked_ids) == bool(distances)
        assert all(
            distances[id_] == min(distances.values()) for id_ in marked_ids
        )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("criterion_cherry", ["origin_time", pl.lit(0)])
def test_default_tiebreak_marks_one_per_extant(
    criterion_cherry: typing.Union[str, pl.Expr], seed: int
):
    df_pl = _make_random_tree(seed, n=300)
    kwargs = dict(
        candidate_mask="is_candidate",
        criterion_cherry=criterion_cherry,
        mark_as="m",
    )
    res = alifestd_mask_persistence_polars(
        df_pl, extant_mask="is_extant", **kwargs
    )
    res_all = alifestd_mask_persistence_polars(
        df_pl,
        criterion_tiebreak=pl.lit(0),
        extant_mask="is_extant",
        **kwargs,
    )
    assert (~res["m"] | res_all["m"]).all()  # subset of all tied
    assert res["m"].sum() < res_all["m"].sum()
    for extant_id in df_pl["is_extant"].arg_true():
        res = alifestd_mask_persistence_polars(
            df_pl, extant_mask=pl.col("id") == extant_id, **kwargs
        )
        assert res["m"].sum() == 1


def test_prune_extinct_lineages():
    res = alifestd_mask_persistence_polars(
        _make_tree(),
        candidate_mask=pl.col("id").is_in([4, 5, 8]),
        extant_mask=pl.col("id").is_in([6, 7]),
        mark_as="is_persistent",
    ).pipe(alifestd_prune_extinct_lineages_polars, criterion="is_persistent")
    assert res["id"].to_list() == [0, 1, 2, 3, 5, 8]


@pytest.mark.parametrize("name", ["is_leaf", "origin_time", "depth"])
def test_null_raises(name: str):
    df_pl = (
        _make_tree()
        .with_columns(depth=pl.Series([0, 1, 1, 2, 2, 2, 2, 3, 3]))
        .with_columns(
            pl.when(pl.col("id") == 4)
            .then(None)
            .otherwise(pl.col(name))
            .alias(name)
        )
    )
    for kwargs in (
        dict(candidate_mask="is_leaf", extant_mask=pl.lit(True)),
        dict(candidate_mask=pl.lit(True), extant_mask="is_leaf"),
    ):
        with pytest.raises(NotImplementedError):
            alifestd_mask_persistence_polars(
                df_pl, criterion_cherry="depth", **kwargs
            )


@pytest.mark.parametrize(
    "name", ["criterion_cherry", "criterion_tiebreak", "criterion_trunk"]
)
@pytest.mark.parametrize(
    "values",
    [
        pytest.param([0.0, None, 1.0], id="null"),
        pytest.param([0.0, float("nan"), 1.0], id="nan"),
        pytest.param([0.0, float("inf"), 1.0], id="inf"),
    ],
)
def test_unsupported_criterion_raises(values: list, name: str):
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
            "origin_time": [0.0, 1.0, 1.0],
            "unsupported": values,
        }
    )
    for criterion in "unsupported", pl.struct("origin_time", "unsupported"):
        with pytest.raises(NotImplementedError):
            alifestd_mask_persistence_polars(
                df_pl,
                candidate_mask=pl.lit(True),
                extant_mask=pl.lit(True),
                **{name: criterion},
            )


@pytest.mark.parametrize("name", ["criterion_cherry", "criterion_trunk"])
def test_decreasing_criterion_raises(name: str):
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
            "origin_time": [0.0, 1.0, 1.0],
            "decreasing": [1.0, 0.0, 2.0],
        }
    )
    kwargs = dict(candidate_mask=pl.lit(True), extant_mask=pl.lit(True))
    for criterion in "decreasing", pl.struct("origin_time", "decreasing"):
        with pytest.raises(NotImplementedError):
            alifestd_mask_persistence_polars(
                df_pl, **{name: criterion}, **kwargs
            )
        alifestd_mask_persistence_polars(
            df_pl, criterion_tiebreak=criterion, **kwargs
        )


@pytest.mark.parametrize(
    "criteria",
    [
        dict(criterion_cherry="missing"),
        dict(criterion_cherry=pl.struct("origin_time", "missing")),
        dict(criterion_tiebreak="missing"),
        dict(criterion_trunk="missing"),
    ],
)
def test_invalid_criterion_raises(criteria: dict):
    df_pl = pl.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 0, 0], "origin_time": [0, 1, 1]}
    )
    with pytest.raises(ValueError):
        alifestd_mask_persistence_polars(
            df_pl,
            candidate_mask=pl.lit(True),
            extant_mask=pl.lit(True),
            **criteria,
        )


@pytest.mark.parametrize(
    "df_pl",
    [
        pytest.param(
            pl.DataFrame(
                {
                    "id": [3, 13, 23],
                    "ancestor_id": [3, 3, 3],
                    "origin_time": [0, 1, 1],
                }
            ),
            id="noncontiguous",
        ),
        pytest.param(
            pl.DataFrame(
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
def test_not_working_format_raises(df_pl: pl.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mask_persistence_polars(
            df_pl, candidate_mask=pl.lit(True), extant_mask=pl.lit(True)
        )

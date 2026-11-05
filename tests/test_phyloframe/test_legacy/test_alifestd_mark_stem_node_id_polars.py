import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_stem_node_id_asexual,
    alifestd_mark_stem_node_id_polars,
)

apply_params = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


@apply_params
def test_empty(apply: typing.Callable):
    df_pl = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
        },
    )
    res = alifestd_mark_stem_node_id_polars(apply(df_pl)).lazy().collect()
    assert res.is_empty()
    assert "stem_node_id" in res.columns


@apply_params
def test_forest(apply: typing.Callable):
    #   0       5
    #   1      6 7
    #  2 3
    #  4
    df_pl = pl.DataFrame(
        {"id": range(8), "ancestor_id": [0, 0, 1, 1, 2, 5, 5, 5]},
    )
    res = (
        alifestd_mark_stem_node_id_polars(apply(df_pl), mark_as="s")
        .lazy()
        .collect()
    )
    assert res["s"].to_list() == [0, 1, 1, 1, 1, 5, 5, 5]


def test_random_tree_matches_pandas():
    rng = np.random.default_rng(1)
    n = 500
    ancestor_id = np.concatenate([[0], rng.integers(0, np.arange(1, n))])
    df_pd = pd.DataFrame({"id": range(n), "ancestor_id": ancestor_id})
    expected = alifestd_mark_stem_node_id_asexual(df_pd)
    res = alifestd_mark_stem_node_id_polars(pl.from_pandas(df_pd))
    assert res["stem_node_id"].to_list() == expected["stem_node_id"].tolist()


@pytest.mark.parametrize(
    "df_pl",
    [
        pytest.param(
            pl.DataFrame({"id": [3, 13, 23], "ancestor_id": [3, 3, 3]}),
            id="noncontiguous",
        ),
        pytest.param(
            pl.DataFrame({"id": [0, 1, 2], "ancestor_id": [1, 1, 1]}),
            id="unsorted",
        ),
    ],
)
def test_not_working_format_raises(df_pl: pl.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mark_stem_node_id_polars(df_pl)

import os

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_aggregate_phylogeny_partitions,
    alifestd_aggregate_phylogeny_partitions_polars,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_aggregate_phylogenies_polars import (
    alifestd_aggregate_phylogenies_polars,
)
from phyloframe.legacy._alifestd_assign_contiguous_ids_polars import (
    alifestd_assign_contiguous_ids_polars,
)
from phyloframe.legacy._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from phyloframe.legacy._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _load(name: str) -> pl.DataFrame:
    df = alifestd_to_working_format(pd.read_csv(f"{assets_path}/{name}.csv"))
    return pl.from_pandas(df).drop("ancestor_list", strict=False)


def _tagged(names):
    dfs = [_load(name) for name in names]
    return dfs, pl.concat(
        [df.with_columns(tree=pl.lit(i)) for i, df in enumerate(dfs)],
        how="diagonal_relaxed",
    )


@pytest.mark.parametrize(
    "names",
    [
        ["nk_ecoeaselection"],
        ["nk_ecoeaselection", "nk_tournamentselection"],
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("by", ["tree", pl.col("tree"), [pl.col("tree") + 0]])
def test_matches_aggregate_phylogenies(names, lazy, by):
    dfs, combined = _tagged(names)
    result = alifestd_aggregate_phylogeny_partitions_polars(
        combined.lazy() if lazy else combined,
        partitioned_by=by,
    )
    assert isinstance(result, pl.LazyFrame) == lazy
    result = result.lazy().collect()
    expected = alifestd_aggregate_phylogenies_polars(
        [
            alifestd_assign_contiguous_ids_polars(df).with_columns(
                tree=pl.lit(i),
            )
            for i, df in enumerate(dfs)
        ],
    )

    assert result.columns == combined.columns
    assert result.equals(expected.select(result.columns))
    assert result["id"].n_unique() == len(result)
    assert alifestd_has_contiguous_ids_polars(result)
    assert alifestd_is_topologically_sorted_polars(result)


def test_noncontiguous_ids_within_partitions():
    df = pl.DataFrame(
        {
            "id": [10, 20, 30, 40],
            "ancestor_id": [10, 20, 10, 20],
            "key": ["a", "b", "a", "b"],
        },
    )
    result = alifestd_aggregate_phylogeny_partitions_polars(df, "key")
    assert result["key"].to_list() == ["a", "a", "b", "b"]
    # partitions shifted by fixed offsets: a by 0, b by 31
    assert result["id"].to_list() == [10, 30, 51, 71]
    assert result["ancestor_id"].to_list() == [10, 10, 51, 51]


def test_multiple_keys_and_expr():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 1, 2, 2],
            "x": [1, 1, 2, 2],
            "y": [5, 6, 7, 8],
        },
    )
    result = alifestd_aggregate_phylogeny_partitions_polars(
        df, ["x", pl.col("y") > 5]
    )
    assert result.columns == df.columns
    # partitions (1, False), (1, True), (2, True) shifted by 0, 1, 3
    assert result["id"].to_list() == [0, 2, 5, 6]
    assert result["ancestor_id"].to_list() == [0, 2, 5, 5]


def test_agrees_with_pandas():
    __, combined = _tagged(["nk_ecoeaselection", "nk_tournamentselection"])
    res_pl = alifestd_aggregate_phylogeny_partitions_polars(combined, "tree")
    res_pd = alifestd_aggregate_phylogeny_partitions(
        combined.to_pandas(),
        "tree",
    )
    assert res_pl["id"].to_list() == res_pd["id"].tolist()
    assert res_pl["ancestor_id"].to_list() == res_pd["ancestor_id"].tolist()


def test_lazy_input_stays_lazy():
    __, combined = _tagged(["nk_ecoeaselection", "nk_tournamentselection"])
    result = alifestd_aggregate_phylogeny_partitions_polars(
        combined.lazy(),
        "tree",
    )
    assert isinstance(result, pl.LazyFrame)
    expected = alifestd_aggregate_phylogeny_partitions_polars(combined, "tree")
    assert isinstance(expected, pl.DataFrame)
    assert result.collect().equals(expected)


def test_empty():
    df = pl.DataFrame(
        {"id": [], "ancestor_id": [], "k": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64, "k": pl.Int64},
    )
    assert alifestd_aggregate_phylogeny_partitions_polars(df, "k").is_empty()


def test_invalid():
    df = pl.DataFrame({"id": [0], "ancestor_id": [0], "k": [1]})
    with pytest.raises(NotImplementedError):
        alifestd_aggregate_phylogeny_partitions_polars(
            df.drop("ancestor_id"), "k"
        )

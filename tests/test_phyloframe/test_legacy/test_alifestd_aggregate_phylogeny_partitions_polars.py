import os

import pandas as pd
import polars as pl
import polars.selectors as cs
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
        partition_by=by,
    )
    assert isinstance(result, pl.DataFrame)
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


def test_multiple_keys_and_expr():
    df = pl.DataFrame(
        {
            "id": [0, 0, 1, 0, 1],
            "ancestor_id": [0, 0, 0, 0, 0],
            "x": [1, 1, 1, 2, 2],
            "y": [5, 6, 7, 8, 9],
        },
    )
    result = alifestd_aggregate_phylogeny_partitions_polars(
        df, ["x", pl.col("y") > 5]
    )
    assert result.columns == df.columns
    # partitions (1, False), (1, True), (2, True) shifted by 0, 1, 3
    assert result["y"].to_list() == [5, 6, 7, 8, 9]
    assert result["id"].to_list() == [0, 1, 2, 3, 4]
    assert result["ancestor_id"].to_list() == [0, 1, 1, 3, 3]


def test_partitions_ordered_by_first_appearance():
    df = pl.DataFrame(
        {
            "id": [0, 0, 1, 1],
            "ancestor_id": [0, 0, 0, 0],
            "key": ["b", "a", "a", "b"],
        },
    )
    result = alifestd_aggregate_phylogeny_partitions_polars(df, "key")
    assert result["key"].to_list() == ["b", "b", "a", "a"]
    assert result["id"].to_list() == [0, 1, 2, 3]
    assert result["ancestor_id"].to_list() == [0, 0, 2, 2]


def test_noncontiguous_ids_within_partitions():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 1, 0, 1],
            "key": ["a", "b", "a", "b"],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_aggregate_phylogeny_partitions_polars(df, "key")


def test_agrees_with_pandas():
    __, combined = _tagged(["nk_ecoeaselection", "nk_tournamentselection"])
    res_pl = alifestd_aggregate_phylogeny_partitions_polars(combined, "tree")
    res_pd = alifestd_aggregate_phylogeny_partitions(
        combined.to_pandas(),
        "tree",
    )
    assert res_pl["id"].to_list() == res_pd["id"].tolist()
    assert res_pl["ancestor_id"].to_list() == res_pd["ancestor_id"].tolist()


def test_empty():
    df = pl.DataFrame(
        {"id": [], "ancestor_id": [], "k": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64, "k": pl.Int64},
    )
    result = alifestd_aggregate_phylogeny_partitions_polars(df, "k")
    assert result.is_empty()
    assert result.columns == df.columns


def test_invalid():
    df = pl.DataFrame({"id": [0], "ancestor_id": [0], "k": [1]})
    with pytest.raises(NotImplementedError):
        alifestd_aggregate_phylogeny_partitions_polars(
            df.drop("ancestor_id"), "k"
        )


def test_preserves_id_dtypes():
    df = pl.DataFrame(
        {"id": [0, 1, 0], "ancestor_id": [0, 0, 0], "k": [1, 1, 2]},
        schema={"id": pl.UInt16, "ancestor_id": pl.UInt16, "k": pl.Int64},
    )
    result = alifestd_aggregate_phylogeny_partitions_polars(df, "k")
    assert result.schema["id"] == pl.UInt16
    assert result.schema["ancestor_id"] == pl.UInt16
    assert result["id"].to_list() == [0, 1, 2]
    assert result["ancestor_id"].to_list() == [0, 0, 2]


@pytest.mark.parametrize(
    "partition_by",
    [
        ["x", pl.col("y") > 5],
        [cs.by_name("x"), pl.col("y") > 5],
        pl.struct("x", pl.col("y") > 5),
    ],
)
def test_partition_by_forms(partition_by):
    df = pl.DataFrame(
        {
            "id": [0, 0, 1, 0, 1],
            "ancestor_id": [0, 0, 0, 0, 0],
            "x": [1, 1, 1, 2, 2],
            "y": [5, 6, 7, 8, 9],
        },
    )
    result = alifestd_aggregate_phylogeny_partitions_polars(df, partition_by)
    assert result["id"].to_list() == [0, 1, 2, 3, 4]
    assert result["ancestor_id"].to_list() == [0, 1, 1, 3, 3]

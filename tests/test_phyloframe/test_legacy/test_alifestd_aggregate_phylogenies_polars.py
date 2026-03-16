import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_aggregate_phylogenies,
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
from phyloframe.legacy._alifestd_mark_leaves_polars import (
    alifestd_mark_leaves_polars,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _prepare_polars(csv_path: str) -> pl.DataFrame:
    """Load CSV, convert to working format, and return as polars DataFrame."""
    df_pd = alifestd_to_working_format(pd.read_csv(csv_path))
    df_pl = pl.from_pandas(df_pd)
    return df_pl.drop("ancestor_list", strict=False)


@pytest.mark.parametrize(
    "phylogeny_df1",
    [
        _prepare_polars(f"{assets_path}/nk_ecoeaselection.csv"),
        _prepare_polars(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize(
    "phylogeny_df2",
    [
        _prepare_polars(f"{assets_path}/nk_ecoeaselection.csv"),
        _prepare_polars(f"{assets_path}/nk_tournamentselection.csv"),
        None,
    ],
)
@pytest.mark.parametrize(
    "phylogeny_df3",
    [
        pytest.param(
            _prepare_polars(f"{assets_path}/nk_ecoeaselection.csv"),
            marks=pytest.mark.heavy,
        ),
        pytest.param(
            _prepare_polars(f"{assets_path}/nk_tournamentselection.csv"),
            marks=pytest.mark.heavy,
        ),
        None,
    ],
)
def test_alifestd_aggregate_phylogenies_polars(
    phylogeny_df1: pl.DataFrame,
    phylogeny_df2: typing.Optional[pl.DataFrame],
    phylogeny_df3: typing.Optional[pl.DataFrame],
):
    if phylogeny_df3 is not None and phylogeny_df2 is None:
        return

    phylogenies = [phylogeny_df1]
    if phylogeny_df2 is not None:
        phylogenies.append(phylogeny_df2)
    if phylogeny_df3 is not None:
        phylogenies.append(phylogeny_df3)

    result = alifestd_aggregate_phylogenies_polars(phylogenies)

    # ids should be unique
    assert result["id"].n_unique() == len(result)

    # ancestor_id column should be present
    assert "ancestor_id" in result.columns

    # result should have contiguous ids
    assert alifestd_has_contiguous_ids_polars(result)

    # result should be topologically sorted
    assert alifestd_is_topologically_sorted_polars(result)

    # total number of leaves should be sum of individual leaf counts
    result_num_leaves = (
        alifestd_mark_leaves_polars(
            alifestd_assign_contiguous_ids_polars(
                result.select("id", "ancestor_id"),
            ),
        )
        .lazy()
        .select(pl.col("is_leaf").sum())
        .collect()
        .item()
    )
    expected_num_leaves = 0
    for df in phylogenies:
        expected_num_leaves += (
            alifestd_mark_leaves_polars(df)
            .lazy()
            .select(pl.col("is_leaf").sum())
            .collect()
            .item()
        )
    assert result_num_leaves == expected_num_leaves

    # total row count should be sum of individual row counts
    assert len(result) == sum(len(df) for df in phylogenies)


def test_alifestd_aggregate_phylogenies_polars_empty():
    df = pl.DataFrame(
        {"id": [], "ancestor_id": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64},
    )
    result = alifestd_aggregate_phylogenies_polars([df, df])
    assert result.is_empty()


def test_alifestd_aggregate_phylogenies_polars_empty_list():
    result = alifestd_aggregate_phylogenies_polars([])
    assert result.is_empty()
    assert "id" in result.columns
    assert "ancestor_id" in result.columns


def test_alifestd_aggregate_phylogenies_polars_single():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
        },
    )
    result = alifestd_aggregate_phylogenies_polars([df])
    assert len(result) == 3
    assert result["id"].to_list() == [0, 1, 2]
    assert result["ancestor_id"].to_list() == [0, 0, 1]


def test_alifestd_aggregate_phylogenies_polars_simple():
    """Test two simple trees get aggregated with shifted ids."""
    df1 = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
        },
    )
    df2 = pl.DataFrame(
        {
            "id": [0, 1],
            "ancestor_id": [0, 0],
        },
    )

    result = alifestd_aggregate_phylogenies_polars([df1, df2])

    assert len(result) == 5
    assert result["id"].n_unique() == 5
    # first tree: ids 0, 1, 2 unchanged
    # second tree: ids shifted by 3 (max_id 2 + 1)
    assert result["id"].to_list() == [0, 1, 2, 3, 4]
    assert result["ancestor_id"].to_list() == [0, 0, 0, 3, 3]


def test_alifestd_aggregate_phylogenies_polars_preserves_columns():
    """Extra columns should be preserved."""
    df1 = pl.DataFrame(
        {
            "id": [0, 1],
            "ancestor_id": [0, 0],
            "origin_time": [0.0, 1.0],
        },
    )
    df2 = pl.DataFrame(
        {
            "id": [0, 1],
            "ancestor_id": [0, 0],
            "origin_time": [5.0, 6.0],
        },
    )

    result = alifestd_aggregate_phylogenies_polars([df1, df2])
    assert "origin_time" in result.columns
    assert result["origin_time"].to_list() == [0.0, 1.0, 5.0, 6.0]


def test_alifestd_aggregate_phylogenies_polars_no_side_effects():
    """Input dataframes should not be mutated."""
    df1 = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
        },
    )
    df2 = pl.DataFrame(
        {
            "id": [0, 1],
            "ancestor_id": [0, 0],
        },
    )

    df1_orig = df1.clone()
    df2_orig = df2.clone()

    alifestd_aggregate_phylogenies_polars([df1, df2])

    assert df1.equals(df1_orig)
    assert df2.equals(df2_orig)


def test_alifestd_aggregate_phylogenies_polars_ancestor_list_raises():
    """Should raise NotImplementedError if ancestor_list column present."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[none]", "[0]", "[1]"],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_aggregate_phylogenies_polars([df])


def test_alifestd_aggregate_phylogenies_polars_no_ancestor_id_raises():
    """Should raise NotImplementedError without ancestor_id column."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_aggregate_phylogenies_polars([df])


def test_alifestd_aggregate_phylogenies_polars_noncontiguous_raises():
    """Should raise NotImplementedError for non-contiguous ids."""
    df = pl.DataFrame(
        {
            "id": [0, 2, 5],
            "ancestor_id": [0, 0, 2],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_aggregate_phylogenies_polars([df])


def test_alifestd_aggregate_phylogenies_polars_not_sorted_raises():
    """Should raise NotImplementedError for non-topologically-sorted data."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 2, 0],  # id 1 has ancestor 2 > 1
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_aggregate_phylogenies_polars([df])


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        _prepare_polars(f"{assets_path}/nk_ecoeaselection.csv"),
        _prepare_polars(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
def test_alifestd_aggregate_phylogenies_polars_matches_pandas(
    phylogeny_df: pl.DataFrame,
):
    """Verify polars result matches pandas result for same inputs."""
    df1_pl = phylogeny_df
    df2_pl = phylogeny_df.clone()

    # Run pandas version
    df1_pd = phylogeny_df.to_pandas()
    df2_pd = phylogeny_df.to_pandas()
    result_pd = alifestd_aggregate_phylogenies([df1_pd, df2_pd])

    # Run polars version
    result_pl = alifestd_aggregate_phylogenies_polars([df1_pl, df2_pl])

    assert len(result_pd) == len(result_pl)
    assert result_pl["id"].to_list() == result_pd["id"].to_list()
    assert (
        result_pl["ancestor_id"].to_list()
        == result_pd["ancestor_id"].to_list()
    )

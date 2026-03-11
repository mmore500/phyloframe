import os

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_aggregate_phylogenies,
    alifestd_count_leaf_nodes,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_downsample_tips_clade_asexual import (
    alifestd_downsample_tips_clade_asexual,
)
from phyloframe.legacy._alifestd_downsample_tips_clade_polars import (
    alifestd_downsample_tips_clade_polars,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _count_leaf_nodes_polars(phylogeny_df: pl.DataFrame) -> int:
    """Count leaf nodes in a polars dataframe (works with any ids)."""
    all_ids = set(phylogeny_df["id"].to_list())
    internal_ids = set(
        phylogeny_df.filter(pl.col("ancestor_id") != pl.col("id"))
        .select("ancestor_id")
        .to_series()
        .to_list()
    )
    return len(all_ids - internal_ids)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            alifestd_aggregate_phylogenies(
                [
                    pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
                    pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
                ]
            )
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
        ),
    ],
)
@pytest.mark.parametrize(
    "n_downsample",
    [1, pytest.param(5, marks=pytest.mark.heavy), 10, 100000000],
)
@pytest.mark.parametrize(
    "seed", [1, pytest.param(42, marks=pytest.mark.heavy)]
)
def test_alifestd_downsample_tips_clade_polars(
    phylogeny_df: pd.DataFrame,
    n_downsample: int,
    seed: int,
):
    phylogeny_df_pl = pl.from_pandas(phylogeny_df)

    original_len = len(phylogeny_df_pl)
    original_leaf_count = _count_leaf_nodes_polars(phylogeny_df_pl)

    result_df = alifestd_downsample_tips_clade_polars(
        phylogeny_df_pl,
        n_downsample,
        seed=seed,
    )

    assert len(result_df) <= original_len
    assert "extant" not in result_df.columns
    assert set(result_df["id"].to_list()).issubset(
        set(phylogeny_df_pl["id"].to_list())
    )
    assert _count_leaf_nodes_polars(result_df) <= min(
        original_leaf_count, n_downsample
    )


@pytest.mark.parametrize("n_downsample", [0, 1])
def test_alifestd_downsample_tips_clade_polars_empty(n_downsample: int):
    phylogeny_df = pl.DataFrame(
        {"id": [], "ancestor_id": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64},
    )

    result_df = alifestd_downsample_tips_clade_polars(
        phylogeny_df, n_downsample
    )

    assert result_df.is_empty()


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            alifestd_aggregate_phylogenies(
                [
                    pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
                    pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
                ]
            )
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
        ),
    ],
)
@pytest.mark.parametrize(
    "n_downsample", [1, pytest.param(5, marks=pytest.mark.heavy), 10]
)
@pytest.mark.parametrize(
    "seed", [1, pytest.param(42, marks=pytest.mark.heavy)]
)
def test_alifestd_downsample_tips_clade_polars_matches_pandas(
    phylogeny_df: pd.DataFrame,
    n_downsample: int,
    seed: int,
):
    """Verify polars result matches pandas result for same prepared input."""
    phylogeny_df_pl = pl.from_pandas(phylogeny_df)

    result_pd = alifestd_downsample_tips_clade_asexual(
        phylogeny_df, n_downsample, mutate=False, seed=seed
    )
    result_pl = alifestd_downsample_tips_clade_polars(
        phylogeny_df_pl, n_downsample, seed=seed
    )

    assert set(result_pd["id"]) == set(result_pl["id"].to_list())
    assert len(result_pd) == len(result_pl)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
        ),
    ],
)
@pytest.mark.parametrize(
    "seed", [1, pytest.param(42, marks=pytest.mark.heavy)]
)
def test_alifestd_downsample_tips_clade_polars_deterministic(
    phylogeny_df: pd.DataFrame,
    seed: int,
):
    """Verify same seed produces same result."""
    phylogeny_df_pl = pl.from_pandas(phylogeny_df)

    result1 = alifestd_downsample_tips_clade_polars(
        phylogeny_df_pl, 5, seed=seed
    )
    result2 = alifestd_downsample_tips_clade_polars(
        phylogeny_df_pl, 5, seed=seed
    )

    assert set(result1["id"].to_list()) == set(result2["id"].to_list())
    assert len(result1) == len(result2)


def test_alifestd_downsample_tips_clade_polars_no_ancestor_id():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[none]", "[0]", "[1]"],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_downsample_tips_clade_polars(df, 1)


def test_alifestd_downsample_tips_clade_polars_simple():
    """Test a simple hand-crafted tree.

    Tree structure:
        0 (root)
        +-- 1
        |   +-- 3 (leaf)
        |   +-- 4 (leaf)
        +-- 2 (leaf)

    With n_downsample=2, keep the sampled clade with at most 2 leaves.
    With n_downsample=1, keep a single leaf and its ancestors.
    """
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
            "destruction_time": [float("inf")] * 5,
        }
    )

    result = alifestd_downsample_tips_clade_polars(df, 2, seed=1)
    assert len(result) <= 5
    assert _count_leaf_nodes_polars(result) <= 2

    result1 = alifestd_downsample_tips_clade_polars(df, 1, seed=1)
    assert _count_leaf_nodes_polars(result1) == 1


def test_alifestd_downsample_tips_clade_polars_all_tips():
    """Requesting more tips than exist should return the full phylogeny."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
            "destruction_time": [float("inf")] * 5,
        }
    )

    result = alifestd_downsample_tips_clade_polars(df, 100000, seed=1)

    assert len(result) == 5

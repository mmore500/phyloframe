import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_aggregate_phylogenies,
    alifestd_aggregate_phylogeny_partitions,
    alifestd_find_leaf_ids,
    alifestd_to_working_format,
    alifestd_validate,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _load(name: str) -> pd.DataFrame:
    return alifestd_to_working_format(
        pd.read_csv(f"{assets_path}/{name}.csv"),
    )


def _tag(dfs):
    return pd.concat(
        [df.assign(tree=i) for i, df in enumerate(dfs)],
    ).reset_index(drop=True)


@pytest.mark.parametrize("n_trees", [1, 2, 3])
@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("drop_ancestor_list", [False, True])
def test_matches_aggregate_phylogenies(n_trees, mutate, drop_ancestor_list):
    sources = [
        _load("nk_ecoeaselection"),
        _load("nk_tournamentselection"),
        _load("nk_ecoeaselection"),
    ][:n_trees]
    if drop_ancestor_list:
        sources = [df.drop(columns="ancestor_list") for df in sources]
    combined = _tag(sources)
    original = combined.copy()

    result = alifestd_aggregate_phylogeny_partitions(
        combined if not mutate else combined.copy(),
        partitioned_by="tree",
        mutate=mutate,
    )
    expected = alifestd_aggregate_phylogenies(
        [df.assign(tree=i) for i, df in enumerate(sources)],
    )

    pd.testing.assert_frame_equal(result, expected)
    assert result["id"].is_unique
    if not drop_ancestor_list:
        assert alifestd_validate(result)
    assert len(alifestd_find_leaf_ids(result)) == sum(
        map(len, map(alifestd_find_leaf_ids, sources)),
    )
    if not mutate:
        pd.testing.assert_frame_equal(combined, original)


def test_interleaved_rows_and_multiple_columns():
    a, b = _load("nk_ecoeaselection"), _load("nk_tournamentselection")
    combined = _tag([a, b])
    combined["group"] = combined["tree"] % 2
    # interleave rows between partitions; relative order preserved
    combined = combined.iloc[
        sorted(range(len(combined)), key=lambda i: (i % 7, i))
    ].reset_index(drop=True)

    result = alifestd_aggregate_phylogeny_partitions(
        combined,
        partitioned_by=["tree", "group"],
    )
    assert result["id"].is_unique
    assert alifestd_validate(result)
    assert len(result) == len(a) + len(b)
    # partitions ordered by first appearance
    firsts = result["tree"].drop_duplicates().tolist()
    assert firsts == combined["tree"].drop_duplicates().tolist()


def test_empty():
    df = _tag([_load("nk_ecoeaselection")]).iloc[:0]
    result = alifestd_aggregate_phylogeny_partitions(df, "tree")
    assert len(result) == 0
    assert list(result.columns) == list(df.columns)


def test_missing_partitioned_by():
    df = _tag([_load("nk_ecoeaselection")])
    with pytest.raises(KeyError):
        alifestd_aggregate_phylogeny_partitions(df, "nonexistent")

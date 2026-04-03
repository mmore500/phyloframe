import os

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_aggregate_phylogenies,
    alifestd_assign_contiguous_ids,
    alifestd_find_leaf_ids,
)
from phyloframe.legacy import (
    alifestd_prune_extinct_lineages_asexual as alifestd_prune_extinct_lineages_asexual_,
)
from phyloframe.legacy import (
    alifestd_to_working_format,
    alifestd_topological_sort,
    alifestd_try_add_ancestor_id_col,
    alifestd_unfurl_lineage_asexual,
    alifestd_validate,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_prune_extinct_lineages_asexual = enforce_dtype_stability_pandas(
    alifestd_prune_extinct_lineages_asexual_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        alifestd_aggregate_phylogenies(
            [
                pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
                pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
            ]
        ),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        alifestd_assign_contiguous_ids,
        alifestd_to_working_format,
        alifestd_topological_sort,
        alifestd_try_add_ancestor_id_col,
        lambda x: x.sample(frac=1, random_state=1),
        lambda x: x,
    ],
)
def test_alifestd_prune_extinct_lineages_asexual_extant_col_nop(
    phylogeny_df, apply
):

    assert "destruction_time" in phylogeny_df
    leaf_mask = phylogeny_df["id"].isin(
        {*alifestd_find_leaf_ids(phylogeny_df)}
    )
    leaf_destruction_times = phylogeny_df.loc[leaf_mask, "destruction_time"]
    assert (
        leaf_destruction_times.isna() | np.isinf(leaf_destruction_times)
    ).all()
    # add extant column based on destruction_time
    phylogeny_df["extant"] = phylogeny_df[
        "destruction_time"
    ].isna() | np.isinf(phylogeny_df["destruction_time"])
    # because the source dataframes are pruned, pruning will be a nop
    phylogeny_df = apply(phylogeny_df.copy()).reset_index(drop=True)

    phylogeny_df_ = phylogeny_df.copy()
    pruned_df = alifestd_prune_extinct_lineages_asexual(phylogeny_df)
    assert phylogeny_df.equals(phylogeny_df_)

    assert len(phylogeny_df) == len(pruned_df)

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df)
    assert {*phylogeny_df} == {*pruned_df}
    assert phylogeny_df.equals(pruned_df), phylogeny_df.compare(pruned_df)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        alifestd_aggregate_phylogenies(
            [
                pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
                pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
            ]
        ),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        alifestd_assign_contiguous_ids,
        alifestd_to_working_format,
        alifestd_topological_sort,
        alifestd_try_add_ancestor_id_col,
        lambda x: x.sample(frac=1, random_state=1),
        lambda x: x,
    ],
)
def test_alifestd_prune_extinct_lineages_asexual(phylogeny_df, apply):
    # because the source dataframes are pruned, pruning will be a nop
    phylogeny_df = apply(phylogeny_df.copy())

    extant_mask = np.random.choice([True, False], size=len(phylogeny_df))
    phylogeny_df["extant"] = extant_mask

    phylogeny_df_ = phylogeny_df.copy()
    pruned_df = alifestd_prune_extinct_lineages_asexual(phylogeny_df)
    assert len(pruned_df) < len(phylogeny_df)
    assert set(pruned_df["id"]) >= set(phylogeny_df.loc[extant_mask, "id"])

    assert alifestd_validate(pruned_df)
    assert phylogeny_df.equals(phylogeny_df_)

    for leaf_id in alifestd_find_leaf_ids(pruned_df):
        assert [*alifestd_unfurl_lineage_asexual(phylogeny_df, leaf_id)] == [
            *alifestd_unfurl_lineage_asexual(pruned_df, leaf_id)
        ]


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        alifestd_aggregate_phylogenies(
            [
                pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
                pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
            ]
        ),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        alifestd_assign_contiguous_ids,
        alifestd_to_working_format,
        alifestd_topological_sort,
        alifestd_try_add_ancestor_id_col,
        lambda x: x.sample(frac=1, random_state=1),
        lambda x: x,
    ],
)
def test_alifestd_prune_extinct_lineages_asexual_independent_trees(
    phylogeny_df, apply
):
    # because the source dataframes are pruned, pruning will be a nop
    phylogeny_df = apply(phylogeny_df.copy())
    phylogeny_df["extant"] = False

    first_df = phylogeny_df.copy()
    extant_mask = first_df["id"].isin(alifestd_find_leaf_ids(first_df))
    first_df.loc[extant_mask, "extant"] = True

    second_df = phylogeny_df.copy()

    pruned_df = alifestd_prune_extinct_lineages_asexual(
        alifestd_aggregate_phylogenies([first_df, second_df]),
    )
    assert len(pruned_df) == len(phylogeny_df)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        alifestd_aggregate_phylogenies(
            [
                pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
                pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
            ]
        ),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        alifestd_assign_contiguous_ids,
        alifestd_to_working_format,
        alifestd_topological_sort,
        alifestd_try_add_ancestor_id_col,
        lambda x: x.sample(frac=1, random_state=1),
        lambda x: x,
    ],
)
def test_alifestd_prune_extinct_lineages_asexual_missing_criterion(
    phylogeny_df, apply
):
    # because the source dataframes are pruned, pruning will be a nop
    phylogeny_df = apply(phylogeny_df.copy())

    with pytest.raises(ValueError):
        alifestd_prune_extinct_lineages_asexual(phylogeny_df)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        alifestd_assign_contiguous_ids,
        alifestd_to_working_format,
        alifestd_topological_sort,
        alifestd_try_add_ancestor_id_col,
        lambda x: x.sample(frac=1, random_state=1),
        lambda x: x,
    ],
)
def test_alifestd_prune_extinct_lineages_asexual_custom_criterion(
    phylogeny_df, apply
):
    phylogeny_df = apply(phylogeny_df.copy())

    extant_mask = np.random.choice([True, False], size=len(phylogeny_df))
    phylogeny_df["my_custom_col"] = extant_mask

    pruned_custom = alifestd_prune_extinct_lineages_asexual(
        phylogeny_df, criterion="my_custom_col"
    )

    phylogeny_df["extant"] = extant_mask
    pruned_default = alifestd_prune_extinct_lineages_asexual(phylogeny_df)

    assert len(pruned_custom) == len(pruned_default)
    assert set(pruned_custom["id"]) == set(pruned_default["id"])

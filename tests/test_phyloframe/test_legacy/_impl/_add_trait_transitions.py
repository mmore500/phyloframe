import typing

import numpy as np
import pandas as pd

from phyloframe.legacy import (
    alifestd_mark_lineage_delta_asexual,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_mark_leaves import (
    _alifestd_mark_leaves_asexual_fast_path,
)
from phyloframe.legacy._alifestd_mask_trait_mdcs_strict_asexual import (
    _alifestd_mask_trait_uniform_clades_asexual_fast_path,
)


def add_trait_transitions(
    phylogeny_df: pd.DataFrame,
    trait_masks: typing.Sequence[np.ndarray],
    names: typing.Optional[typing.Sequence[str]] = None,
) -> typing.Tuple[pd.DataFrame, typing.List[str]]:
    """Return a copy of `phylogeny_df` in working format with one transition
    column per trait mask, named by `names` (default, trait index), and the
    column names.

    Reconstructs states from leaf values under parsimony without loss: a
    node has a trait if and only if every leaf of its clade does.
    """
    phylogeny_df = phylogeny_df.copy()
    if names is None:
        names = [str(i) for i in range(len(trait_masks))]
    names = list(names)
    for name, mask in zip(names, trait_masks):
        phylogeny_df[name] = np.asarray(mask, dtype=bool)
    phylogeny_df = alifestd_to_working_format(phylogeny_df, mutate=True)
    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    is_leaf = _alifestd_mark_leaves_asexual_fast_path(ancestor_ids)
    for name in names:
        phylogeny_df[
            name
        ] = _alifestd_mask_trait_uniform_clades_asexual_fast_path(
            ancestor_ids, is_leaf, phylogeny_df[name].to_numpy()
        )
        phylogeny_df = alifestd_mark_lineage_delta_asexual(
            phylogeny_df, name, mutate=True, basal_value=False, mark_as=name
        )
    return phylogeny_df, names

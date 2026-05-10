import polars as pl
import tqdist

from ._alifestd_make_distance_newicks_polars import (
    _alifestd_make_distance_newicks_polars,
)


# adapted from https://github.com/mmore500/hstrat/blob/d23917cf/tests/test_hstrat/test_phylogenetic_inference/test_tree/_impl/_tree_quartet_distance.py
def alifestd_calc_triplet_distance_polars(
    ref: pl.DataFrame,
    cmp: pl.DataFrame,
    taxon_label_key: str = "taxon_label",
) -> float:
    """Calculate the triplet distance between two trees.

    See Also
    --------
    alifestd_calc_triplet_distance_asexual :
        Pandas-based implementation.
    """
    ref_newick, cmp_newick = _alifestd_make_distance_newicks_polars(
        ref,
        cmp,
        taxon_label_key,
        caller_name=alifestd_calc_triplet_distance_polars.__name__,
    )
    return tqdist.triplet_distance(ref_newick, cmp_newick)

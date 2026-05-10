import pandas as pd
import tqdist

from ._alifestd_make_distance_newicks_asexual import (
    _alifestd_make_distance_newicks_asexual,
)


# adapted from https://github.com/mmore500/hstrat/blob/d23917cf/tests/test_hstrat/test_phylogenetic_inference/test_tree/_impl/_tree_quartet_distance.py
def alifestd_calc_quartet_distance_asexual(
    ref: pd.DataFrame,
    cmp: pd.DataFrame,
    taxon_label_key: str = "taxon_label",
) -> float:
    """Calculate the quartet distance between two trees."""
    ref_newick, cmp_newick = _alifestd_make_distance_newicks_asexual(
        ref,
        cmp,
        taxon_label_key,
        caller_name=alifestd_calc_quartet_distance_asexual.__name__,
    )
    return tqdist.quartet_distance(ref_newick, cmp_newick)

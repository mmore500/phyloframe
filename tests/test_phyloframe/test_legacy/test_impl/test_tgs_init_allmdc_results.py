import numpy as np

from phyloframe.legacy._impl import TgsAllmdcUnits, tgs_init_allmdc_results


def test_init():
    units = TgsAllmdcUnits(
        group_nodes=[np.arange(3), np.arange(2)],
        group_bounds=[(1, 1), (2, 3)],
        trait_units=[
            [np.arange(3), np.arange(2)],
            [np.arange(1), np.arange(2)],
        ],
        trait_gains=[
            [np.arange(1), np.arange(0)],
            [np.arange(0), np.arange(2)],
        ],
    )
    res = tgs_init_allmdc_results(trait_names=["a", "b"], units=units)
    assert res["num_leaves"].tolist() == [3, 2]
    assert res["num_trait_present"].tolist() == [1, 0]
    assert res["num_clades"].tolist() == [1, 2]
    assert res["num_clade_gains"].tolist() == [0, 2]
    assert res["clade_size_shift_2_3"].isna().all()
    assert np.isnan(res["selection"]).all()

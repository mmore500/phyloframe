import numpy as np

from phyloframe.legacy._impl import TgsAllmdcBranches, tgs_init_allmdc_results


def test_init():
    branches = TgsAllmdcBranches(
        clade_sizes=np.array([1, 1, 2, 1, 5]),
        branch_lengths=np.ones(5),
        survival_times=np.zeros(5),
        origin_times=np.zeros(5),
        trait_units=[np.array([0, 1, 2, 3, 4]), np.array([0, 2])],
        trait_gains=[np.array([1, 2]), np.array([], dtype=int)],
    )
    res = tgs_init_allmdc_results(trait_names=["a", "b"], branches=branches)
    assert res.columns[:5].tolist() == [
        "trait",
        "num_leaves",
        "num_trait_present",
        "num_clades",
        "num_clade_gains",
    ]
    assert res["num_leaves"].tolist() == [3, 1]
    assert res["num_trait_present"].tolist() == [1, 0]
    assert res["num_clades"].tolist() == [2, 1]
    assert res["num_clade_gains"].tolist() == [1, 0]
    assert (res["gain_rate"] == 0.0).all()
    assert np.isnan(res["selection"]).all()

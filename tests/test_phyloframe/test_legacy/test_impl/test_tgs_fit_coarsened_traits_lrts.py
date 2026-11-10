import numpy as np
import polars as pl

from phyloframe._auxlib._identity import identity
from phyloframe.legacy._impl import tgs_fit_coarsened_traits

from ._make_pendant_nll_kwargs import make_pendant_nll_kwargs


def _simulate(gain_rate: float, selection_coef: float, seed: int):
    rng = np.random.default_rng(seed)
    length = np.linspace(0.5, 60.0, 120)
    num_trials = np.full_like(length, 20_000)
    p = (
        gain_rate * length
        if selection_coef == 0
        else gain_rate * np.expm1(selection_coef * length) / selection_coef
    )
    return make_pendant_nll_kwargs(
        pl.DataFrame(
            {
                "num_trials": num_trials,
                "num_successes": rng.binomial(
                    num_trials.astype(int), p
                ).astype(float),
                "mean_episode_length": length,
            },
        ),
    )


def _fit(traits, **kwargs) -> pl.DataFrame:
    return tgs_fit_coarsened_traits(
        trait_nll_kwargs=[
            _simulate(gain_rate, selection_coef, seed)
            for seed, (gain_rate, selection_coef) in enumerate(traits)
        ],
        loss_rate=0.0,
        selection_coef_starts=(0.0, -0.3, -1.8),
        growth_exponent_bound=600.0,
        progress_wrap=identity,
        **kwargs,
    )


def test_selection_lrts():
    # trait 0 has a cost; traits 1 and 2 are neutral; trait 3 has no gains
    res = _fit(
        [(2e-3, -0.05), (1e-3, 0.0), (1e-3, 0.0), (0.0, 0.0)],
        neutral_trait_ids=[1, 2],
    )
    lrts = res["selection_coef_lrt_naive"].to_numpy()
    assert lrts[0] > 100
    assert (-1e-6 < lrts[1:3]).all() and (lrts[1:3] < 20).all()
    assert np.isnan(lrts[3])
    assert np.isnan(res["gain_ratio_lrt_naive"][3])


def test_gain_ratio_lrts():
    # trait 0 gains at twice the rate of neutral traits 1-3
    res = _fit(
        [(2e-4, 0.0), (1e-4, 0.0), (1e-4, 0.0), (1e-4, 0.0)],
        neutral_trait_ids=[1, 2, 3],
    )
    lrts = res["gain_ratio_lrt_naive"].to_numpy()
    assert lrts[0] > 100
    assert (lrts[1:] < 20).all()
    # the constrained fit can never beat the unconstrained one
    assert (lrts > -1e-6).all()


def test_gain_ratio_undefined_without_reference_gains():
    # neutral traits 1 and 2 have no gains, so no trait has a reference
    res = _fit([(1e-4, 0.0), (0.0, 0.0), (0.0, 0.0)], neutral_trait_ids=[1, 2])
    assert res["gain_ratio"].is_nan().all()
    assert res["gain_ratio_lrt_naive"].is_nan().all()
    assert np.isfinite(res["selection_coef_lrt_naive"][0])

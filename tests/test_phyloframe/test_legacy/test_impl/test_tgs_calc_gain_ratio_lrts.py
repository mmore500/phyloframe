import numpy as np
import pandas as pd

from phyloframe.legacy._impl import tgs_calc_gain_ratio_lrts, tgs_fit
from phyloframe.legacy._impl._tgs_fit import _tgs_make_nll


def _simulate(gain_rate: float, seed: int):
    rng = np.random.default_rng(seed)
    lengths = rng.uniform(1, 60, 40)
    num_trials = np.full(40, 20_000.0)
    num_successes = rng.binomial(20_000, gain_rate * lengths).astype(float)
    return num_trials, num_successes, lengths


def _fit_traits(gain_rates):
    data = [_simulate(g, seed) for seed, g in enumerate(gain_rates)]
    fits = pd.DataFrame(
        [
            tgs_fit(num_trials=n, num_successes=k, pendant_length=x)._asdict()
            for n, k, x in data
        ],
    )
    nlls = [
        _tgs_make_nll(
            num_trials=n,
            num_successes=k,
            pendant_length=x,
            growth_exponent_bound=600.0,
        )
        for n, k, x in data
    ]
    return fits, [f for f, __ in nlls], [s for __, s in nlls]


def test_gain_ratio_lrts():
    # trait 0 gains at twice the rate of neutral traits 1-3
    fits, calc_nlls, scales = _fit_traits([2e-4, 1e-4, 1e-4, 1e-4])
    lrts = tgs_calc_gain_ratio_lrts(
        fits=fits,
        calc_nlls=calc_nlls,
        selection_scales=scales,
        neutral_traits=[1, 2, 3],
        loss_rate=0.0,
        selection_starts=(0.0, -0.3),
    )
    assert lrts[0] > 100
    assert (lrts[1:] < 20).all()
    # the constrained fit can never beat the unconstrained one
    assert (lrts > -1e-6).all()


def test_matches_free_fit_at_reference_rate():
    fits, calc_nlls, scales = _fit_traits([1e-4, 1e-4, 1e-4])
    # make trait 0's fitted gain rate exactly its reference's mean
    fits.loc[0, "gain_rate"] = fits.loc[[1, 2], "gain_rate"].mean()
    lrts = tgs_calc_gain_ratio_lrts(
        fits=fits,
        calc_nlls=calc_nlls,
        selection_scales=scales,
        neutral_traits=[1, 2],
        loss_rate=0.0,
        selection_starts=(0.0,),
    )
    # trait 0's "free" point now satisfies the null, which refits selection,
    # so the null can only fit as well or better
    assert lrts[0] <= 1e-6


def test_undefined_without_reference_or_gains():
    fits, calc_nlls, scales = _fit_traits([1e-4, 1e-4])
    lrts = tgs_calc_gain_ratio_lrts(
        fits=fits,
        calc_nlls=calc_nlls,
        selection_scales=scales,
        neutral_traits=[],
        loss_rate=0.0,
        selection_starts=(0.0,),
    )
    assert np.isnan(lrts).all()
    lrts = tgs_calc_gain_ratio_lrts(
        fits=fits,
        calc_nlls=[None, calc_nlls[1]],
        selection_scales=scales,
        neutral_traits=[1],
        loss_rate=0.0,
        selection_starts=(0.0,),
    )
    # trait 0 has no gains; trait 1 is the only neutral trait, so has no
    # reference
    assert np.isnan(lrts).all()

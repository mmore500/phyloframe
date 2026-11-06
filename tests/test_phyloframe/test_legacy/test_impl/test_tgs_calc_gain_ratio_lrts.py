import functools

import numpy as np
import pandas as pd

from phyloframe.legacy._impl import (
    tgs_calc_gain_ratio_lrts,
    tgs_calc_nll,
    tgs_fit,
    tgs_normalize_gain_rates_to_neutral,
)


def _simulate(gain_rate: float, seed: int):
    rng = np.random.default_rng(seed)
    lengths = rng.uniform(1, 60, 40)
    num_trials = np.full(40, 20_000.0)
    num_successes = rng.binomial(20_000, gain_rate * lengths).astype(float)
    return pd.DataFrame(
        {
            "num_trials": num_trials,
            "num_successes": num_successes,
            "length": lengths,
        },
    )


def _fit_traits(gain_rates):
    trait_partitions = [
        _simulate(g, seed) for seed, g in enumerate(gain_rates)
    ]
    fits = pd.DataFrame(
        [
            tgs_fit(
                num_trials=partitions["num_trials"],
                num_successes=partitions["num_successes"],
                pendant_length=partitions["length"],
            )._asdict()
            for partitions in trait_partitions
        ],
    )
    calc_nlls = [
        functools.partial(
            tgs_calc_nll, partitions=partitions, growth_exponent_bound=600.0
        )
        for partitions in trait_partitions
    ]
    return fits, trait_partitions, calc_nlls


def _lrts(fits, trait_partitions, calc_nlls, *, neutral_traits, **kwargs):
    fits = fits.assign(
        gain_ratio=tgs_normalize_gain_rates_to_neutral(
            gain_rates=fits["gain_rate"].to_numpy(),
            neutral_traits=neutral_traits,
        ),
    )
    return tgs_calc_gain_ratio_lrts(
        fits=fits,
        trait_partitions=trait_partitions,
        calc_nlls=calc_nlls,
        loss_rate=0.0,
        **kwargs,
    )


def test_gain_ratio_lrts():
    # trait 0 gains at twice the rate of neutral traits 1-3
    data = _fit_traits([2e-4, 1e-4, 1e-4, 1e-4])
    lrts = _lrts(*data, neutral_traits=[1, 2, 3], selection_starts=(0.0, -0.3))
    assert lrts[0] > 100
    assert (lrts[1:] < 20).all()
    # the constrained fit can never beat the unconstrained one
    assert (lrts > -1e-6).all()


def test_matches_free_fit_at_reference_rate():
    fits, trait_partitions, calc_nlls = _fit_traits([1e-4, 1e-4, 1e-4])
    # make trait 0's fitted gain rate exactly its reference's mean
    fits.loc[0, "gain_rate"] = fits.loc[[1, 2], "gain_rate"].mean()
    lrts = _lrts(
        fits,
        trait_partitions,
        calc_nlls,
        neutral_traits=[1, 2],
        selection_starts=(0.0,),
    )
    # trait 0's "free" point now satisfies the null, which refits selection,
    # so the null can only fit as well or better
    assert lrts[0] <= 1e-6


def test_undefined_without_reference_or_gains():
    fits, trait_partitions, calc_nlls = _fit_traits([1e-4, 1e-4])
    lrts = _lrts(
        fits,
        trait_partitions,
        calc_nlls,
        neutral_traits=[],
        selection_starts=(0.0,),
    )
    assert np.isnan(lrts).all()
    # without gains, a trait's fitted gain rate is 0
    lrts = _lrts(
        fits.assign(gain_rate=[0.0, fits.loc[1, "gain_rate"]]),
        trait_partitions,
        calc_nlls,
        neutral_traits=[1],
        selection_starts=(0.0,),
    )
    # trait 0 has no gains; trait 1 is the only neutral trait, so has no
    # reference
    assert np.isnan(lrts).all()

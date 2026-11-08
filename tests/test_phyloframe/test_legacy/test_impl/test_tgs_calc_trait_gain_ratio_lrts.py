import numpy as np
import polars as pl

from phyloframe.legacy._impl import (
    tgs_calc_neutral_reference_gain_rates,
    tgs_calc_trait_gain_ratio_lrts,
    tgs_fit,
)


def _simulate(gain_rate: float, seed: int):
    rng = np.random.default_rng(seed)
    lengths = rng.uniform(1, 60, 40)
    num_trials = np.full(40, 20_000.0)
    num_successes = rng.binomial(20_000, gain_rate * lengths).astype(float)
    return pl.DataFrame(
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
    fits = pl.DataFrame(
        [
            tgs_fit(
                num_trials=partitions["num_trials"],
                num_successes=partitions["num_successes"],
                pendant_length=partitions["length"],
            )._asdict()
            for partitions in trait_partitions
        ],
    )
    trait_nll_kwargs = [
        dict(partitions=partitions, growth_exponent_bound=600.0)
        for partitions in trait_partitions
    ]
    return fits, trait_nll_kwargs


def _lrts(fits, trait_nll_kwargs, *, neutral_traits, **kwargs):
    fits = fits.with_columns(
        gain_rate_neutral=tgs_calc_neutral_reference_gain_rates(
            gain_rates=fits["gain_rate"].to_numpy(),
            neutral_traits=neutral_traits,
        ),
    )
    res = tgs_calc_trait_gain_ratio_lrts(
        fits_df=fits,
        trait_nll_kwargs=trait_nll_kwargs,
        loss_rate=0.0,
        **kwargs,
    )
    assert res["trait"].to_list() == list(range(len(fits)))
    return res["gain_ratio_lrt_naive"].to_numpy()


def test_gain_ratio_lrts():
    # trait 0 gains at twice the rate of neutral traits 1-3
    data = _fit_traits([2e-4, 1e-4, 1e-4, 1e-4])
    lrts = _lrts(
        *data, neutral_traits=[1, 2, 3], selection_coef_starts=(0.0, -0.3)
    )
    assert lrts[0] > 100
    assert (lrts[1:] < 20).all()
    # the constrained fit can never beat the unconstrained one
    assert (lrts > -1e-6).all()


def test_matches_free_fit_at_reference_rate():
    fits, trait_nll_kwargs = _fit_traits([1e-4, 1e-4, 1e-4])
    # make trait 0's fitted gain rate exactly its reference's mean
    fits = fits.with_columns(
        gain_rate=pl.Series(
            [fits["gain_rate"][1:].mean(), *fits["gain_rate"][1:]]
        )
    )
    lrts = _lrts(
        fits,
        trait_nll_kwargs,
        neutral_traits=[1, 2],
        selection_coef_starts=(0.0,),
    )
    # trait 0's "free" point now satisfies the null, which refits the selection
    # coefficient, so the null can only fit as well or better
    assert lrts[0] <= 1e-6


def test_undefined_without_reference_or_gains():
    fits, trait_nll_kwargs = _fit_traits([1e-4, 1e-4])
    lrts = _lrts(
        fits,
        trait_nll_kwargs,
        neutral_traits=[],
        selection_coef_starts=(0.0,),
    )
    assert np.isnan(lrts).all()
    # without gains, a trait's fitted gain rate is 0
    lrts = _lrts(
        fits.with_columns(gain_rate=pl.Series([0.0, fits["gain_rate"][1]])),
        trait_nll_kwargs,
        neutral_traits=[1],
        selection_coef_starts=(0.0,),
    )
    # trait 0 has no gains; trait 1 is the only neutral trait, so has no
    # reference
    assert np.isnan(lrts).all()

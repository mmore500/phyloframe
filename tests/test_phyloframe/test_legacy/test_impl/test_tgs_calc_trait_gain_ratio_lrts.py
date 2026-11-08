import numpy as np
import polars as pl

from phyloframe.legacy._impl import (
    tgs_calc_neutral_reference_gain_rates,
    tgs_calc_trait_gain_ratio_lrts,
    tgs_fit_trait,
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
    trait_coarsened_dfs = [
        _simulate(g, seed) for seed, g in enumerate(gain_rates)
    ]
    trait_nll_kwargs = [
        dict(
            coarsened_df=coarsened_df,
            growth_exponent_bound=600.0,
            survival_shares=None,
            distinct_survival_durations=None,
        )
        for coarsened_df in trait_coarsened_dfs
    ]
    fits = pl.DataFrame(
        [
            tgs_fit_trait(
                nll_kwargs=nll_kwargs,
                loss_rate=0.0,
                selection_coef_starts=(0.0, -0.3, -1.8),
            )
            for nll_kwargs in trait_nll_kwargs
        ],
    )
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
    assert len(res) == len(fits)
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
    fits, trait_nll_kwargs = _fit_traits([1e-4, 1e-4, 1e-4])
    # without gains, a trait's fitted gain rate is 0
    lrts = _lrts(
        fits.with_columns(
            gain_rate=pl.Series([fits["gain_rate"][0], 0.0, 0.0])
        ),
        trait_nll_kwargs,
        neutral_traits=[1, 2],
        selection_coef_starts=(0.0,),
    )
    # traits 1 and 2 have no gains, so no trait has a reference
    assert np.isnan(lrts).all()
    lrts = _lrts(
        fits.with_columns(gain_rate=pl.Series([0.0, *fits["gain_rate"][1:]])),
        trait_nll_kwargs,
        neutral_traits=[1, 2],
        selection_coef_starts=(0.0,),
    )
    # trait 0 has no gains
    assert np.isnan(lrts[0])
    assert np.isfinite(lrts[1:]).all()

import numpy as np
import polars as pl

from phyloframe.legacy._impl import (
    tgs_calc_trait_selection_lrts,
    tgs_fit_trait,
)
from phyloframe.legacy._impl._tgs_calc_trait_selection_lrts import (
    _tgs_calc_trait_selection_lrt,
)


def _coarsened_df(gain_rate: float, selection_coef: float, seed: int):
    rng = np.random.default_rng(seed)
    length = np.linspace(0.5, 60.0, 120)
    num_trials = np.full_like(length, 20_000)
    p = (
        gain_rate * length
        if selection_coef == 0
        else gain_rate * np.expm1(selection_coef * length) / selection_coef
    )
    return pl.DataFrame(
        {
            "num_trials": num_trials,
            "num_successes": rng.binomial(num_trials.astype(int), p).astype(
                float
            ),
            "length": length,
        },
    )


def test_trait_selection_lrts():
    # trait 0 has a cost; trait 1 is neutral; trait 2 has no gains
    trait_nll_kwargs = [
        dict(
            coarsened_df=_coarsened_df(gain_rate, selection_coef, seed),
            growth_exponent_bound=600.0,
            survival_shares=None,
            distinct_survival_durations=None,
        )
        for seed, (gain_rate, selection_coef) in enumerate(
            [(2e-3, -0.05), (1e-3, 0.0), (0.0, 0.0)]
        )
    ]
    fits_df = pl.DataFrame(
        [
            tgs_fit_trait(
                nll_kwargs=kwargs,
                loss_rate=0.0,
                selection_coef_starts=(0.0, -0.3, -1.8),
            )
            for kwargs in trait_nll_kwargs
        ],
    )
    res = tgs_calc_trait_selection_lrts(
        fits_df=fits_df, trait_nll_kwargs=trait_nll_kwargs, loss_rate=0.0
    )
    assert res.columns == ["selection_coef_lrt_naive"]
    assert len(res) == 3
    lrts = res["selection_coef_lrt_naive"].to_numpy()
    # matches each trait's statistic alone
    np.testing.assert_allclose(
        lrts,
        [
            _tgs_calc_trait_selection_lrt(
                nll_kwargs=kwargs, loss_rate=0.0, **fit
            )
            for kwargs, fit in zip(
                trait_nll_kwargs, fits_df.iter_rows(named=True)
            )
        ],
    )
    assert lrts[0] > 100
    assert -1e-6 < lrts[1] < 20
    assert np.isnan(lrts[2])


def test_no_gains_is_nan():
    lrt = _tgs_calc_trait_selection_lrt(
        nll_kwargs=dict(
            coarsened_df=_coarsened_df(0.0, 0.0, 1),
            growth_exponent_bound=600.0,
            survival_shares=None,
            distinct_survival_durations=None,
        ),
        gain_rate=0.0,
        selection_coef=np.nan,
        loss_rate=0.0,
    )
    assert np.isnan(lrt)

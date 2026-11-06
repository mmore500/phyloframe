import numpy as np
import polars as pl
import pytest

from phyloframe.legacy._impl import (
    tgs_calc_selection_lrt,
    tgs_calc_trait_selection_lrts,
    tgs_fit,
)


def _partitions(gain_rate: float, selection_coef: float, seed: int):
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
            partitions=_partitions(gain_rate, selection_coef, seed),
            growth_exponent_bound=600.0,
        )
        for seed, (gain_rate, selection_coef) in enumerate(
            [(2e-3, -0.05), (1e-3, 0.0), (0.0, 0.0)]
        )
    ]
    with pytest.warns(UserWarning, match="no trait carriers"):
        fits_df = pl.DataFrame(
            [
                tgs_fit(
                    num_trials=kwargs["partitions"]["num_trials"],
                    num_successes=kwargs["partitions"]["num_successes"],
                    pendant_length=kwargs["partitions"]["length"],
                )._asdict()
                for kwargs in trait_nll_kwargs
            ],
        )
    res = tgs_calc_trait_selection_lrts(
        fits_df=fits_df, trait_nll_kwargs=trait_nll_kwargs, loss_rate=0.0
    )
    assert res.columns == ["trait", "selection_coef_lrt_naive"]
    assert res["trait"].to_list() == [0, 1, 2]
    lrts = res["selection_coef_lrt_naive"].to_numpy()
    # matches tgs_fit's own statistic
    np.testing.assert_allclose(
        lrts, fits_df["selection_coef_lrt_naive"].to_numpy()
    )
    assert lrts[0] > 100
    assert -1e-6 < lrts[1] < 20
    assert np.isnan(lrts[2])


def test_no_gains_is_nan():
    lrt = tgs_calc_selection_lrt(
        nll_kwargs=dict(partitions=_partitions(0.0, 0.0, 1)),
        gain_rate=0.0,
        selection_coef=np.nan,
        loss_rate=0.0,
    )
    assert np.isnan(lrt)

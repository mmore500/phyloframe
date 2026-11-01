import numpy as np
import pytest

from phyloframe._auxlib import identity
from phyloframe.legacy._impl import tgs_fit_allmdc_traits

from ._simulate_allmdc_groups import SHIFT, simulate_allmdc_groups

_KWARGS = dict(
    neutral_traits=[1, 2],
    calibrate=False,
    loss_rate=0.0,
    max_shift=100.0,
    num_traits=3,
    selection_starts=(0.0, -0.3, -1.8),
    max_growth_exponent=600.0,
    max_shift_iterations=50,
)


def test_recovers():
    groups = simulate_allmdc_groups()
    calls = []

    def progress_wrap(iterable):
        calls.append(1)
        return iterable

    res, shifts, params = tgs_fit_allmdc_traits(
        groups=groups,
        unit_weights=[np.ones(len(g.partitions)) for g in groups],
        starts=None,
        shifts_start=None,
        progress_wrap=progress_wrap,
        **_KWARGS,
    )
    assert calls == [1]
    assert list(res.columns) == [
        "gain_rate",
        "selection",
        "selection_lrt_naive",
        "gain_ratio",
    ]
    assert res.loc[0, "gain_ratio"] == pytest.approx(2.0, rel=0.2)
    assert res.loc[0, "selection"] == pytest.approx(-0.05, abs=0.015)
    assert shifts[0] == 0.0
    assert shifts[1] == pytest.approx(SHIFT, abs=6.0)
    assert params.shape == (3, 2)


def test_group_without_gains():
    groups = simulate_allmdc_groups((4_000, 2_000))
    empty = groups[1]._replace(trait_gains=[np.array([], dtype=int)] * 3)
    res, shifts, __ = tgs_fit_allmdc_traits(
        groups=[groups[0], empty],
        unit_weights=[np.ones(len(g.partitions)) for g in (groups[0], empty)],
        starts=None,
        shifts_start=None,
        progress_wrap=identity,
        **_KWARGS,
    )
    assert np.isnan(shifts[1])
    assert np.isfinite(res["gain_rate"]).all()


def test_trait_without_gains():
    groups = simulate_allmdc_groups((4_000, 2_000))
    groups = [
        g._replace(trait_gains=[np.array([], dtype=int)] + g.trait_gains[1:])
        for g in groups
    ]
    res, __, params = tgs_fit_allmdc_traits(
        groups=groups,
        unit_weights=[np.ones(len(g.partitions)) for g in groups],
        starts=None,
        shifts_start=None,
        progress_wrap=identity,
        **_KWARGS,
    )
    assert res.loc[0, "gain_rate"] == 0.0
    assert np.isnan(res.loc[0, "selection"])
    assert np.isnan(params[0]).all()

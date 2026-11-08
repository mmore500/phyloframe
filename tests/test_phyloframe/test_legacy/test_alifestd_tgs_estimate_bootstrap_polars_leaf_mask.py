import typing
import warnings

import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pltest
import pytest

from phyloframe._auxlib._ExperimentalWarning import ExperimentalWarning
from phyloframe.legacy import alifestd_tgs_estimate_bootstrap_polars

from .test_alifestd_tgs_estimate_polars_leaf_mask import (
    _estimate_from_masks,
    _estimate_leaves,
    _make_cherries,
    _make_cherries_polars,
    _with_leaf_mask,
    apply_params,
)


def _estimate_leaves_bootstrap(phylogeny_df, **kwargs):
    """Bootstrap-estimate from leaf pendant branches only."""
    return alifestd_tgs_estimate_bootstrap_polars(
        _with_leaf_mask(phylogeny_df), episode_mask="is_leaf", **kwargs
    )


def _estimate_bootstrap_from_masks(
    phylogeny_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    return _estimate_from_masks(
        phylogeny_df,
        estimator=_estimate_leaves_bootstrap,
        **kwargs,
    )


def test_point_columns_match_point_estimate():
    df_pl = _make_cherries_polars(2_000)
    kwargs = dict(
        trait_transitions=["focal", "n1", "n2", "n3", "n4"],
        neutral_traits=["n1", "n2", "n3", "n4"],
    )
    point = _estimate_leaves(df_pl, **kwargs)
    res = _estimate_leaves_bootstrap(df_pl, num_bootstrap=2, seed=1, **kwargs)
    assert res.columns[: point.width] == point.columns
    pltest.assert_frame_equal(res.select(point.columns), point)


@pytest.mark.parametrize("num_bootstrap", [0, -1])
def test_num_bootstrap_not_positive_raises(num_bootstrap: int):
    with pytest.raises(ValueError, match="num_bootstrap"):
        _estimate_leaves_bootstrap(
            _make_cherries_polars(100),
            trait_transitions=["focal", "n1", "n2"],
            neutral_traits=["n1", "n2"],
            num_bootstrap=num_bootstrap,
        )


@pytest.mark.parametrize("calibrate", [False, True])
def test_recovers_gain_selection(calibrate: bool):
    phylogeny_df, traits = _make_cherries()
    res = _estimate_bootstrap_from_masks(
        phylogeny_df,
        trait_masks=traits,
        trait_names=["focal", "n1", "n2", "n3", "n4"],
        neutral_traits=[1, 2, 3, 4],
        calibrate=calibrate,
        num_bootstrap=20,
        seed=1,
    )
    assert res["trait"].tolist() == ["focal", "n1", "n2", "n3", "n4"]
    assert (res["num_leaf_episodes"] <= 80_000).all()

    focal = res.iloc[0]
    assert focal["gain_ratio"] == pytest.approx(2.0, rel=0.2)
    assert focal["selection_coef"] == pytest.approx(-0.05, abs=0.02)
    assert focal["selection_coef_p"] < 1e-3
    assert focal["gain_ratio_p"] < 1e-3

    neutral = res.iloc[1:]
    # gain rate trades off against the selection coefficient, so with ~2k
    # carriers per trait, neutral gain ratios scatter by ~20%
    assert neutral["gain_ratio"].tolist() == pytest.approx([1.0] * 4, rel=0.3)
    assert (neutral["selection_coef"].abs() < 0.02).all()
    assert np.isfinite(res["selection_coef_se"]).all()
    assert np.isfinite(res["log_gain_ratio_se"]).all()


@pytest.mark.parametrize("parallel_backend", ["threading", "loky"])
def test_parallel_backend_matches_serial(parallel_backend: str):
    phylogeny_df, traits = _make_cherries(2_000)
    kwargs = dict(
        trait_masks=traits,
        neutral_traits=[1, 2, 3, 4],
        num_bootstrap=4,
        seed=1,
    )
    expected = _estimate_bootstrap_from_masks(phylogeny_df, **kwargs)
    res = _estimate_bootstrap_from_masks(
        phylogeny_df, parallel_backend=parallel_backend, **kwargs
    )
    pd.testing.assert_frame_equal(res, expected)


def test_seed_reproducible():
    phylogeny_df, traits = _make_cherries(2_000)

    def run(seed):
        return _estimate_bootstrap_from_masks(
            phylogeny_df,
            trait_masks=traits,
            num_bootstrap=4,
            seed=seed,
        )

    np.random.seed(5)
    expected_state = np.random.get_state()[1].copy()
    res1, res2, res3 = run(1), run(1), run(2)
    # seeded runs leave global rng state untouched
    np.testing.assert_array_equal(np.random.get_state()[1], expected_state)
    pd.testing.assert_frame_equal(res1, res2)
    assert not res1["selection_coef_se"].equals(res3["selection_coef_se"])


def test_bootstrap_without_degenerate_replicates():
    phylogeny_df, traits = _make_cherries(2_000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no degenerate-replicate warning
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        res = _estimate_bootstrap_from_masks(
            phylogeny_df,
            trait_masks=traits,
            num_bootstrap=4,
            seed=1,
        )
    assert np.isfinite(res["log_gain_ratio_se"]).all()
    assert np.isfinite(res["selection_coef_se"]).all()
    assert (res["num_bootstrap_degenerate"] == 0).all()


def test_bootstrap_rare_trait():
    phylogeny_df, traits = _make_cherries(20_000)
    rare = np.zeros_like(traits[0])
    rare[np.flatnonzero(traits[1])[0]] = True
    with pytest.warns(UserWarning, match="bootstrap") as record:
        res = _estimate_bootstrap_from_masks(
            phylogeny_df,
            trait_masks=[rare, *traits[1:]],
            neutral_traits=[1, 2, 3, 4],
            num_bootstrap=50,
            seed=1,
        )
    # with a single carrier, replicates often resample no carriers
    num_degenerate = res.loc[0, "num_bootstrap_degenerate"]
    assert 0 < num_degenerate < 50
    assert any(
        f"0 ({num_degenerate} of 50)" in str(warning.message)
        for warning in record
    )
    assert (res.loc[1:, "num_bootstrap_degenerate"] == 0).all()
    assert np.isfinite(res["selection_coef_se"]).all()
    assert np.isfinite(res["log_gain_ratio_se"]).all()


def test_parallel_backend_matches_serial_many_replicates():
    # more replicates than joblib dispatches up front, so later replicates
    # are dispatched from joblib's callback thread
    phylogeny_df, traits = _make_cherries(500)
    kwargs = dict(
        trait_masks=traits,
        neutral_traits=[1, 2, 3, 4],
        num_bootstrap=96,
        seed=1,
    )
    expected = _estimate_bootstrap_from_masks(phylogeny_df, **kwargs)
    for backend in ("threading", "loky"):
        res = _estimate_bootstrap_from_masks(
            phylogeny_df, parallel_backend=backend, **kwargs
        )
        pd.testing.assert_frame_equal(res, expected)


def test_bootstrap_neutral_reference_without_carriers():
    # neutral traits with one carrier each, so some replicates resample no
    # carriers of the focal trait's whole neutral reference
    phylogeny_df, traits = _make_cherries(20_000)
    rare = np.zeros_like(traits[1:3])
    rare[0, np.flatnonzero(traits[1])[:1]] = True
    rare[1, np.flatnonzero(traits[2])[:1]] = True
    with pytest.warns(UserWarning, match="neutral reference"):
        res = _estimate_bootstrap_from_masks(
            phylogeny_df,
            trait_masks=[traits[0], *rare],
            neutral_traits=[1, 2],
            num_bootstrap=50,
            seed=1,
        )
    # the focal trait always has carriers, but its reference sometimes not
    assert np.isfinite(res.loc[0, "selection_coef_se"])
    assert 0 < res.loc[0, "num_bootstrap_degenerate"] < 50
    assert np.isfinite(res.loc[0, "log_gain_ratio_se"])


@apply_params
def test_empty(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Int64),
                "ancestor_id": pl.Series([], dtype=pl.Int64),
                "origin_time": pl.Series([], dtype=pl.Float64),
                **{t: pl.Series([], dtype=pl.Int8) for t in ("a", "b", "c")},
            }
        ),
    )
    with pytest.raises(ValueError, match="no exposure episodes"):
        _estimate_leaves_bootstrap(
            df_pl,
            trait_transitions=["a", "b", "c"],
            neutral_traits=["b", "c"],
            num_bootstrap=2,
        )

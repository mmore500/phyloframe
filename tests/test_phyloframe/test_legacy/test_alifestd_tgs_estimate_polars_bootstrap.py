import typing

import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pltest
import pytest

from phyloframe.legacy import alifestd_tgs_estimate_polars

from .test_alifestd_tgs_estimate_polars import (
    _estimate_from_masks,
    _make_cherries,
    _make_cherries_polars,
    apply_params,
)


def _estimate_bootstrap_from_masks(
    phylogeny_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    return _estimate_from_masks(
        phylogeny_df,
        estimator=alifestd_tgs_estimate_polars,
        **kwargs,
    )


def test_point_columns_match_point_estimate():
    df_pl = _make_cherries_polars(2_000)
    kwargs = dict(
        trait_transitions=["focal", "n1", "n2", "n3", "n4"],
        neutral_traits=["n1", "n2", "n3", "n4"],
    )
    point = alifestd_tgs_estimate_polars(df_pl, **kwargs)
    res = alifestd_tgs_estimate_polars(
        df_pl, num_bootstrap=2, seed=1, **kwargs
    )
    assert res.columns[: point.width] == point.columns
    pltest.assert_frame_equal(res.select(point.columns), point)


def test_seed_reproducible():
    phylogeny_df, traits = _make_cherries(2_000)

    def run(seed):
        return _estimate_bootstrap_from_masks(
            phylogeny_df, trait_masks=traits, num_bootstrap=4, seed=seed
        )

    res1, res2, res3 = run(1), run(1), run(2)
    pd.testing.assert_frame_equal(res1, res2)
    assert not res1["selection_coef_se"].equals(res3["selection_coef_se"])


@pytest.mark.parametrize("calibrate", [False, True])
def test_recovers_gain_selection(calibrate: bool):
    phylogeny_df, traits = _make_cherries()
    res = _estimate_bootstrap_from_masks(
        phylogeny_df,
        trait_masks=traits,
        trait_names=["focal", "n1", "n2", "n3", "n4"],
        neutral_traits=[1, 2, 3, 4],
        calibrate=calibrate,
        num_bootstrap=10,
        seed=1,
    )
    assert res["trait"].tolist() == ["focal", "n1", "n2", "n3", "n4"]
    assert (res["num_inner_episodes"] == 40_000).all()
    assert (res["num_inner_gains"] > 0).all()

    focal = res.iloc[0]
    assert focal["gain_ratio"] == pytest.approx(2.0, rel=0.25)
    assert focal["selection_coef"] == pytest.approx(-0.05, abs=0.02)
    assert focal["gain_ratio_p"] < 1e-3
    assert focal["selection_coef_p"] < 1e-3
    assert np.isfinite(res["selection_coef_se"]).all()
    assert not any("shift" in c for c in res.columns)

    neutral = res.iloc[1:]
    assert neutral["gain_ratio"].tolist() == pytest.approx([1.0] * 4, rel=0.3)
    assert (neutral["selection_coef"].abs() < 0.02).all()


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
        alifestd_tgs_estimate_polars(
            df_pl,
            trait_transitions=["a", "b", "c"],
            neutral_traits=["b", "c"],
            num_bootstrap=2,
        )


def test_unknown_kwarg_raises():
    with pytest.raises(TypeError):
        alifestd_tgs_estimate_polars(
            _make_cherries_polars(100),
            trait_transitions=["focal", "n1", "n2"],
            neutral_traits=["n1", "n2"],
            num_bootstrap=2,
            not_an_argument=1,
        )

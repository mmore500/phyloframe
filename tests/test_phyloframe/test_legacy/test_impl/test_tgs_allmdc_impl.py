import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe._auxlib import identity
from phyloframe.legacy import alifestd_tgs_allmdc_estimate_polars
from phyloframe.legacy._impl import (
    tgs_allmdc_impl,
    tgs_build_exposure_df_polars,
    tgs_calc_exposure_df,
)

from .._impl import add_trait_transitions
from ..test_alifestd_tgs_allmdc_estimate_polars import _make_cherries

_KWARGS = dict(
    calibrate=True,
    calibrate_neighborhood_fractions=(0.25, 1.0),
    calibrate_num_partitions=64,
    calibrate_max_rate_ratio=np.inf,
    calibrate_size_partition_ratio=2.0,
    calibrate_size_partition_min_neutral_gains=1,
    num_bootstrap=2,
    loss_rate=0.0,
    seed=1,
    parallel_backend=None,
)
_IMPL_KWARGS = dict(
    selection_starts=(0.0, -0.3, -1.8),
    growth_exponent_bound=600.0,
    progress_wrap=identity,
)


def test_matches_public():
    phylogeny_df, traits = _make_cherries(2_000)
    phylogeny_df, names = add_trait_transitions(phylogeny_df, traits)
    df_pl = pl.from_pandas(phylogeny_df)
    exposure_df = tgs_build_exposure_df_polars(
        df_pl, trait_transitions=names, caller="test"
    )
    res = tgs_allmdc_impl(
        exposure_df,
        trait_names=names,
        neutral_traits=[1, 2, 3, 4],
        **_IMPL_KWARGS,
        **_KWARGS,
    )
    expected = alifestd_tgs_allmdc_estimate_polars(
        df_pl,
        trait_transitions=names,
        neutral_traits=names[1:],
        **_KWARGS,
    )
    pd.testing.assert_frame_equal(res, expected.to_pandas(), check_dtype=False)
    assert "selection_se" in res.columns


def _no_branches() -> pd.DataFrame:
    return tgs_calc_exposure_df(
        ancestor_ids=np.array([0]),
        origin_time=np.array([0.0]),
        trait_states=[np.array([0])],
    )


def test_no_branches_raises():
    with pytest.raises(ValueError, match="no exposure episodes"):
        tgs_allmdc_impl(
            _no_branches(),
            trait_names=["t"],
            neutral_traits=[],
            **_IMPL_KWARGS,
            **{**_KWARGS, "calibrate": False},
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(calibrate_size_partition_ratio=1.0),
        dict(calibrate_size_partition_min_neutral_gains=0),
    ],
)
def test_invalid(kwargs: dict):
    with pytest.raises(ValueError):
        tgs_allmdc_impl(
            _no_branches(),
            trait_names=["t"],
            neutral_traits=[],
            **_IMPL_KWARGS,
            **{**_KWARGS, **kwargs},
        )

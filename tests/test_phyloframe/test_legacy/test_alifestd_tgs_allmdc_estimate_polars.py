import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_tgs_allmdc_estimate_asexual,
    alifestd_tgs_allmdc_estimate_polars,
)

from .test_alifestd_tgs_allmdc_estimate_asexual import _make_cherries

apply_params = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


def _make_cherries_polars(num_cherries: int) -> pl.DataFrame:
    phylogeny_df, traits = _make_cherries(num_cherries)
    return pl.from_pandas(phylogeny_df).with_columns(
        *(
            pl.Series(name, trait)
            for name, trait in zip(["focal", "n1", "n2", "n3", "n4"], traits)
        ),
    )


@apply_params
def test_empty(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Int64),
                "ancestor_id": pl.Series([], dtype=pl.Int64),
                "origin_time": pl.Series([], dtype=pl.Float64),
                "trait": pl.Series([], dtype=pl.Boolean),
            }
        ),
    )
    res = alifestd_tgs_allmdc_estimate_polars(
        df_pl, trait_masks=["trait"], num_bootstrap=2
    )
    assert res["trait"].to_list() == ["trait"]
    assert res["num_leaves"].to_list() == [0]
    assert res["num_clades"].to_list() == [0]
    assert "selection_p" in res.columns


@apply_params
@pytest.mark.parametrize("calibrate", [False, True])
def test_matches_pandas(apply: typing.Callable, calibrate: bool):
    df_pl = _make_cherries_polars(5_000)
    names = ["focal", "n1", "n2", "n3", "n4"]
    res = alifestd_tgs_allmdc_estimate_polars(
        apply(df_pl),
        trait_masks=names,
        neutral_traits=["n1", "n2", 3, 4],
        calibrate=calibrate,
        num_bootstrap=3,
        seed=1,
    )
    df_pd = df_pl.to_pandas()
    expected = alifestd_tgs_allmdc_estimate_asexual(
        df_pd,
        trait_masks=[df_pd[name].to_numpy() for name in names],
        trait_names=names,
        neutral_traits=[1, 2, 3, 4],
        calibrate=calibrate,
        num_bootstrap=3,
        seed=1,
    )
    pd.testing.assert_frame_equal(
        res.to_pandas(), expected, check_dtype=False, rtol=1e-6
    )


def test_expression_masks():
    df_pl = _make_cherries_polars(2_000)
    res = alifestd_tgs_allmdc_estimate_polars(
        df_pl,
        trait_masks=[pl.col("focal"), pl.col("n1") & pl.col("n2")],
        trait_names=["focal", "both"],
    )
    assert res["trait"].to_list() == ["focal", "both"]


def test_not_asexual():
    df_pl = pl.DataFrame(
        {
            "id": [0, 1],
            "ancestor_list": ["[none]", "[0]"],
            "origin_time": [0.0, 1.0],
            "trait": [False, True],
        },
    ).with_columns(ancestor_list=pl.lit("[0, 1]"))
    with pytest.raises(NotImplementedError):
        alifestd_tgs_allmdc_estimate_polars(df_pl, trait_masks=["trait"])

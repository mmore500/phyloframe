import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_tgs_pendantmdc_estimate_asexual,
    alifestd_tgs_pendantmdc_estimate_polars,
    alifestd_to_working_format,
)

from ._impl import add_trait_transitions
from .test_alifestd_tgs_pendantmdc_estimate_asexual import _make_cherries

apply_params = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


def _make_cherries_polars(num_cherries: int) -> pl.DataFrame:
    phylogeny_df, traits = _make_cherries(num_cherries)
    phylogeny_df, names = add_trait_transitions(phylogeny_df, traits)
    return pl.from_pandas(
        phylogeny_df.rename(
            columns=dict(zip(names, ["focal", "n1", "n2", "n3", "n4"]))
        ).drop(columns=["ancestor_list"], errors="ignore"),
    )


@apply_params
def test_empty(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Int64),
                "ancestor_id": pl.Series([], dtype=pl.Int64),
                "origin_time": pl.Series([], dtype=pl.Float64),
                "trait": pl.Series([], dtype=pl.Int8),
            }
        ),
    )
    res = alifestd_tgs_pendantmdc_estimate_polars(
        df_pl, trait_transitions=["trait"], num_bootstrap=2
    )
    assert res["trait"].to_list() == ["trait"]
    assert res["num_leaves"].to_list() == [0]
    assert "selection_p" in res.columns


@apply_params
@pytest.mark.parametrize("calibrate", [False, True])
def test_matches_pandas(apply: typing.Callable, calibrate: bool):
    df_pl = _make_cherries_polars(5_000)
    names = ["focal", "n1", "n2", "n3", "n4"]
    res = alifestd_tgs_pendantmdc_estimate_polars(
        apply(df_pl),
        trait_transitions=names,
        neutral_traits=["n1", "n2", 3, 4],
        calibrate=calibrate,
        num_bootstrap=5,
        seed=1,
    )
    df_pd = df_pl.to_pandas()
    expected = alifestd_tgs_pendantmdc_estimate_asexual(
        df_pd,
        trait_transitions=names,
        neutral_traits=[1, 2, 3, 4],
        calibrate=calibrate,
        num_bootstrap=5,
        seed=1,
    )
    pd.testing.assert_frame_equal(res.to_pandas(), expected, check_dtype=False)


def test_expression_transitions_raise():
    df_pl = _make_cherries_polars(500)
    with pytest.raises(TypeError):
        alifestd_tgs_pendantmdc_estimate_polars(
            df_pl, trait_transitions=["focal", pl.col("n1")]
        )


def test_calibrate_requires_neutral():
    with pytest.raises(ValueError):
        alifestd_tgs_pendantmdc_estimate_polars(
            _make_cherries_polars(100),
            trait_transitions=["focal", "n1"],
            neutral_traits=["n1"],
            calibrate=True,
        )


@pytest.mark.parametrize(
    "df_pl",
    [
        pl.DataFrame(
            {"id": [1, 0], "ancestor_id": [0, 0], "origin_time": [0.0, 1.0]}
        ),
        pl.DataFrame(
            {"id": [0, 2], "ancestor_id": [0, 0], "origin_time": [0.0, 1.0]}
        ),
    ],
)
def test_unsupported(df_pl: pl.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_tgs_pendantmdc_estimate_polars(
            df_pl.with_columns(trait=pl.lit(0)), trait_transitions=["trait"]
        )


def test_null_transitions_raise():
    df_pl = _make_cherries_polars(2_000).with_columns(
        focal=pl.when(pl.col("id") == 5).then(None).otherwise(pl.col("focal"))
    )
    with pytest.raises(NotImplementedError):
        alifestd_tgs_pendantmdc_estimate_polars(
            df_pl, trait_transitions=["focal"]
        )


def test_ancestor_list_column():
    df_pl = _make_cherries_polars(2_000)
    expected = alifestd_tgs_pendantmdc_estimate_polars(
        df_pl,
        trait_transitions=["focal", "n1", "n2"],
        neutral_traits=["n1", "n2"],
    )
    with_list = df_pl.with_columns(
        ancestor_list=pl.when(pl.col("id") == pl.col("ancestor_id"))
        .then(pl.lit("[none]"))
        .otherwise(pl.format("[{}]", pl.col("ancestor_id"))),
    )
    res = alifestd_tgs_pendantmdc_estimate_polars(
        with_list,
        trait_transitions=["focal", "n1", "n2"],
        neutral_traits=["n1", "n2"],
    )
    assert res.equals(expected)


def test_unifurcation_transitions_credit_merged_branch():
    phylogeny_df, traits = _make_cherries(2_000)
    phylogeny_df, names = add_trait_transitions(phylogeny_df, traits[:1])
    # insert a knuckle midway along each pendant branch
    leaves = phylogeny_df.index[
        ~phylogeny_df["id"].isin(phylogeny_df["ancestor_id"])
    ]
    knuckles = phylogeny_df.loc[leaves].copy()
    knuckles["id"] = np.arange(len(knuckles)) + len(phylogeny_df)
    knuckles["origin_time"] = (
        knuckles["origin_time"]
        + phylogeny_df.loc[knuckles["ancestor_id"], "origin_time"].to_numpy()
    ) / 2
    knuckled = phylogeny_df.copy()
    knuckled.loc[leaves, "ancestor_id"] = knuckles["id"].to_numpy()
    # a gain on each pendant branch, recorded below or above the knuckle
    below = pd.concat(
        [knuckled, knuckles.assign(**{names[0]: 0})], ignore_index=True
    )
    above = pd.concat(
        [knuckled.assign(**{names[0]: 0}), knuckles], ignore_index=True
    )
    kwargs = dict(trait_transitions=names)
    res_below = alifestd_tgs_pendantmdc_estimate_polars(
        pl.from_pandas(alifestd_to_working_format(below)).drop(
            "ancestor_list", strict=False
        ),
        **kwargs,
    )
    res_above = alifestd_tgs_pendantmdc_estimate_polars(
        pl.from_pandas(alifestd_to_working_format(above)).drop(
            "ancestor_list", strict=False
        ),
        **kwargs,
    )
    assert res_below.equals(res_above)
    assert res_below["num_trait_present"][0] == traits[0].sum()


def test_trait_names_length_mismatch_raises():
    with pytest.raises(ValueError):
        alifestd_tgs_pendantmdc_estimate_polars(
            _make_cherries_polars(100),
            trait_transitions=["focal", "n1"],
            trait_names=["focal"],
        )

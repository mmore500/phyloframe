import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_estimate_trait_gain_selection_pendant_asexual,
    alifestd_estimate_trait_gain_selection_pendant_polars,
)

from .test_alifestd_estimate_trait_gain_selection_pendant_asexual import (
    _make_cherries,
)

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
    res = alifestd_estimate_trait_gain_selection_pendant_polars(
        df_pl, trait_masks=["trait"], num_bootstrap=2
    )
    assert res["trait"].to_list() == ["trait"]
    assert res["num_leaves"].to_list() == [0]
    assert "selection_p" in res.columns


@apply_params
@pytest.mark.parametrize("calibrate", [False, True])
def test_matches_pandas(apply: typing.Callable, calibrate: bool):
    df_pl = _make_cherries_polars(5_000)
    names = ["focal", "n1", "n2", "n3", "n4"]
    res = alifestd_estimate_trait_gain_selection_pendant_polars(
        apply(df_pl),
        trait_masks=names,
        neutral_traits=["n1", "n2", 3, 4],
        calibrate=calibrate,
        num_bootstrap=5,
        seed=1,
    )
    df_pd = df_pl.to_pandas()
    expected = alifestd_estimate_trait_gain_selection_pendant_asexual(
        df_pd,
        trait_masks=[df_pd[name].to_numpy() for name in names],
        trait_names=names,
        neutral_traits=[1, 2, 3, 4],
        calibrate=calibrate,
        num_bootstrap=5,
        seed=1,
    )
    pd.testing.assert_frame_equal(res.to_pandas(), expected, check_dtype=False)


def test_trait_expr_names():
    df_pl = _make_cherries_polars(500)
    res = alifestd_estimate_trait_gain_selection_pendant_polars(
        df_pl, trait_masks=["focal", pl.col("n1") & pl.col("n2")]
    )
    assert res["trait"].to_list() == ["focal", "1"]
    assert np.isnan(res["gain_ratio"].to_numpy()).all()


def test_calibrate_requires_neutral():
    with pytest.raises(ValueError):
        alifestd_estimate_trait_gain_selection_pendant_polars(
            _make_cherries_polars(100),
            trait_masks=["focal", "n1"],
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
        alifestd_estimate_trait_gain_selection_pendant_polars(
            df_pl.with_columns(trait=pl.lit(True)), trait_masks=["trait"]
        )


def test_null_trait_mask_raises():
    df_pl = _make_cherries_polars(2_000)
    nullable = pl.when(pl.col("focal")).then(True).otherwise(None)
    with pytest.raises(NotImplementedError):
        alifestd_estimate_trait_gain_selection_pendant_polars(
            df_pl, trait_masks=[nullable], trait_names=["focal"]
        )


def test_ancestor_list_column():
    df_pl = _make_cherries_polars(2_000)
    expected = alifestd_estimate_trait_gain_selection_pendant_polars(
        df_pl, trait_masks=["focal", "n1", "n2"], neutral_traits=["n1", "n2"]
    )
    with_list = df_pl.with_columns(
        ancestor_list=pl.when(pl.col("id") == pl.col("ancestor_id"))
        .then(pl.lit("[none]"))
        .otherwise(pl.format("[{}]", pl.col("ancestor_id"))),
    )
    res = alifestd_estimate_trait_gain_selection_pendant_polars(
        with_list,
        trait_masks=["focal", "n1", "n2"],
        neutral_traits=["n1", "n2"],
    )
    assert res.equals(expected)


def test_id_based_mask_with_unifurcations():
    from phyloframe.legacy import alifestd_to_working_format

    phylogeny_df, traits = _make_cherries(2_000)
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
    knuckled = phylogeny_df.assign(focal=traits[0])
    knuckled.loc[leaves, "ancestor_id"] = knuckles["id"].to_numpy()
    knuckled = pd.concat(
        [knuckled, knuckles.assign(focal=False)], ignore_index=True
    )
    knuckled = alifestd_to_working_format(knuckled).drop(
        columns=["ancestor_list"], errors="ignore"
    )
    df_pl = pl.from_pandas(knuckled)

    expected = alifestd_estimate_trait_gain_selection_pendant_polars(
        df_pl, trait_masks=["focal"]
    )
    carrier_ids = knuckled.loc[knuckled["focal"], "id"].tolist()
    res = alifestd_estimate_trait_gain_selection_pendant_polars(
        df_pl,
        trait_masks=[pl.col("id").is_in(carrier_ids)],
        trait_names=["focal"],
    )
    assert res.equals(expected)
    assert res["num_trait_present"][0] == len(carrier_ids)


def test_trait_names_length_mismatch_raises():
    with pytest.raises(ValueError):
        alifestd_estimate_trait_gain_selection_pendant_polars(
            _make_cherries_polars(100),
            trait_masks=["focal", "n1"],
            trait_names=["focal"],
        )

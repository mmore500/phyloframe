import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mask_leaf_trait_pendant_gain_asexual,
    alifestd_mask_leaf_trait_pendant_gain_polars,
    alifestd_to_working_format,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")

apply_params = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


@apply_params
def test_empty(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Int64),
                "ancestor_id": pl.Series([], dtype=pl.Int64),
                "trait": pl.Series([], dtype=pl.Boolean),
            }
        ),
    )
    res = (
        alifestd_mask_leaf_trait_pendant_gain_polars(df_pl, trait_mask="trait")
        .lazy()
        .collect()
    )
    assert res.is_empty()
    assert "alifestd_mask_leaf_trait_pendant_gain_polars" in res.columns


@apply_params
def test_simple_tree(apply: typing.Callable):
    #        0
    #     1     2
    #    3 4   5 6
    #   7 8
    df_pl = apply(
        pl.DataFrame(
            {
                "id": range(9),
                "ancestor_id": [0, 0, 0, 1, 1, 2, 2, 3, 3],
                "trait": [0, 0, 0, 0, 1, 1, 0, 1, 1],
            }
        ),
    )
    res = (
        alifestd_mask_leaf_trait_pendant_gain_polars(
            df_pl, trait_mask=pl.col("trait") == 1, mark_as="m"
        )
        .lazy()
        .collect()
    )
    assert res["m"].to_list() == list(map(bool, [0, 0, 0, 0, 0, 1, 0, 0, 0]))


def test_matches_pandas():
    df = alifestd_to_working_format(
        pd.read_csv(f"{assets_path}/nk_ecoeaselection-workingformat.csv"),
    )
    df["trait"] = np.random.default_rng(1).random(len(df)) < 0.8
    expected = alifestd_mask_leaf_trait_pendant_gain_asexual(
        df, trait_mask=df["trait"].to_numpy()
    )["alifestd_mask_leaf_trait_pendant_gain_asexual"]
    assert expected.any()

    res = alifestd_mask_leaf_trait_pendant_gain_polars(
        pl.from_pandas(df.drop(columns=["ancestor_list"])),
        trait_mask="trait",
    )
    assert (
        res["alifestd_mask_leaf_trait_pendant_gain_polars"].to_list()
        == expected.tolist()
    )


def test_null_warns():
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
            "trait": [None, True, None],
        }
    )
    with pytest.warns(UserWarning):
        res = alifestd_mask_leaf_trait_pendant_gain_polars(
            df_pl, trait_mask="trait"
        )
    assert res["alifestd_mask_leaf_trait_pendant_gain_polars"].to_list() == [
        False,
        True,
        False,
    ]


@pytest.mark.parametrize(
    "df_pl",
    [
        pl.DataFrame({"id": [1, 0], "ancestor_id": [0, 0], "trait": [1, 1]}),
        pl.DataFrame({"id": [0, 2], "ancestor_id": [0, 0], "trait": [1, 1]}),
    ],
)
def test_unsupported(df_pl: pl.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mask_leaf_trait_pendant_gain_polars(df_pl, trait_mask="trait")

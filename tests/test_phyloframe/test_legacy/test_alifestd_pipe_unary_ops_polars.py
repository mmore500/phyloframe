import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import alifestd_pipe_unary_ops_polars

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_pipe_unary_ops_polars_no_ops(apply: typing.Callable):
    phylogeny_df = pl.from_pandas(
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    )
    original_df = phylogeny_df.clone()

    result = alifestd_pipe_unary_ops_polars(apply(phylogeny_df))

    assert (
        result.collect().equals(original_df)
        if hasattr(result, "collect")
        else result.equals(original_df)
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_pipe_unary_ops_polars_single_op(apply: typing.Callable):
    phylogeny_df = pl.from_pandas(
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    )

    result = alifestd_pipe_unary_ops_polars(
        apply(phylogeny_df),
        lambda df: df.with_columns(pl.lit(True).alias("test_col")),
    )

    collected = result.collect() if hasattr(result, "collect") else result
    assert "test_col" in collected.columns
    assert collected["test_col"].all()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_pipe_unary_ops_polars_multiple_ops(apply: typing.Callable):
    phylogeny_df = pl.from_pandas(
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    )

    result = alifestd_pipe_unary_ops_polars(
        apply(phylogeny_df),
        lambda df: df.with_columns(pl.lit(1).alias("col_a")),
        lambda df: df.with_columns((pl.col("col_a") + 1).alias("col_b")),
    )

    collected = result.collect() if hasattr(result, "collect") else result
    assert "col_a" in collected.columns
    assert "col_b" in collected.columns
    assert (collected["col_b"] == 2).all()


def test_alifestd_pipe_unary_ops_polars_order():
    phylogeny_df = pl.from_pandas(
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    )
    log = []

    def op_a(df):
        log.append("a")
        return df

    def op_b(df):
        log.append("b")
        return df

    alifestd_pipe_unary_ops_polars(phylogeny_df, op_a, op_b)

    assert log == ["a", "b"]

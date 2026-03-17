import os

import pandas as pd

from phyloframe.legacy import alifestd_pipe_unary_ops

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def test_alifestd_pipe_unary_ops_no_ops():
    phylogeny_df = pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    original_df = phylogeny_df.copy()

    result = alifestd_pipe_unary_ops(phylogeny_df)

    pd.testing.assert_frame_equal(result, original_df)


def test_alifestd_pipe_unary_ops_single_op():
    phylogeny_df = pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")

    result = alifestd_pipe_unary_ops(
        phylogeny_df,
        lambda df: df.assign(test_col=True),
    )

    assert "test_col" in result.columns
    assert result["test_col"].all()


def test_alifestd_pipe_unary_ops_multiple_ops():
    phylogeny_df = pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")

    result = alifestd_pipe_unary_ops(
        phylogeny_df,
        lambda df: df.assign(col_a=1),
        lambda df: df.assign(col_b=df["col_a"] + 1),
    )

    assert "col_a" in result.columns
    assert "col_b" in result.columns
    assert (result["col_b"] == 2).all()


def test_alifestd_pipe_unary_ops_does_not_mutate():
    phylogeny_df = pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    original_df = phylogeny_df.copy()

    alifestd_pipe_unary_ops(
        phylogeny_df,
        lambda df: df.assign(mutated=True),
    )

    pd.testing.assert_frame_equal(phylogeny_df, original_df)


def test_alifestd_pipe_unary_ops_order():
    phylogeny_df = pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    log = []

    def op_a(df):
        log.append("a")
        return df

    def op_b(df):
        log.append("b")
        return df

    alifestd_pipe_unary_ops(phylogeny_df, op_a, op_b)

    assert log == ["a", "b"]

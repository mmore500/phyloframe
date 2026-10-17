import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_minify_asexual,
    alifestd_minify_polars,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_list_col_polars,
    alifestd_unminify_asexual,
)
from phyloframe.legacy._alifestd_unminify_polars import (
    alifestd_unminify_polars,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
        ),
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny-tworoots.csv"
        ),
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_unminify_polars_matches_pandas(
    phylogeny_df: pd.DataFrame,
    apply: typing.Callable,
):
    """Verify polars result matches pandas result."""
    minified_df = alifestd_minify_asexual(phylogeny_df)
    result_pd = alifestd_unminify_asexual(minified_df)

    df_pl = apply(pl.from_pandas(minified_df))
    result_pl = alifestd_unminify_polars(df_pl).lazy().collect()

    pd.testing.assert_frame_equal(
        result_pl.to_pandas(),
        result_pd.reset_index(drop=True),
    )


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(
                f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
            )
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_unminify_polars_round_trip(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify unminify(minify(x)) relabels x without changing topology."""
    df_pl = pl.from_pandas(phylogeny_df).with_columns(
        original_id=pl.col("id"),
    )
    minified = alifestd_minify_polars(apply(df_pl))
    restored = alifestd_unminify_polars(minified).lazy().collect()

    assert len(restored) == len(df_pl)
    assert restored["id"].to_list() == list(range(len(restored)))
    assert (restored["ancestor_id"] <= restored["id"]).all()

    # ids are contiguous, so index positionally
    original_ids = restored["original_id"]
    assert (
        original_ids.gather(restored["ancestor_id"])
        == df_pl["ancestor_id"].gather(original_ids)
    ).all()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_unminify_polars_simple_tree(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "ancestor_offset": [0, 1, 1, 3, 1],
                "name": ["a", "b", "d", "c", "e"],
            }
        ),
    )

    result = alifestd_unminify_polars(df_pl)

    assert isinstance(result, type(df_pl))
    result = result.lazy().collect()
    assert result.columns == ["id", "ancestor_id", "name"]
    assert result["id"].to_list() == [0, 1, 2, 3, 4]
    assert result["ancestor_id"].to_list() == [0, 0, 1, 0, 3]
    assert result["name"].to_list() == ["a", "b", "d", "c", "e"]

    result = alifestd_try_add_ancestor_list_col_polars(result)
    assert result["ancestor_list"].to_list() == [
        "[none]",
        "[0]",
        "[1]",
        "[0]",
        "[3]",
    ]


@pytest.mark.parametrize("offset_dtype", [pl.Int8, pl.Int64, pl.UInt32])
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_unminify_polars_dtype_default(
    apply: typing.Callable, offset_dtype: pl.DataType
):
    # offsets fit in int8, but ids do not
    df_pl = apply(
        pl.DataFrame(
            {"ancestor_offset": [0] + [1] * 299},
            schema={"ancestor_offset": offset_dtype},
        ),
    )

    result = alifestd_unminify_polars(df_pl).lazy().collect()

    assert result.schema["id"] == pl.Int64
    assert result.schema["ancestor_id"] == pl.Int64
    assert result["id"].to_list() == list(range(300))
    assert result["ancestor_id"].to_list() == [0, *range(299)]


@pytest.mark.parametrize("dtype_id", [pl.Int32, pl.UInt64])
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_unminify_polars_dtype_explicit(
    apply: typing.Callable, dtype_id: pl.DataType
):
    df_pl = apply(
        pl.DataFrame(
            {"ancestor_offset": [0, 1, 1]},
            schema={"ancestor_offset": pl.Int8},
        ),
    )

    result = alifestd_unminify_polars(df_pl, dtype_id=dtype_id)

    result = result.lazy().collect()
    assert result.schema["id"] == dtype_id
    assert result.schema["ancestor_id"] == dtype_id
    assert result["ancestor_id"].to_list() == [0, 0, 1]


@pytest.mark.parametrize("column", ["id", "ancestor_id"])
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_unminify_polars_warns_existing_id_columns(
    apply: typing.Callable, column: str
):
    df_pl = apply(
        pl.DataFrame({"ancestor_offset": [0, 1, 1], column: [9, 9, 9]}),
    )

    with pytest.warns(UserWarning, match=column):
        result = alifestd_unminify_polars(df_pl).lazy().collect()

    assert result.columns == ["id", "ancestor_id"]
    assert result["id"].to_list() == [0, 1, 2]
    assert result["ancestor_id"].to_list() == [0, 0, 1]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_unminify_polars_empty(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {"ancestor_offset": []},
            schema={"ancestor_offset": pl.Int64},
        ),
    )

    result = alifestd_unminify_polars(df_pl).lazy().collect()

    assert result.is_empty()
    assert result.columns == ["id", "ancestor_id"]

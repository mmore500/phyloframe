import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_minify_asexual,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_minify_polars import alifestd_minify_polars

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(
                f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
            )
        ),
        alifestd_to_working_format(
            pd.read_csv(
                f"{assets_path}/example-standard-toy-asexual-phylogeny-tworoots.csv"
            )
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
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
@pytest.mark.parametrize("shrink_dtype", [True, False])
def test_alifestd_minify_polars_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable, shrink_dtype: bool
):
    """Verify polars result matches pandas result."""
    result_pd = alifestd_minify_asexual(
        phylogeny_df, shrink_dtype=shrink_dtype
    )

    df_pl = apply(pl.from_pandas(phylogeny_df))
    result_pl = (
        alifestd_minify_polars(df_pl, shrink_dtype=shrink_dtype)
        .lazy()
        .collect()
    )

    pd.testing.assert_frame_equal(
        result_pl.to_pandas(),
        result_pd.reset_index(drop=True),
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_minify_polars_simple_tree(apply: typing.Callable):
    """Tree structure:
        0 (root)
        +-- 1
        |   +-- 3
        +-- 2
            +-- 4

    Preorder: 0, 1, 3, 2, 4.
    """
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 2],
                "name": ["a", "b", "c", "d", "e"],
            }
        ),
    )

    result = alifestd_minify_polars(df_pl)

    assert isinstance(result, type(df_pl))
    result = result.lazy().collect()
    assert result.columns == ["name", "ancestor_offset"]
    assert result["name"].to_list() == ["a", "b", "d", "c", "e"]
    assert result["ancestor_offset"].to_list() == [0, 1, 1, 3, 1]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_minify_polars_drops_ancestor_list(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_list": ["[none]", "[0]", "[1]"],
            }
        ),
    )

    result = alifestd_minify_polars(df_pl).lazy().collect()

    assert result.columns == ["ancestor_offset"]
    assert result["ancestor_offset"].to_list() == [0, 1, 1]


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64])
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_minify_polars_dtype_no_shrink(
    apply: typing.Callable, dtype: pl.DataType
):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
            },
            schema={"id": dtype, "ancestor_id": dtype},
        ),
    )

    result = alifestd_minify_polars(df_pl, shrink_dtype=False).lazy().collect()

    assert result.schema["ancestor_offset"] == dtype
    assert result["ancestor_offset"].to_list() == [0, 1, 1]


@pytest.mark.parametrize(
    "num_rows, expected_dtype",
    [(1, pl.Int8), (3, pl.Int8), (200, pl.Int16), (40000, pl.Int32)],
)
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_minify_polars_dtype_shrink(
    apply: typing.Callable, num_rows: int, expected_dtype: pl.DataType
):
    # star topology: maximum offset is num_rows - 1
    df_pl = apply(
        pl.DataFrame(
            {
                "id": pl.int_range(num_rows, eager=True),
                "ancestor_id": pl.zeros(num_rows, dtype=pl.Int64, eager=True),
            }
        ),
    )

    result = alifestd_minify_polars(df_pl).lazy().collect()

    assert result.schema["ancestor_offset"] == expected_dtype
    assert result["ancestor_offset"].to_list() == list(range(num_rows))


@pytest.mark.parametrize("dtype", [pl.UInt8, pl.UInt32, pl.UInt64])
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_minify_polars_dtype_shrink_unsigned(
    apply: typing.Callable, dtype: pl.DataType
):
    df_pl = apply(
        pl.DataFrame(
            {"id": [0, 1, 2], "ancestor_id": [0, 0, 1]},
            schema={"id": dtype, "ancestor_id": dtype},
        ),
    )

    result = alifestd_minify_polars(df_pl).lazy().collect()

    assert result.schema["ancestor_offset"] == pl.Int8
    assert result["ancestor_offset"].to_list() == [0, 1, 1]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_minify_polars_warns_existing_ancestor_offset(
    apply: typing.Callable,
):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "ancestor_offset": [9, 9, 9],
            }
        ),
    )

    with pytest.warns(UserWarning, match="ancestor_offset"):
        result = alifestd_minify_polars(df_pl).lazy().collect()

    assert result["ancestor_offset"].to_list() == [0, 1, 1]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_minify_polars_empty(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_minify_polars(df_pl).lazy().collect()

    assert result.is_empty()
    assert result.columns == ["ancestor_offset"]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_minify_polars_non_contiguous_ids(apply: typing.Callable):
    """Verify NotImplementedError for non-contiguous ids."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_minify_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_minify_polars_unsorted(apply: typing.Callable):
    """Verify NotImplementedError for topologically unsorted data."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_minify_polars(df_pl).lazy().collect()

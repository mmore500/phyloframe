import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_mask_persistence_polars import (
    _validate_values,
)

_CRITERIA = ("criterion_cherry", "criterion_trunk", "criterion_tiebreak")


def _make_selected(**kwargs) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ancestor_id": [0, 0, 1],
            "candidate_mask": [False, True, True],
            "extant_mask": [True, False, True],
            "criterion_cherry": [0, 1, 1],
            "criterion_trunk": [0.0, 1.5, 1.5],
            "criterion_tiebreak": [2, 1, 0],
        },
    ).with_columns(**kwargs)


def test_valid():
    _validate_values(_make_selected())
    _validate_values(
        _make_selected(
            criterion_cherry=pl.struct("criterion_cherry", "ancestor_id"),
            criterion_tiebreak=pl.lit(0),
            criterion_trunk=pl.col("criterion_cherry").cast(pl.UInt8),
        ),
    )


@pytest.mark.parametrize("name", ["candidate_mask", "extant_mask"])
def test_null_mask(name: str):
    selected = _make_selected(**{name: pl.Series([True, None, False])})
    with pytest.raises(NotImplementedError, match=f"null {name}"):
        _validate_values(selected)


@pytest.mark.parametrize("name", _CRITERIA)
@pytest.mark.parametrize(
    "values",
    [pl.lit("a"), pl.lit(True), pl.struct(a=pl.lit(0), b=pl.lit("a"))],
)
def test_non_numeric_criterion(values: pl.Expr, name: str):
    selected = _make_selected(**{name: values})
    with pytest.raises(NotImplementedError, match=f"non-numeric {name}"):
        _validate_values(selected)


@pytest.mark.parametrize("name", _CRITERIA)
@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
@pytest.mark.parametrize("as_field", [False, True])
def test_null_or_non_finite_criterion(
    as_field: bool, value: typing.Optional[float], name: str
):
    values = pl.Series([0.0, 1.0, value])
    selected = _make_selected(
        **{name: pl.struct(pl.lit(0), values) if as_field else values},
    )
    with pytest.raises(
        NotImplementedError, match=f"null or non-finite {name}"
    ):
        _validate_values(selected)


@pytest.mark.parametrize("name", ["criterion_cherry", "criterion_trunk"])
@pytest.mark.parametrize("dtype", [pl.Int64, pl.UInt8, pl.Float64])
@pytest.mark.parametrize("as_field", [False, True])
def test_decreasing_criterion(as_field: bool, dtype: type, name: str):
    values = pl.Series([1, 2, 1], dtype=dtype)
    selected = _make_selected(
        **{name: pl.struct(pl.lit(0), values) if as_field else values},
    )
    with pytest.raises(NotImplementedError, match=f"{name} values decreas"):
        _validate_values(selected)


def test_decreasing_tiebreak():
    _validate_values(
        _make_selected(criterion_tiebreak=pl.Series([2, 1, 0])),
    )

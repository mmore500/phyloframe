import typing

import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_assign_contiguous_ids_polars,
    alifestd_calc_quartet_distance_polars,
    alifestd_topological_sort_polars,
)


def _prepare(df: pl.DataFrame) -> pl.DataFrame:
    return alifestd_assign_contiguous_ids_polars(
        alifestd_topological_sort_polars(
            alifestd_assign_contiguous_ids_polars(df),
        ),
    )


@pytest.fixture(
    params=[
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def apply(request: pytest.FixtureRequest) -> typing.Callable:
    return request.param


@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5],
                "taxon_label": [0, 1, 2, 3, 4, 5],
                "ancestor_id": [0, 0, 0, 0, 0, 0],
            },
        ),
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6],
                "taxon_label": ["0", "1", "2", "3", "4", "5", "6"],
                "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            },
        ),
        pl.DataFrame(
            {
                "id": [9, 1, 2, 3, 4, 5, 6],
                "taxon_label": ["9", "1", "2", "3", "4", "5", "6"],
                "ancestor_id": [9, 9, 9, 1, 1, 2, 2],
            },
        ),
    ],
)
def test_polytomy_identical(df: pl.DataFrame, apply: typing.Callable):
    df = _prepare(df)
    assert alifestd_calc_quartet_distance_polars(apply(df), apply(df)) == 0


def test_differing_wrong1(apply: typing.Callable):
    adf = _prepare(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7],
                "taxon_label": ["0", "1", "2", "3", "4", "5", "6", "7"],
                "ancestor_id": [0, 0, 1, 2, 2, 1, 5, 5],
            },
        ),
    )
    bdf = _prepare(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7],
                "taxon_label": ["0", "1", "2", "3", "4", "5", "6", "7"],
                "ancestor_id": [0, 0, 1, 2, 2, 1, 2, 5],
            },
        ),
    )
    assert (
        0 < alifestd_calc_quartet_distance_polars(apply(adf), apply(bdf)) < 1
    )


def test_differing_wrong_big(apply: typing.Callable):
    adf = _prepare(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "taxon_label": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "ancestor_id": [0, 0, 1, 2, 2, 1, 5, 5, 4, 4],
            },
        ),
    )
    bdf = _prepare(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "taxon_label": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "ancestor_id": [0, 0, 1, 5, 2, 1, 2, 5, 4, 4],
            },
        ),
    )
    est = alifestd_calc_quartet_distance_polars(
        apply(adf),
        apply(bdf),
        "taxon_label",
    )
    assert 0 < est < 1


def test_identical_polytomy(apply: typing.Callable):
    adf = _prepare(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "taxon_label": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
                "ancestor_id": [0, 0, 1, 2, 2, 1, 3, 3, 2],
            },
        ),
    )
    bdf = _prepare(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "taxon_label": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
                "ancestor_id": [0, 0, 1, 2, 2, 1, 3, 3, 2],
            },
        ),
    )
    assert (
        alifestd_calc_quartet_distance_polars(
            apply(adf),
            apply(bdf),
            "taxon_label",
        )
        == 0
    )


def test_differing_polytomy_asymmetrical(apply: typing.Callable):
    adf = _prepare(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "taxon_label": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
                "ancestor_id": [0, 0, 1, 2, 2, 1, 5, 5, 1],
            },
        ),
    )
    bdf = _prepare(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "taxon_label": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
                "ancestor_id": [0, 0, 1, 2, 2, 2, 5, 5, 1],
            },
        ),
    )
    assert (
        0
        < alifestd_calc_quartet_distance_polars(
            apply(adf),
            apply(bdf),
            "taxon_label",
        )
        < 1
    )


def test_id_as_taxon_label(apply: typing.Callable):
    adf = _prepare(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "ancestor_id": [0, 0, 1, 2, 2, 1, 5, 5, 4, 4],
            },
        ),
    )
    bdf = _prepare(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "ancestor_id": [0, 0, 1, 5, 2, 1, 2, 5, 4, 4],
            },
        ),
    )
    assert (
        alifestd_calc_quartet_distance_polars(apply(adf), apply(adf), "id")
        == 0
    )
    assert (
        0
        < alifestd_calc_quartet_distance_polars(apply(adf), apply(bdf), "id")
        < 1
    )

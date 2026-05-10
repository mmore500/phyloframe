import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_calc_triplet_distance_polars,
)


@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "taxon_label": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 0, 0],
            },
        ),
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5],
                "taxon_label": ["0", "1", "2", "3", "4", "5"],
                "ancestor_id": [0, 0, 0, 0, 1, 2],
            },
        ),
        pl.DataFrame(
            {
                "id": [9, 1, 2, 3, 4, 5],
                "taxon_label": ["9", "1", "2", "3", "4", "5"],
                "ancestor_id": [9, 9, 9, 9, 1, 2],
            },
        ),
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6],
                "taxon_label": ["0", "1", "2", "3", "4", "5", "6"],
                "ancestor_id": [0, 0, 0, 0, 1, 2, 3],
            },
        ),
    ],
)
def test_polytomy_identical(df: pl.DataFrame):
    for adf, bdf in (df, df), (df, df.sample(fraction=1.0, shuffle=True)):
        assert alifestd_calc_triplet_distance_polars(adf, bdf) == 0


def test_differing_wrong1():
    adf = pl.DataFrame(
        {
            "id": list(reversed([9, 1, 2, 3, 4, 5])),
            "taxon_label": list(reversed(["0", "1", "2", "3", "4", "5"])),
            "ancestor_id": list(reversed([9, 9, 1, 2, 2, 1])),
        },
    )
    bdf = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 5, 4],
            "taxon_label": ["0", "1", "2", "3", "5", "4"],
            "ancestor_id": [0, 0, 1, 2, 2, 1],
        },
    )
    assert 0 < alifestd_calc_triplet_distance_polars(adf, bdf) < 1
    assert (
        0
        < alifestd_calc_triplet_distance_polars(
            adf,
            bdf.sample(fraction=1.0, shuffle=True),
        )
        < 1
    )


def test_differing_wrong2():
    adf = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "taxon_label": ["0", "1", "2", "3", "4", "5"],
            "ancestor_id": [0, 0, 1, 2, 2, 1],
        },
    )
    bdf = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "taxon_label": ["0", "1", "2", "3", "4", "5"],
            "ancestor_id": [0, 0, 1, 2, 1, 2],
        },
    )
    assert 0 < alifestd_calc_triplet_distance_polars(adf, bdf) < 1
    assert (
        0
        < alifestd_calc_triplet_distance_polars(
            adf,
            bdf.sample(fraction=1.0, shuffle=True),
        )
        < 1
    )
    assert (
        0
        < alifestd_calc_triplet_distance_polars(
            adf.sample(fraction=1.0, shuffle=True),
            bdf.sample(fraction=1.0, shuffle=True),
        )
        < 1
    )


def test_identical_polytomy1():
    adf = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "taxon_label": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
            "ancestor_id": [0, 0, 1, 2, 2, 1, 3, 3, 2],
        },
    )
    bdf = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "taxon_label": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
            "ancestor_id": [0, 0, 1, 2, 2, 1, 3, 3, 2],
        },
    )
    assert alifestd_calc_triplet_distance_polars(adf, bdf, "taxon_label") == 0
    assert (
        alifestd_calc_triplet_distance_polars(
            adf,
            bdf.sample(fraction=1.0, shuffle=True),
            "taxon_label",
        )
        == 0
    )


def test_differing_wrong_big():
    adf = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "taxon_label": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "ancestor_id": [0, 0, 1, 2, 2, 1, 5, 5, 4, 4],
        },
    )
    bdf = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "taxon_label": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "ancestor_id": [0, 0, 1, 5, 2, 1, 2, 5, 4, 4],
        },
    )
    est = alifestd_calc_triplet_distance_polars(adf, bdf, "taxon_label")
    assert 0 < est < 1

    est = alifestd_calc_triplet_distance_polars(
        adf,
        bdf.sample(fraction=1.0, shuffle=True),
        "taxon_label",
    )
    assert 0 < est < 1


def test_differing_polytomy_asymmetrical_strict():
    adf = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "taxon_label": ["0", "1", "2", "3", "4", "5"],
            "ancestor_id": [0, 0, 1, 2, 2, 1],
        },
    )
    bdf = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "taxon_label": ["0", "1", "2", "3", "4", "5"],
            "ancestor_id": [0, 0, 1, 2, 2, 2],
        },
    )
    assert (
        0 < alifestd_calc_triplet_distance_polars(adf, bdf, "taxon_label") < 1
    )
    assert (
        0
        < alifestd_calc_triplet_distance_polars(
            adf,
            bdf.sample(fraction=1.0, shuffle=True),
            "taxon_label",
        )
        < 1
    )

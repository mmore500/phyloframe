import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_is_ultrametric,
    alifestd_make_empty,
    alifestd_ultrametricize,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def test_missing_origin_time():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
        }
    )
    with pytest.raises(ValueError):
        alifestd_is_ultrametric(phylogeny_df)


def test_empty():
    df = alifestd_make_empty()
    df["origin_time"] = pd.Series(dtype=float)
    assert alifestd_is_ultrametric(df) is True


def test_single_node():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0],
            "ancestor_list": ["[None]"],
            "origin_time": [3.0],
        }
    )
    assert alifestd_is_ultrametric(phylogeny_df)


def test_simple_chain_is_ultrametric():
    # Single tip -> trivially ultrametric
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
            "origin_time": [0.0, 1.0, 2.0],
        }
    )
    assert alifestd_is_ultrametric(phylogeny_df)


def test_non_ultrametric():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]"],
            "origin_time": [0.0, 1.0, 2.0, 5.0],
        }
    )
    assert not alifestd_is_ultrametric(phylogeny_df)


def test_ultrametric_after_extend():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]"],
            "origin_time": [0.0, 1.0, 2.0, 5.0],
        }
    )
    res = alifestd_ultrametricize(phylogeny_df)
    assert alifestd_is_ultrametric(res)


def test_atol():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[0]"],
            "origin_time": [0.0, 4.9, 5.0],
        }
    )
    assert not alifestd_is_ultrametric(phylogeny_df)
    assert not alifestd_is_ultrametric(phylogeny_df, atol=0.05)
    assert alifestd_is_ultrametric(phylogeny_df, atol=0.1)
    assert alifestd_is_ultrametric(phylogeny_df, atol=1.0)


def test_inputs_not_mutated():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]"],
            "origin_time": [0.0, 1.0, 2.0, 5.0],
        }
    )
    original = phylogeny_df.copy()
    alifestd_is_ultrametric(phylogeny_df)
    assert original.equals(phylogeny_df)


def test_forest():
    # Non-ultrametric forest: leaves at different times
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_list": [
                "[None]",
                "[0]",
                "[0]",
                "[None]",
                "[3]",
                "[3]",
            ],
            "origin_time": [0.0, 2.0, 3.0, 0.0, 4.0, 7.0],
        }
    )
    assert not alifestd_is_ultrametric(phylogeny_df)

    # Ultrametric forest: leaves at the same time
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_list": [
                "[None]",
                "[0]",
                "[0]",
                "[None]",
                "[3]",
                "[3]",
            ],
            "origin_time": [0.0, 5.0, 5.0, 0.0, 5.0, 5.0],
        }
    )
    assert alifestd_is_ultrametric(phylogeny_df)


def test_sexual_phylogeny():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": [
                "[None]",
                "[None]",
                "[0,1]",
                "[2]",
                "[2]",
            ],
            "origin_time": [0.0, 0.0, 1.0, 5.0, 3.0],
        }
    )
    assert not alifestd_is_ultrametric(phylogeny_df)

    res = alifestd_ultrametricize(phylogeny_df)
    assert alifestd_is_ultrametric(res)


def test_preexisting_is_leaf_column():
    # if is_leaf is overridden, it should be respected
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
            "origin_time": [0.0, 1.0, 2.0],
            "is_leaf": [True, False, False],
        }
    )
    # the only "leaf" is id 0; trivially ultrametric
    assert alifestd_is_ultrametric(phylogeny_df)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
def test_fuzz_after_extend_is_ultrametric(phylogeny_df: pd.DataFrame):
    res = alifestd_ultrametricize(phylogeny_df)
    assert alifestd_is_ultrametric(res)

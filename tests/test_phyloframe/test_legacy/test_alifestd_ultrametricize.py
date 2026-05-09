import os

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
)
from phyloframe.legacy import (
    alifestd_ultrametricize as alifestd_ultrametricize_,
)
from phyloframe.legacy import (
    alifestd_validate,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_ultrametricize = enforce_dtype_stability_pandas(
    alifestd_ultrametricize_,
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
        alifestd_ultrametricize(phylogeny_df)


def test_unknown_method():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
            "origin_time": [0.0, 1.0, 2.0],
        }
    )
    with pytest.raises(ValueError):
        alifestd_ultrametricize(phylogeny_df, method="bogus")


def test_empty():
    df = alifestd_make_empty()
    df["origin_time"] = pd.Series(dtype=float)
    res = alifestd_ultrametricize(df)
    assert "origin_time" in res
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_chain(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
            "origin_time": [0.0, 1.0, 2.0],
        }
    )
    original = phylogeny_df.copy()
    result = alifestd_ultrametricize(phylogeny_df, mutate=mutate)
    result.index = result["id"]

    # only leaf is id 2; origin_time stays at max (2.0)
    assert result.loc[2, "origin_time"] == 2.0
    # internal nodes are unchanged
    assert result.loc[0, "origin_time"] == 0.0
    assert result.loc[1, "origin_time"] == 1.0

    if not mutate:
        assert original.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree_extends(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]"],
            "origin_time": [0.0, 1.0, 2.0, 5.0],
        }
    )
    original = phylogeny_df.copy()
    result = alifestd_ultrametricize(phylogeny_df, mutate=mutate)
    result.index = result["id"]

    # leaves are id 2 and 3; both should have origin_time=5.0
    assert result.loc[2, "origin_time"] == 5.0
    assert result.loc[3, "origin_time"] == 5.0
    # inner nodes unchanged
    assert result.loc[0, "origin_time"] == 0.0
    assert result.loc[1, "origin_time"] == 1.0

    if not mutate:
        assert original.equals(phylogeny_df)


def test_forest_multiple_roots():
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
    original = phylogeny_df.copy()
    result = alifestd_ultrametricize(phylogeny_df)
    result.index = result["id"]

    # leaves: 1, 2, 4, 5 -> all should equal max (7.0)
    for leaf_id in [1, 2, 4, 5]:
        assert result.loc[leaf_id, "origin_time"] == 7.0
    # roots unchanged
    assert result.loc[0, "origin_time"] == 0.0
    assert result.loc[3, "origin_time"] == 0.0

    assert original.equals(phylogeny_df)


def test_sexual_phylogeny():
    # 0 and 1 are root parents; 2 has ancestors [0,1]; 3 has ancestors [2]
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
            "origin_time": [0.0, 0.0, 1.0, 3.0, 5.0],
        }
    )
    original = phylogeny_df.copy()
    result = alifestd_ultrametricize(phylogeny_df)
    result.index = result["id"]

    # leaves: 3, 4 -> both should equal max (5.0)
    assert result.loc[3, "origin_time"] == 5.0
    assert result.loc[4, "origin_time"] == 5.0
    # non-leaves unchanged
    assert result.loc[0, "origin_time"] == 0.0
    assert result.loc[1, "origin_time"] == 0.0
    assert result.loc[2, "origin_time"] == 1.0

    assert original.equals(phylogeny_df)


def test_preexisting_is_leaf_column():
    # is_leaf is misleadingly set; alifestd_ultrametricize should respect
    # it for which rows to touch, while pulling the target time from the
    # max across all rows.
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
            "origin_time": [0.0, 1.0, 2.0],
            "is_leaf": [False, True, False],
        }
    )
    result = alifestd_ultrametricize(phylogeny_df)
    result.index = result["id"]
    # only is_leaf row is touched, and is set to max-over-all = 2.0
    assert result.loc[0, "origin_time"] == 0.0
    assert result.loc[1, "origin_time"] == 2.0
    assert result.loc[2, "origin_time"] == 2.0


def test_noncontiguous_ids():
    phylogeny_df = pd.DataFrame(
        {
            "id": [10, 20, 30, 40],
            "ancestor_list": ["[None]", "[10]", "[10]", "[20]"],
            "origin_time": [0.0, 1.0, 2.0, 5.0],
        }
    )
    original = phylogeny_df.copy()
    result = alifestd_ultrametricize(phylogeny_df)
    result.index = result["id"]

    # leaves are 30 and 40; both -> 5.0
    assert result.loc[30, "origin_time"] == 5.0
    assert result.loc[40, "origin_time"] == 5.0
    # inner unchanged
    assert result.loc[10, "origin_time"] == 0.0
    assert result.loc[20, "origin_time"] == 1.0

    assert original.equals(phylogeny_df)


def test_ancestor_id_only():
    # ancestor_id is provided in lieu of ancestor_list
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 0, 1],
            "origin_time": [0.0, 1.0, 2.0, 5.0],
        }
    )
    original = phylogeny_df.copy()
    result = alifestd_ultrametricize(phylogeny_df)
    result.index = result["id"]
    # leaves: 2, 3 -> 5.0
    assert result.loc[2, "origin_time"] == 5.0
    assert result.loc[3, "origin_time"] == 5.0

    assert original.equals(phylogeny_df)


def test_ancestor_id_and_ancestor_list():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 0, 1],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]"],
            "origin_time": [0.0, 1.0, 2.0, 5.0],
        }
    )
    original = phylogeny_df.copy()
    result = alifestd_ultrametricize(phylogeny_df)
    result.index = result["id"]
    assert result.loc[2, "origin_time"] == 5.0
    assert result.loc[3, "origin_time"] == 5.0

    assert original.equals(phylogeny_df)


def test_noncontiguous_ids_ancestor_id_only():
    phylogeny_df = pd.DataFrame(
        {
            "id": [10, 20, 30, 40],
            "ancestor_id": [10, 10, 10, 20],
            "origin_time": [0.0, 1.0, 2.0, 5.0],
        }
    )
    original = phylogeny_df.copy()
    result = alifestd_ultrametricize(phylogeny_df)
    result.index = result["id"]
    assert result.loc[30, "origin_time"] == 5.0
    assert result.loc[40, "origin_time"] == 5.0

    assert original.equals(phylogeny_df)


def test_already_ultrametric():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[0]"],
            "origin_time": [0.0, 5.0, 5.0],
        }
    )
    result = alifestd_ultrametricize(phylogeny_df)
    np.testing.assert_array_equal(
        result["origin_time"].to_numpy(),
        phylogeny_df["origin_time"].to_numpy(),
    )


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
def test_fuzz(phylogeny_df: pd.DataFrame):
    original = phylogeny_df.copy()
    result = alifestd_ultrametricize(phylogeny_df)

    assert alifestd_validate(result)
    assert original.equals(phylogeny_df)

    from phyloframe.legacy import alifestd_find_leaf_ids

    leaf_ids = list(alifestd_find_leaf_ids(phylogeny_df))
    target = phylogeny_df["origin_time"].max()
    leaf_ots = result.set_index("id").loc[leaf_ids, "origin_time"]
    np.testing.assert_array_equal(leaf_ots.to_numpy(), target)

import os

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_is_working_format_asexual,
    alifestd_make_empty,
    alifestd_parse_ancestor_ids,
)
from phyloframe.legacy import (
    alifestd_shuffle_working_format as alifestd_shuffle_working_format_,
)
from phyloframe.legacy import (
    alifestd_test_leaves_isomorphic_asexual,
    alifestd_to_working_format,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_shuffle_working_format = enforce_dtype_stability_pandas(
    alifestd_shuffle_working_format_
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _label_parents(phylogeny_df: pd.DataFrame) -> dict:
    """Map each taxon's label to its parents' labels."""
    label_lookup = dict(zip(phylogeny_df["id"], phylogeny_df["label"]))
    return {
        label: sorted(
            label_lookup[ancestor_id]
            for ancestor_id in alifestd_parse_ancestor_ids(ancestor_list)
        )
        for label, ancestor_list in zip(
            phylogeny_df["label"], phylogeny_df["ancestor_list"]
        )
    }


_assets = pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv"),
        pd.read_csv(f"{assets}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets}/nk_tournamentselection.csv"),
        pd.read_csv(f"{assets}/example-standard-toy-asexual-phylogeny.csv"),
        pd.read_csv(
            f"{assets}/example-standard-toy-asexual-phylogeny-tworoots.csv",
        ),
    ],
)


def test_shuffle_working_format_empty():
    df = alifestd_make_empty(ancestor_id=True)
    result = alifestd_shuffle_working_format(df)
    assert result.empty
    assert alifestd_is_working_format_asexual(result)


@_assets
@pytest.mark.parametrize("mutate", [True, False])
@pytest.mark.parametrize("seed", range(3))
def test_shuffle_working_format_asexual(
    phylogeny_df: pd.DataFrame, mutate: bool, seed: int
):
    phylogeny_df = phylogeny_df.copy()
    phylogeny_df["label"] = phylogeny_df["id"]
    original_df = phylogeny_df.copy()

    result = alifestd_shuffle_working_format(
        phylogeny_df, mutate=mutate, seed=seed
    )
    if not mutate:
        assert phylogeny_df.equals(original_df)

    assert alifestd_is_working_format_asexual(result)
    assert len(result) == len(original_df)
    assert set(result.columns) >= set(original_df.columns)
    assert _label_parents(result) == _label_parents(original_df)
    assert alifestd_test_leaves_isomorphic_asexual(
        alifestd_to_working_format(original_df),
        result,
        taxon_label="label",
    )
    # non-topological columns travel with their taxa
    merged = original_df.merge(result, on="label", suffixes=("", "_"))
    for col in original_df.columns:
        if col not in ("id", "ancestor_id", "ancestor_list", "label"):
            assert merged[col].equals(merged[f"{col}_"])


@pytest.mark.parametrize("seed", range(3))
def test_shuffle_working_format_sexual(seed: int):
    df = pd.read_csv(f"{assets}/example-standard-toy-sexual-phylogeny.csv")
    df["label"] = df["id"]
    result = alifestd_shuffle_working_format(df, seed=seed)
    assert "ancestor_id" not in result
    assert result["id"].tolist() == list(range(len(df)))
    assert _label_parents(result) == _label_parents(df)


def test_shuffle_working_format_seed():
    df = pd.read_csv(f"{assets}/nk_lexicaseselection.csv")
    df["label"] = df["id"]
    result1 = alifestd_shuffle_working_format(df, seed=1)
    result2 = alifestd_shuffle_working_format(df, seed=1)
    result3 = alifestd_shuffle_working_format(df, seed=2)
    assert result1.equals(result2)
    assert not result1["label"].equals(result3["label"])
    working_df = alifestd_to_working_format(df)
    assert not result1["label"].equals(working_df["label"])


def test_shuffle_working_format_ids_dtype():
    df = pd.DataFrame(
        {
            "id": np.array([0, 1, 2, 3], dtype=np.uint32),
            "ancestor_id": np.array([0, 0, 1, 1], dtype=np.uint32),
        }
    )
    result = alifestd_shuffle_working_format(df, seed=1)
    assert result["id"].dtype == np.uint32
    assert result["ancestor_id"].dtype == np.uint32

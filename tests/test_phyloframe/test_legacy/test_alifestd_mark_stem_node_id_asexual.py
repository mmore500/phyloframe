import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
)
from phyloframe.legacy import (
    alifestd_mark_stem_node_id_asexual as alifestd_mark_stem_node_id_asexual_,
)
from phyloframe.legacy._alifestd_mark_stem_node_id_asexual import (
    _alifestd_mark_stem_node_id_asexual_fast_path,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_mark_stem_node_id_asexual = enforce_dtype_stability_pandas(
    alifestd_mark_stem_node_id_asexual_
)


def _make_forest() -> pd.DataFrame:
    #   0       5
    #   1      6 7
    #  2 3
    #  4
    # root 0 is unifurcating, root 5 is branching, and 2 is unifurcating
    return pd.DataFrame(
        {"id": range(8), "ancestor_id": [0, 0, 1, 1, 2, 5, 5, 5]},
    )


def test_empty():
    res = alifestd_mark_stem_node_id_asexual(alifestd_make_empty())
    assert "stem_node_id" in res.columns
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [False, True])
def test_forest(mutate: bool):
    phylogeny_df = _make_forest()
    original_df = phylogeny_df.copy()
    res = alifestd_mark_stem_node_id_asexual(phylogeny_df, mutate=mutate)
    # 0, 1, and 5 have no branching node above them, so are their own; 4
    # bypasses unifurcation 2
    assert res["stem_node_id"].tolist() == [0, 1, 1, 1, 1, 5, 5, 5]
    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_mark_as():
    res = alifestd_mark_stem_node_id_asexual(_make_forest(), mark_as="s")
    assert "s" in res.columns
    assert "stem_node_id" not in res.columns


def test_bifurcating():
    #        0
    #     1     2
    #    3 4   5 6
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
    res = _alifestd_mark_stem_node_id_asexual_fast_path(ancestor_ids)
    # without unifurcations, each non-root node's stem node is its parent
    assert res.tolist() == [0, 0, 0, 1, 1, 2, 2]


def test_ancestor_list():
    phylogeny_df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_list": ["[none]", "[0]", "[0]"]},
    )
    res = alifestd_mark_stem_node_id_asexual(phylogeny_df)
    assert res["stem_node_id"].tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pytest.param(
            pd.DataFrame({"id": [3, 13, 23], "ancestor_id": [3, 3, 3]}),
            id="noncontiguous",
        ),
        pytest.param(
            pd.DataFrame({"id": [0, 1, 2], "ancestor_id": [1, 1, 1]}),
            id="unsorted",
        ),
    ],
)
def test_not_working_format_raises(phylogeny_df: pd.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mark_stem_node_id_asexual(phylogeny_df)

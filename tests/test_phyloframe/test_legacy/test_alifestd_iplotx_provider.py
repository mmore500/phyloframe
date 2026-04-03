import numpy as np
import pandas as pd
import polars as pl
import pytest

iplotx = pytest.importorskip("iplotx")

from phyloframe.legacy import (  # noqa: E402
    AlifestdIplotxShimNumpy,
    AlifestdIplotxShimPandas,
    AlifestdIplotxShimPolars,
)


# ---- helpers --------------------------------------------------------
def _make_chain_pandas(n: int = 4) -> pd.DataFrame:
    """Chain phylogeny: 0 -> 1 -> 2 -> ... -> n-1."""
    ids = list(range(n))
    ancestor_ids = [0] + list(range(n - 1))
    return pd.DataFrame({"id": ids, "ancestor_id": ancestor_ids})


def _make_balanced_pandas() -> pd.DataFrame:
    """Balanced bifurcating tree of depth 3 (7 nodes).

    ::

            0
           / \\
          1   2
         / \\ / \\
        3  4 5  6
    """
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
        }
    )


def _make_balanced_with_times_pandas() -> pd.DataFrame:
    """Balanced tree with origin_time for branch length testing."""
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            "origin_time": [0.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
        }
    )


def _make_balanced_with_deltas_pandas() -> pd.DataFrame:
    """Balanced tree with origin_time_delta for branch length testing."""
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            "origin_time_delta": [
                float("nan"),
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ],
        }
    )


def _make_single_root_pandas() -> pd.DataFrame:
    """Single root node."""
    return pd.DataFrame({"id": [0], "ancestor_id": [0]})


def _make_with_taxon_labels_pandas() -> pd.DataFrame:
    """Simple tree with taxon labels."""
    return pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
            "taxon_label": ["root", "left", "right"],
        }
    )


# ---- AlifestdIplotxShimNumpy tests -----------------------------------
def test_numpy_check_dependencies():
    assert AlifestdIplotxShimNumpy.check_dependencies() is True


def test_numpy_tree_type():
    assert AlifestdIplotxShimNumpy.tree_type() is AlifestdIplotxShimNumpy


def test_numpy_is_rooted():
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
    shim = AlifestdIplotxShimNumpy(ancestor_ids)
    assert shim.is_rooted() is True


def test_numpy_get_root():
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
    shim = AlifestdIplotxShimNumpy(ancestor_ids)
    root = shim.get_root()
    assert root._id == 0


def test_numpy_preorder():
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
    shim = AlifestdIplotxShimNumpy(ancestor_ids)
    order = [n._id for n in shim.preorder()]
    assert order == [0, 1, 3, 4, 2, 5, 6]


def test_numpy_postorder():
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
    shim = AlifestdIplotxShimNumpy(ancestor_ids)
    order = [n._id for n in shim.postorder()]
    # Verify postorder property: all children appear before parent
    seen = set()
    for nid in order:
        parent = int(ancestor_ids[nid])
        if parent != nid:
            assert parent not in seen
        seen.add(nid)
    assert set(order) == set(range(7))


def test_numpy_levelorder():
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
    shim = AlifestdIplotxShimNumpy(ancestor_ids)
    order = [n._id for n in shim.levelorder()]
    assert order == [0, 1, 2, 3, 4, 5, 6]


def test_numpy_get_leaves():
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
    shim = AlifestdIplotxShimNumpy(ancestor_ids)
    leaf_ids = sorted(n._id for n in shim._get_leaves())
    assert leaf_ids == [3, 4, 5, 6]


def test_numpy_get_children():
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
    shim = AlifestdIplotxShimNumpy(ancestor_ids)
    root = shim.get_root()
    child_ids = [c._id for c in shim.get_children(root)]
    assert child_ids == [1, 2]


def test_numpy_get_branch_length_none():
    ancestor_ids = np.array([0, 0, 0])
    shim = AlifestdIplotxShimNumpy(ancestor_ids)
    root = shim.get_root()
    assert shim.get_branch_length(root) is None


def test_numpy_get_branch_length_with_values():
    ancestor_ids = np.array([0, 0, 0])
    branch_lengths = np.array([np.nan, 1.5, 2.5])
    shim = AlifestdIplotxShimNumpy(ancestor_ids, branch_lengths=branch_lengths)
    children = shim.get_children(shim.get_root())
    assert shim.get_branch_length(children[0]) == 1.5
    assert shim.get_branch_length(children[1]) == 2.5


def test_numpy_names_default():
    ancestor_ids = np.array([0, 0])
    shim = AlifestdIplotxShimNumpy(ancestor_ids)
    assert shim.get_root().name == "0"


def test_numpy_names_custom():
    ancestor_ids = np.array([0, 0])
    names = np.array(["root", "child"])
    shim = AlifestdIplotxShimNumpy(ancestor_ids, names=names)
    assert shim.get_root().name == "root"


def test_numpy_single_node():
    ancestor_ids = np.array([0])
    shim = AlifestdIplotxShimNumpy(ancestor_ids)
    assert shim.get_root()._id == 0
    assert [n._id for n in shim.preorder()] == [0]
    assert [n._id for n in shim.postorder()] == [0]
    assert [n._id for n in shim.levelorder()] == [0]
    assert [n._id for n in shim._get_leaves()] == [0]
    assert shim.get_children(shim.get_root()) == []


def test_numpy_chain():
    ancestor_ids = np.array([0, 0, 1, 2])
    shim = AlifestdIplotxShimNumpy(ancestor_ids)
    assert [n._id for n in shim.preorder()] == [0, 1, 2, 3]
    assert [n._id for n in shim.postorder()] == [3, 2, 1, 0]
    assert [n._id for n in shim._get_leaves()] == [3]


def test_numpy_node_hashable():
    ancestor_ids = np.array([0, 0, 0])
    shim = AlifestdIplotxShimNumpy(ancestor_ids)
    nodes = list(shim.preorder())
    d = {node: i for i, node in enumerate(nodes)}
    assert len(d) == 3


def test_numpy_node_equality():
    ancestor_ids = np.array([0, 0])
    shim = AlifestdIplotxShimNumpy(ancestor_ids)
    nodes1 = list(shim.preorder())
    nodes2 = list(shim.preorder())
    assert nodes1[0] == nodes2[0]
    assert nodes1[0] != nodes1[1]


# ---- AlifestdIplotxShimPandas tests ----------------------------------
def test_pandas_check_dependencies():
    assert AlifestdIplotxShimPandas.check_dependencies() is True


def test_pandas_tree_type():
    assert AlifestdIplotxShimPandas.tree_type() is pd.DataFrame


def test_pandas_balanced_tree():
    df = _make_balanced_pandas()
    shim = AlifestdIplotxShimPandas(df)
    assert shim.is_rooted() is True
    assert shim.get_root()._id == 0
    assert [n._id for n in shim.preorder()] == [0, 1, 3, 4, 2, 5, 6]
    assert sorted(n._id for n in shim._get_leaves()) == [3, 4, 5, 6]


def test_pandas_chain():
    df = _make_chain_pandas(4)
    shim = AlifestdIplotxShimPandas(df)
    assert [n._id for n in shim.preorder()] == [0, 1, 2, 3]
    assert [n._id for n in shim._get_leaves()] == [3]


def test_pandas_single_root():
    df = _make_single_root_pandas()
    shim = AlifestdIplotxShimPandas(df)
    assert shim.get_root()._id == 0
    assert [n._id for n in shim._get_leaves()] == [0]


def test_pandas_branch_lengths_from_origin_time():
    df = _make_balanced_with_times_pandas()
    shim = AlifestdIplotxShimPandas(df)
    root = shim.get_root()
    assert shim.get_branch_length(root) is None
    children = shim.get_children(root)
    for child in children:
        assert shim.get_branch_length(child) == pytest.approx(1.0)
    for child in children:
        for grandchild in shim.get_children(child):
            assert shim.get_branch_length(grandchild) == pytest.approx(2.0)


def test_pandas_branch_lengths_from_delta():
    df = _make_balanced_with_deltas_pandas()
    shim = AlifestdIplotxShimPandas(df)
    root = shim.get_root()
    assert shim.get_branch_length(root) is None
    children = shim.get_children(root)
    for child in children:
        assert shim.get_branch_length(child) == pytest.approx(1.0)


def test_pandas_taxon_labels():
    df = _make_with_taxon_labels_pandas()
    shim = AlifestdIplotxShimPandas(df)
    root = shim.get_root()
    assert root.name == "root"
    children = shim.get_children(root)
    child_names = sorted(c.name for c in children)
    assert child_names == ["left", "right"]


def test_pandas_no_branch_lengths():
    df = _make_balanced_pandas()
    shim = AlifestdIplotxShimPandas(df)
    for node in shim.preorder():
        assert shim.get_branch_length(node) is None


def test_pandas_from_ancestor_list():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[0]"],
        }
    )
    shim = AlifestdIplotxShimPandas(df)
    assert shim.get_root()._id == 0
    assert sorted(n._id for n in shim._get_leaves()) == [1, 2]


def test_pandas_non_contiguous_ids_raises():
    df = pd.DataFrame(
        {
            "id": [0, 5, 10],
            "ancestor_id": [0, 0, 5],
        }
    )
    with pytest.raises(NotImplementedError, match="contiguous"):
        AlifestdIplotxShimPandas(df)


def test_pandas_not_topologically_sorted_raises():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 2, 0],
        }
    )
    with pytest.raises(NotImplementedError, match="topologically"):
        AlifestdIplotxShimPandas(df)


def test_pandas_does_not_mutate_input():
    df = _make_balanced_pandas()
    original = df.copy()
    AlifestdIplotxShimPandas(df)
    assert df.equals(original)


# ---- AlifestdIplotxShimPolars tests ----------------------------------
def test_polars_check_dependencies():
    assert AlifestdIplotxShimPolars.check_dependencies() is True


def test_polars_tree_type():
    assert AlifestdIplotxShimPolars.tree_type() is pl.DataFrame


def test_polars_balanced_tree():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
        }
    )
    shim = AlifestdIplotxShimPolars(df)
    assert shim.is_rooted() is True
    assert shim.get_root()._id == 0
    assert [n._id for n in shim.preorder()] == [0, 1, 3, 4, 2, 5, 6]
    assert sorted(n._id for n in shim._get_leaves()) == [3, 4, 5, 6]


def test_polars_chain():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 2],
        }
    )
    shim = AlifestdIplotxShimPolars(df)
    assert [n._id for n in shim.preorder()] == [0, 1, 2, 3]
    assert [n._id for n in shim._get_leaves()] == [3]


def test_polars_branch_lengths_from_origin_time():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            "origin_time": [0.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
        }
    )
    shim = AlifestdIplotxShimPolars(df)
    root = shim.get_root()
    assert shim.get_branch_length(root) is None
    for child in shim.get_children(root):
        assert shim.get_branch_length(child) == pytest.approx(1.0)


def test_polars_branch_lengths_from_delta():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            "origin_time_delta": [
                float("nan"),
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ],
        }
    )
    shim = AlifestdIplotxShimPolars(df)
    root = shim.get_root()
    assert shim.get_branch_length(root) is None
    for child in shim.get_children(root):
        assert shim.get_branch_length(child) == pytest.approx(1.0)


def test_polars_taxon_labels():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
            "taxon_label": ["root", "left", "right"],
        }
    )
    shim = AlifestdIplotxShimPolars(df)
    root = shim.get_root()
    assert root.name == "root"


def test_polars_non_contiguous_ids_raises():
    df = pl.DataFrame(
        {
            "id": [0, 5, 10],
            "ancestor_id": [0, 0, 5],
        }
    )
    with pytest.raises(NotImplementedError, match="contiguous"):
        AlifestdIplotxShimPolars(df)


def test_polars_not_topologically_sorted_raises():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 2, 0],
        }
    )
    with pytest.raises(NotImplementedError, match="topologically"):
        AlifestdIplotxShimPolars(df)


# ---- iplotx integration smoke tests --------------------------------
def test_iplotx_call_pandas():
    df = _make_balanced_with_times_pandas()
    shim = AlifestdIplotxShimPandas(df)
    tree_data = shim(layout="horizontal")
    assert tree_data["rooted"] is True
    assert "vertex_df" in tree_data
    assert "edge_df" in tree_data
    assert "leaf_df" in tree_data
    assert len(tree_data["leaf_df"]) == 4
    assert len(tree_data["edge_df"]) == 6


def test_iplotx_call_polars():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            "origin_time": [0.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
        }
    )
    shim = AlifestdIplotxShimPolars(df)
    tree_data = shim(layout="horizontal")
    assert tree_data["rooted"] is True
    assert len(tree_data["leaf_df"]) == 4


def test_iplotx_call_numpy():
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
    branch_lengths = np.array([np.nan, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    shim = AlifestdIplotxShimNumpy(ancestor_ids, branch_lengths=branch_lengths)
    tree_data = shim(layout="horizontal")
    assert tree_data["rooted"] is True
    assert len(tree_data["leaf_df"]) == 4


def test_iplotx_call_no_branch_lengths():
    df = _make_balanced_pandas()
    shim = AlifestdIplotxShimPandas(df)
    tree_data = shim(layout="horizontal")
    assert tree_data["rooted"] is True
    assert len(tree_data["leaf_df"]) == 4


@pytest.mark.parametrize("layout", ["horizontal", "vertical", "radial"])
def test_iplotx_layouts(layout):
    df = _make_balanced_with_times_pandas()
    shim = AlifestdIplotxShimPandas(df)
    tree_data = shim(layout=layout)
    assert tree_data["layout_name"] == layout


def test_iplotx_call_single_node():
    df = _make_single_root_pandas()
    shim = AlifestdIplotxShimPandas(df)
    tree_data = shim(layout="horizontal")
    assert tree_data["rooted"] is True
    assert len(tree_data["leaf_df"]) == 1


def test_iplotx_leaf_labels():
    df = _make_with_taxon_labels_pandas()
    shim = AlifestdIplotxShimPandas(df)
    tree_data = shim(layout="horizontal", leaf_labels=True)
    assert "label" in tree_data["leaf_df"].columns


# ---- iplotx drawing integration tests ------------------------------
def test_iplotx_draw_tree_pandas():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _make_balanced_with_times_pandas()
    fig, ax = plt.subplots()
    artist = iplotx.tree(df, layout="horizontal", ax=ax, show=False)
    assert artist is not None
    assert len(ax.get_children()) > 0
    plt.close(fig)


def test_iplotx_draw_tree_polars():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            "origin_time": [0.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
        }
    )
    fig, ax = plt.subplots()
    artist = iplotx.tree(df, layout="horizontal", ax=ax, show=False)
    assert artist is not None
    assert len(ax.get_children()) > 0
    plt.close(fig)


@pytest.mark.parametrize("layout", ["horizontal", "vertical", "radial"])
def test_iplotx_draw_layouts(layout):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _make_balanced_with_times_pandas()
    fig, ax = plt.subplots()
    artist = iplotx.tree(df, layout=layout, ax=ax, show=False)
    assert artist is not None
    plt.close(fig)


def test_iplotx_draw_with_leaf_labels():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _make_with_taxon_labels_pandas()
    fig, ax = plt.subplots()
    artist = iplotx.tree(
        df, layout="horizontal", leaf_labels=True, ax=ax, show=False
    )
    assert artist is not None
    plt.close(fig)

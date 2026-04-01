import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
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
class TestAlifestdIplotxShimNumpy:
    def test_check_dependencies(self):
        assert AlifestdIplotxShimNumpy.check_dependencies() is True

    def test_tree_type(self):
        assert AlifestdIplotxShimNumpy.tree_type() is AlifestdIplotxShimNumpy

    def test_is_rooted(self):
        ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
        shim = AlifestdIplotxShimNumpy(ancestor_ids)
        assert shim.is_rooted() is True

    def test_get_root(self):
        ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
        shim = AlifestdIplotxShimNumpy(ancestor_ids)
        root = shim.get_root()
        assert root._id == 0

    def test_preorder(self):
        ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
        shim = AlifestdIplotxShimNumpy(ancestor_ids)
        order = [n._id for n in shim.preorder()]
        assert order == [0, 1, 3, 4, 2, 5, 6]

    def test_postorder(self):
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

    def test_levelorder(self):
        ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
        shim = AlifestdIplotxShimNumpy(ancestor_ids)
        order = [n._id for n in shim.levelorder()]
        assert order == [0, 1, 2, 3, 4, 5, 6]

    def test_get_leaves(self):
        ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
        shim = AlifestdIplotxShimNumpy(ancestor_ids)
        leaf_ids = sorted(n._id for n in shim._get_leaves())
        assert leaf_ids == [3, 4, 5, 6]

    def test_get_children(self):
        ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
        shim = AlifestdIplotxShimNumpy(ancestor_ids)
        root = shim.get_root()
        child_ids = [c._id for c in shim.get_children(root)]
        assert child_ids == [1, 2]

    def test_get_branch_length_none(self):
        ancestor_ids = np.array([0, 0, 0])
        shim = AlifestdIplotxShimNumpy(ancestor_ids)
        root = shim.get_root()
        assert shim.get_branch_length(root) is None

    def test_get_branch_length_with_values(self):
        ancestor_ids = np.array([0, 0, 0])
        branch_lengths = np.array([np.nan, 1.5, 2.5])
        shim = AlifestdIplotxShimNumpy(
            ancestor_ids, branch_lengths=branch_lengths
        )
        children = shim.get_children(shim.get_root())
        assert shim.get_branch_length(children[0]) == 1.5
        assert shim.get_branch_length(children[1]) == 2.5

    def test_names_default(self):
        ancestor_ids = np.array([0, 0])
        shim = AlifestdIplotxShimNumpy(ancestor_ids)
        assert shim.get_root().name == "0"

    def test_names_custom(self):
        ancestor_ids = np.array([0, 0])
        names = np.array(["root", "child"])
        shim = AlifestdIplotxShimNumpy(ancestor_ids, names=names)
        assert shim.get_root().name == "root"

    def test_single_node(self):
        ancestor_ids = np.array([0])
        shim = AlifestdIplotxShimNumpy(ancestor_ids)
        assert shim.get_root()._id == 0
        assert [n._id for n in shim.preorder()] == [0]
        assert [n._id for n in shim.postorder()] == [0]
        assert [n._id for n in shim.levelorder()] == [0]
        assert [n._id for n in shim._get_leaves()] == [0]
        assert shim.get_children(shim.get_root()) == []

    def test_chain(self):
        ancestor_ids = np.array([0, 0, 1, 2])
        shim = AlifestdIplotxShimNumpy(ancestor_ids)
        assert [n._id for n in shim.preorder()] == [0, 1, 2, 3]
        assert [n._id for n in shim.postorder()] == [3, 2, 1, 0]
        assert [n._id for n in shim._get_leaves()] == [3]

    def test_node_hashable(self):
        ancestor_ids = np.array([0, 0, 0])
        shim = AlifestdIplotxShimNumpy(ancestor_ids)
        nodes = list(shim.preorder())
        d = {node: i for i, node in enumerate(nodes)}
        assert len(d) == 3

    def test_node_equality(self):
        ancestor_ids = np.array([0, 0])
        shim = AlifestdIplotxShimNumpy(ancestor_ids)
        nodes1 = list(shim.preorder())
        nodes2 = list(shim.preorder())
        assert nodes1[0] == nodes2[0]
        assert nodes1[0] != nodes1[1]


# ---- AlifestdIplotxShimPandas tests ----------------------------------
class TestAlifestdIplotxShimPandas:
    def test_check_dependencies(self):
        assert AlifestdIplotxShimPandas.check_dependencies() is True

    def test_tree_type(self):
        assert AlifestdIplotxShimPandas.tree_type() is pd.DataFrame

    def test_balanced_tree(self):
        df = _make_balanced_pandas()
        shim = AlifestdIplotxShimPandas(df)
        assert shim.is_rooted() is True
        assert shim.get_root()._id == 0
        assert [n._id for n in shim.preorder()] == [0, 1, 3, 4, 2, 5, 6]
        assert sorted(n._id for n in shim._get_leaves()) == [3, 4, 5, 6]

    def test_chain(self):
        df = _make_chain_pandas(4)
        shim = AlifestdIplotxShimPandas(df)
        assert [n._id for n in shim.preorder()] == [0, 1, 2, 3]
        assert [n._id for n in shim._get_leaves()] == [3]

    def test_single_root(self):
        df = _make_single_root_pandas()
        shim = AlifestdIplotxShimPandas(df)
        assert shim.get_root()._id == 0
        assert [n._id for n in shim._get_leaves()] == [0]

    def test_branch_lengths_from_origin_time(self):
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

    def test_branch_lengths_from_delta(self):
        df = _make_balanced_with_deltas_pandas()
        shim = AlifestdIplotxShimPandas(df)
        root = shim.get_root()
        assert shim.get_branch_length(root) is None
        children = shim.get_children(root)
        for child in children:
            assert shim.get_branch_length(child) == pytest.approx(1.0)

    def test_taxon_labels(self):
        df = _make_with_taxon_labels_pandas()
        shim = AlifestdIplotxShimPandas(df)
        root = shim.get_root()
        assert root.name == "root"
        children = shim.get_children(root)
        child_names = sorted(c.name for c in children)
        assert child_names == ["left", "right"]

    def test_no_branch_lengths(self):
        df = _make_balanced_pandas()
        shim = AlifestdIplotxShimPandas(df)
        for node in shim.preorder():
            assert shim.get_branch_length(node) is None

    def test_from_ancestor_list(self):
        df = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_list": ["[None]", "[0]", "[0]"],
            }
        )
        shim = AlifestdIplotxShimPandas(df)
        assert shim.get_root()._id == 0
        assert sorted(n._id for n in shim._get_leaves()) == [1, 2]

    def test_non_contiguous_ids_raises(self):
        df = pd.DataFrame(
            {
                "id": [0, 5, 10],
                "ancestor_id": [0, 0, 5],
            }
        )
        with pytest.raises(NotImplementedError, match="contiguous"):
            AlifestdIplotxShimPandas(df)

    def test_not_topologically_sorted_raises(self):
        df = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
            }
        )
        with pytest.raises(NotImplementedError, match="topologically"):
            AlifestdIplotxShimPandas(df)

    def test_does_not_mutate_input(self):
        df = _make_balanced_pandas()
        original = df.copy()
        AlifestdIplotxShimPandas(df)
        assert df.equals(original)


# ---- AlifestdIplotxShimPolars tests ----------------------------------
class TestAlifestdIplotxShimPolars:
    def test_check_dependencies(self):
        assert AlifestdIplotxShimPolars.check_dependencies() is True

    def test_tree_type(self):
        assert AlifestdIplotxShimPolars.tree_type() is pl.DataFrame

    def test_balanced_tree(self):
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

    def test_chain(self):
        df = pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 1, 2],
            }
        )
        shim = AlifestdIplotxShimPolars(df)
        assert [n._id for n in shim.preorder()] == [0, 1, 2, 3]
        assert [n._id for n in shim._get_leaves()] == [3]

    def test_branch_lengths_from_origin_time(self):
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

    def test_branch_lengths_from_delta(self):
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

    def test_taxon_labels(self):
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

    def test_non_contiguous_ids_raises(self):
        df = pl.DataFrame(
            {
                "id": [0, 5, 10],
                "ancestor_id": [0, 0, 5],
            }
        )
        with pytest.raises(NotImplementedError, match="contiguous"):
            AlifestdIplotxShimPolars(df)

    def test_not_topologically_sorted_raises(self):
        df = pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
            }
        )
        with pytest.raises(NotImplementedError, match="topologically"):
            AlifestdIplotxShimPolars(df)


# ---- iplotx integration smoke tests --------------------------------
class TestIplotxIntegration:
    """Test that iplotx can actually use the providers end-to-end."""

    def test_iplotx_call_pandas(self):
        """Test the full TreeDataProvider.__call__ workflow with pandas."""
        df = _make_balanced_with_times_pandas()
        shim = AlifestdIplotxShimPandas(df)
        tree_data = shim(layout="horizontal")
        assert tree_data["rooted"] is True
        assert "vertex_df" in tree_data
        assert "edge_df" in tree_data
        assert "leaf_df" in tree_data
        assert len(tree_data["leaf_df"]) == 4
        assert len(tree_data["edge_df"]) == 6

    def test_iplotx_call_polars(self):
        """Test the full TreeDataProvider.__call__ workflow with polars."""
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

    def test_iplotx_call_numpy(self):
        """Test the full TreeDataProvider.__call__ workflow with numpy."""
        ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
        branch_lengths = np.array([np.nan, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        shim = AlifestdIplotxShimNumpy(
            ancestor_ids, branch_lengths=branch_lengths
        )
        tree_data = shim(layout="horizontal")
        assert tree_data["rooted"] is True
        assert len(tree_data["leaf_df"]) == 4

    def test_iplotx_call_no_branch_lengths(self):
        """Trees without branch length info should still work."""
        df = _make_balanced_pandas()
        shim = AlifestdIplotxShimPandas(df)
        tree_data = shim(layout="horizontal")
        assert tree_data["rooted"] is True
        assert len(tree_data["leaf_df"]) == 4

    @pytest.mark.parametrize("layout", ["horizontal", "vertical", "radial"])
    def test_iplotx_layouts(self, layout):
        """Test multiple layout options."""
        df = _make_balanced_with_times_pandas()
        shim = AlifestdIplotxShimPandas(df)
        tree_data = shim(layout=layout)
        assert tree_data["layout_name"] == layout

    def test_iplotx_call_single_node(self):
        """Single-node tree should work."""
        df = _make_single_root_pandas()
        shim = AlifestdIplotxShimPandas(df)
        tree_data = shim(layout="horizontal")
        assert tree_data["rooted"] is True
        assert len(tree_data["leaf_df"]) == 1

    def test_iplotx_leaf_labels(self):
        """Test leaf label assignment."""
        df = _make_with_taxon_labels_pandas()
        shim = AlifestdIplotxShimPandas(df)
        tree_data = shim(layout="horizontal", leaf_labels=True)
        assert "label" in tree_data["leaf_df"].columns

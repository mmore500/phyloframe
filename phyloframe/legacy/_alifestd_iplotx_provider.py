import typing

from iplotx.ingest.typing import TreeDataProvider
import numpy as np
import pandas as pd

from ._alifestd_find_leaf_ids import (
    _alifestd_find_leaf_ids_asexual_fast_path,
)
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import (
    alifestd_is_topologically_sorted,
)
from ._alifestd_try_add_ancestor_id_col import (
    alifestd_try_add_ancestor_id_col,
)
from ._alifestd_unfurl_traversal_levelorder_asexual import (
    alifestd_unfurl_traversal_levelorder_asexual,
)
from ._alifestd_unfurl_traversal_postorder_asexual import (
    alifestd_unfurl_traversal_postorder_asexual,
)
from ._alifestd_unfurl_traversal_preorder_asexual import (
    alifestd_unfurl_traversal_preorder_asexual,
)


class _AlifestdNode:
    """Lightweight hashable node wrapper for iplotx compatibility.

    Each node corresponds to one row in the alife-standard phylogeny
    dataframe, identified by its integer ``id``.
    """

    __slots__ = ("_id", "name", "branch_length", "_children")

    def __init__(
        self,
        id_: int,
        name: str = "",
        branch_length: typing.Optional[float] = None,
    ) -> None:
        self._id = id_
        self.name = name
        self.branch_length = branch_length
        self._children: typing.List["_AlifestdNode"] = []

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _AlifestdNode):
            return self._id == other._id
        return NotImplemented

    def __repr__(self) -> str:
        return f"_AlifestdNode(id={self._id}, name={self.name!r})"


def _build_nodes(
    ancestor_ids: np.ndarray,
    names: typing.Optional[np.ndarray] = None,
    branch_lengths: typing.Optional[np.ndarray] = None,
) -> typing.List[_AlifestdNode]:
    """Build node list from contiguous, topologically-sorted arrays.

    Parameters
    ----------
    ancestor_ids : np.ndarray
        Array of ancestor ids; roots have ``ancestor_ids[i] == i``.
    names : np.ndarray, optional
        Per-node name strings.  Defaults to ``str(id)``.
    branch_lengths : np.ndarray, optional
        Per-node branch lengths (from parent to this node).  ``None``
        entries are kept as-is for leaves/root.

    Returns
    -------
    nodes : list of _AlifestdNode
    """
    n = len(ancestor_ids)
    nodes: typing.List[_AlifestdNode] = []
    for i in range(n):
        name = str(names[i]) if names is not None else str(i)
        bl = (
            float(branch_lengths[i])
            if branch_lengths is not None and not np.isnan(branch_lengths[i])
            else None
        )
        nodes.append(_AlifestdNode(i, name, bl))

    for i in range(n):
        parent = int(ancestor_ids[i])
        if parent != i:
            nodes[parent]._children.append(nodes[i])

    return nodes


def _extract_branch_lengths(
    ancestor_ids: np.ndarray,
    origin_time_delta: typing.Optional[np.ndarray] = None,
    origin_time: typing.Optional[np.ndarray] = None,
) -> typing.Optional[np.ndarray]:
    """Extract branch lengths from numpy arrays.

    Uses ``origin_time_delta`` if present, otherwise computes from
    ``origin_time`` and ``ancestor_ids``.  Returns ``None`` if neither
    is available.
    """
    if origin_time_delta is not None:
        return origin_time_delta.astype(float)

    if origin_time is not None:
        ot = origin_time.astype(float)
        deltas = ot - ot[ancestor_ids]
        # Root self-references give 0; mark as NaN so they become None
        is_root = ancestor_ids == np.arange(len(ancestor_ids))
        deltas[is_root] = np.nan
        return deltas

    return None


class AlifestdIplotxShimNumpy(TreeDataProvider):
    """Numpy-backed iplotx ``TreeDataProvider`` for alife-standard data.

    This class assumes *contiguous* ids (``id == row index``) and
    *topologically sorted* rows (ancestors appear before descendants).

    Parameters
    ----------
    ancestor_ids : np.ndarray
        Integer array of ancestor ids; roots satisfy
        ``ancestor_ids[i] == i``.
    names : np.ndarray, optional
        Per-node name strings.
    branch_lengths : np.ndarray, optional
        Per-node branch lengths (edge from parent to this node).
    """

    def __init__(
        self,
        ancestor_ids: np.ndarray,
        names: typing.Optional[np.ndarray] = None,
        branch_lengths: typing.Optional[np.ndarray] = None,
    ) -> None:
        self._nodes = _build_nodes(ancestor_ids, names, branch_lengths)
        self._phylogeny_df = pd.DataFrame(
            {
                "id": np.arange(len(ancestor_ids)),
                "ancestor_id": ancestor_ids,
            }
        )
        self._ancestor_ids = ancestor_ids
        # Root is the first node where ancestor_id == id (guaranteed by
        # contiguous ids + topological sort).
        (root_indices,) = np.where(
            ancestor_ids == np.arange(len(ancestor_ids))
        )
        self._root = self._nodes[root_indices[0]]
        # Store a reference as ``tree`` for TreeDataProvider protocol.
        self.tree = self

    # -- TreeDataProvider interface ----------------------------------------

    def is_rooted(self) -> bool:
        return True

    def get_root(self) -> _AlifestdNode:
        return self._root

    def preorder(self) -> typing.Iterable[_AlifestdNode]:
        order = alifestd_unfurl_traversal_preorder_asexual(
            self._phylogeny_df,
            mutate=True,
        )
        return [self._nodes[i] for i in order]

    def postorder(self) -> typing.Iterable[_AlifestdNode]:
        order = alifestd_unfurl_traversal_postorder_asexual(
            self._phylogeny_df,
            mutate=True,
        )
        return [self._nodes[i] for i in order]

    def levelorder(self) -> typing.Iterable[_AlifestdNode]:
        order = alifestd_unfurl_traversal_levelorder_asexual(
            self._phylogeny_df,
            mutate=True,
        )
        return [self._nodes[i] for i in order]

    def _get_leaves(self) -> typing.Sequence[_AlifestdNode]:
        leaf_ids = _alifestd_find_leaf_ids_asexual_fast_path(
            self._ancestor_ids,
        )
        return [self._nodes[i] for i in leaf_ids]

    @staticmethod
    def get_children(
        node: _AlifestdNode,
    ) -> typing.Sequence[_AlifestdNode]:
        return node._children

    @staticmethod
    def get_branch_length(
        node: _AlifestdNode,
    ) -> typing.Optional[float]:
        return node.branch_length

    @staticmethod
    def check_dependencies() -> bool:
        return True

    @staticmethod
    def tree_type() -> type:
        return AlifestdIplotxShimNumpy


class AlifestdIplotxShimPandas(AlifestdIplotxShimNumpy):
    """Iplotx ``TreeDataProvider`` for *pandas* alife-standard dataframes.

    The dataframe must be asexual with contiguous ids and topologically
    sorted rows.  An ``ancestor_id`` column will be derived from
    ``ancestor_list`` if needed.

    Parameters
    ----------
    tree : pd.DataFrame
        Pandas phylogeny dataframe in alife standard format.
    """

    def __init__(self, tree: pd.DataFrame) -> None:
        df = alifestd_try_add_ancestor_id_col(tree.copy())
        if not alifestd_has_contiguous_ids(df):
            raise NotImplementedError(
                "AlifestdIplotxShimPandas requires contiguous ids "
                "(id == row index)."
            )
        if not alifestd_is_topologically_sorted(df):
            raise NotImplementedError(
                "AlifestdIplotxShimPandas requires topologically "
                "sorted rows."
            )

        ancestor_ids = df["ancestor_id"].to_numpy()
        names = (
            df["taxon_label"].astype(str).to_numpy()
            if "taxon_label" in df.columns
            else None
        )
        branch_lengths = _extract_branch_lengths(
            ancestor_ids,
            origin_time_delta=(
                df["origin_time_delta"].to_numpy()
                if "origin_time_delta" in df.columns
                else None
            ),
            origin_time=(
                df["origin_time"].to_numpy()
                if "origin_time" in df.columns
                else None
            ),
        )

        super().__init__(ancestor_ids, names, branch_lengths)
        self.tree = tree  # type: ignore[assignment]
        self._phylogeny_df = df[["id", "ancestor_id"]]

    @staticmethod
    def check_dependencies() -> bool:
        return True

    @staticmethod
    def tree_type() -> type:
        return pd.DataFrame


class AlifestdIplotxShimPolars(AlifestdIplotxShimNumpy):
    """Iplotx ``TreeDataProvider`` for *polars* alife-standard dataframes.

    The dataframe must be asexual with contiguous ids and topologically
    sorted rows, and must contain an ``ancestor_id`` column.

    Parameters
    ----------
    tree : polars.DataFrame
        Polars phylogeny dataframe in alife standard format.
    """

    def __init__(self, tree: "pl.DataFrame") -> None:  # noqa: F821
        import polars as pl

        from ._alifestd_has_contiguous_ids_polars import (
            alifestd_has_contiguous_ids_polars,
        )
        from ._alifestd_is_topologically_sorted_polars import (
            alifestd_is_topologically_sorted_polars,
        )

        if not alifestd_has_contiguous_ids_polars(tree):
            raise NotImplementedError(
                "AlifestdIplotxShimPolars requires contiguous ids "
                "(id == row index)."
            )
        if not alifestd_is_topologically_sorted_polars(tree):
            raise NotImplementedError(
                "AlifestdIplotxShimPolars requires topologically "
                "sorted rows."
            )

        ancestor_ids = tree["ancestor_id"].to_numpy()

        names = None
        if "taxon_label" in tree.columns:
            names = tree["taxon_label"].cast(pl.Utf8).to_numpy()

        branch_lengths = _extract_branch_lengths(
            ancestor_ids,
            origin_time_delta=(
                tree["origin_time_delta"].to_numpy()
                if "origin_time_delta" in tree.columns
                else None
            ),
            origin_time=(
                tree["origin_time"].to_numpy()
                if "origin_time" in tree.columns
                else None
            ),
        )

        super().__init__(ancestor_ids, names, branch_lengths)
        self.tree = tree  # type: ignore[assignment]

    @staticmethod
    def check_dependencies() -> bool:
        try:
            import polars  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def tree_type() -> type:
        import polars as pl

        return pl.DataFrame

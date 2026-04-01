import typing

from iplotx.ingest.typing import TreeDataProvider
import numpy as np
import pandas as pd

from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import (
    alifestd_is_topologically_sorted,
)
from ._alifestd_try_add_ancestor_id_col import (
    alifestd_try_add_ancestor_id_col,
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
) -> typing.Tuple[
    typing.List[_AlifestdNode],
    typing.List[_AlifestdNode],
    typing.List[_AlifestdNode],
]:
    """Build node graph from contiguous, topologically-sorted arrays.

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
    roots : list of _AlifestdNode
    leaves : list of _AlifestdNode
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

    roots: typing.List[_AlifestdNode] = []
    for i in range(n):
        parent = int(ancestor_ids[i])
        if parent == i:
            roots.append(nodes[i])
        else:
            nodes[parent]._children.append(nodes[i])

    leaves = [nd for nd in nodes if not nd._children]
    return nodes, roots, leaves


def _extract_branch_lengths_pandas(
    phylogeny_df: pd.DataFrame,
) -> typing.Optional[np.ndarray]:
    """Extract branch lengths from a pandas phylogeny dataframe.

    Uses ``origin_time_delta`` if present, otherwise computes from
    ``origin_time`` and ``ancestor_id``.  Returns ``None`` if neither
    column is available.
    """
    if "origin_time_delta" in phylogeny_df.columns:
        return phylogeny_df["origin_time_delta"].to_numpy(dtype=float)

    if "origin_time" in phylogeny_df.columns:
        origin_time = phylogeny_df["origin_time"].to_numpy(dtype=float)
        ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
        deltas = origin_time - origin_time[ancestor_ids]
        # Root self-references give 0; mark as NaN so they become None
        is_root = ancestor_ids == np.arange(len(ancestor_ids))
        deltas[is_root] = np.nan
        return deltas

    return None


class LegacyIplotxShimNumpy(TreeDataProvider):
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
        self._nodes, self._roots, self._leaves = _build_nodes(
            ancestor_ids, names, branch_lengths
        )
        # Store a reference as ``tree`` for TreeDataProvider protocol.
        self.tree = self

    # -- TreeDataProvider interface ----------------------------------------

    def is_rooted(self) -> bool:
        return True

    def get_root(self) -> _AlifestdNode:
        return self._roots[0]

    def preorder(self) -> typing.Iterable[_AlifestdNode]:
        stack = list(reversed(self._roots))
        while stack:
            node = stack.pop()
            yield node
            for child in reversed(node._children):
                stack.append(child)

    def postorder(self) -> typing.Iterable[_AlifestdNode]:
        # Two-stack iterative postorder
        stack1 = list(reversed(self._roots))
        result: typing.List[_AlifestdNode] = []
        while stack1:
            node = stack1.pop()
            result.append(node)
            for child in node._children:
                stack1.append(child)
        yield from reversed(result)

    def levelorder(self) -> typing.Iterable[_AlifestdNode]:
        from collections import deque

        queue: typing.Deque[_AlifestdNode] = deque(self._roots)
        while queue:
            node = queue.popleft()
            yield node
            for child in node._children:
                queue.append(child)

    def _get_leaves(self) -> typing.Sequence[_AlifestdNode]:
        return self._leaves

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
        return LegacyIplotxShimNumpy


class LegacyIplotxShimPandas(LegacyIplotxShimNumpy):
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
        self.tree = tree  # type: ignore[assignment]
        df = alifestd_try_add_ancestor_id_col(tree.copy())
        if not alifestd_has_contiguous_ids(df):
            raise NotImplementedError(
                "LegacyIplotxShimPandas requires contiguous ids "
                "(id == row index)."
            )
        if not alifestd_is_topologically_sorted(df):
            raise NotImplementedError(
                "LegacyIplotxShimPandas requires topologically " "sorted rows."
            )

        ancestor_ids = df["ancestor_id"].to_numpy()
        names = (
            df["taxon_label"].astype(str).to_numpy()
            if "taxon_label" in df.columns
            else None
        )
        branch_lengths = _extract_branch_lengths_pandas(df)

        self._nodes, self._roots, self._leaves = _build_nodes(
            ancestor_ids, names, branch_lengths
        )

    @staticmethod
    def check_dependencies() -> bool:
        return True

    @staticmethod
    def tree_type() -> type:
        return pd.DataFrame


class LegacyIplotxShimPolars(LegacyIplotxShimNumpy):
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

        self.tree = tree  # type: ignore[assignment]

        if not alifestd_has_contiguous_ids_polars(tree):
            raise NotImplementedError(
                "LegacyIplotxShimPolars requires contiguous ids "
                "(id == row index)."
            )
        if not alifestd_is_topologically_sorted_polars(tree):
            raise NotImplementedError(
                "LegacyIplotxShimPolars requires topologically " "sorted rows."
            )

        ancestor_ids = tree["ancestor_id"].to_numpy()

        names = None
        if "taxon_label" in tree.columns:
            names = tree["taxon_label"].cast(pl.Utf8).to_numpy()

        branch_lengths: typing.Optional[np.ndarray] = None
        if "origin_time_delta" in tree.columns:
            branch_lengths = tree["origin_time_delta"].to_numpy().astype(float)
        elif "origin_time" in tree.columns:
            origin_time = tree["origin_time"].to_numpy().astype(float)
            deltas = origin_time - origin_time[ancestor_ids]
            is_root = ancestor_ids == np.arange(len(ancestor_ids))
            deltas[is_root] = np.nan
            branch_lengths = deltas

        self._nodes, self._roots, self._leaves = _build_nodes(
            ancestor_ids, names, branch_lengths
        )

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

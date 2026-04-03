import typing

import numpy as np
import pandas as pd
import polars as pl

try:
    from iplotx.ingest.typing import TreeDataProvider
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    TreeDataProvider = object  # type: ignore[assignment,misc]

from ._alifestd_find_leaf_ids import (
    _alifestd_find_leaf_ids_asexual_fast_path,
)
from ._alifestd_find_root_ids import alifestd_find_root_ids
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted import (
    alifestd_is_topologically_sorted,
)
from ._alifestd_mark_csr_children_asexual import (
    _alifestd_mark_csr_children_asexual_fast_path,
)
from ._alifestd_mark_csr_offsets_asexual import (
    _alifestd_mark_csr_offsets_asexual_fast_path,
)
from ._alifestd_mark_origin_time_delta_asexual import (
    alifestd_mark_origin_time_delta_asexual,
)
from ._alifestd_try_add_ancestor_id_col import (
    alifestd_try_add_ancestor_id_col,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
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

    __slots__ = ("_id", "name", "branch_length")

    def __init__(
        self,
        id_: int,
        name: str = "",
        branch_length: typing.Optional[float] = None,
    ) -> None:
        self._id = id_
        self.name = name
        self.branch_length = branch_length

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _AlifestdNode):
            return self._id == other._id
        return NotImplemented

    def __repr__(self) -> str:
        return f"_AlifestdNode(id={self._id}, name={self.name!r})"


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
        if not alifestd_is_topologically_sorted(
            pd.DataFrame(
                {
                    "id": np.arange(len(ancestor_ids)),
                    "ancestor_id": ancestor_ids,
                }
            )
        ):
            raise NotImplementedError(
                "AlifestdIplotxShimNumpy requires topologically "
                "sorted rows."
            )

        n = len(ancestor_ids)
        nodes: typing.List[_AlifestdNode] = []
        for i in range(n):
            node_name = str(names[i]) if names is not None else str(i)
            node_bl = (
                None
                if branch_lengths is None or np.isnan(branch_lengths[i])
                else float(branch_lengths[i])
            )
            nodes.append(_AlifestdNode(i, node_name, node_bl))

        self._nodes = nodes
        self._phylogeny_df = pd.DataFrame(
            {
                "id": np.arange(n),
                "ancestor_id": ancestor_ids,
            }
        )
        self._ancestor_ids = ancestor_ids

        # CSR child storage — append sentinel (n-1) so slicing always works
        csr_offsets = _alifestd_mark_csr_offsets_asexual_fast_path(
            ancestor_ids,
        )
        self._csr_children = _alifestd_mark_csr_children_asexual_fast_path(
            ancestor_ids,
            csr_offsets,
        )
        self._csr_offsets = np.append(csr_offsets, n - 1)

        # Find and validate root
        root_ids = alifestd_find_root_ids(self._phylogeny_df)
        if len(root_ids) != 1:
            raise ValueError(
                f"Expected exactly 1 root, found {len(root_ids)}."
            )
        self._root = self._nodes[root_ids[0]]

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

    def get_children(
        self,
        node: _AlifestdNode,
    ) -> typing.Sequence[_AlifestdNode]:
        idx = node._id
        children_ids = self._csr_children[
            self._csr_offsets[idx] : self._csr_offsets[idx + 1]
        ]
        return [self._nodes[c] for c in children_ids]

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
    mutate : bool, default False
        If True, allow modification of the input dataframe.
    """

    def __init__(self, tree: pd.DataFrame, mutate: bool = False) -> None:
        if isinstance(tree, AlifestdIplotxShimPandas):
            self.__dict__.update(tree.__dict__)
            return
        if not mutate:
            tree = tree.copy()
        df = alifestd_try_add_ancestor_id_col(tree, mutate=True)
        if not alifestd_has_contiguous_ids(df):
            raise NotImplementedError(
                "AlifestdIplotxShimPandas requires contiguous ids "
                "(id == row index)."
            )

        ancestor_ids = df["ancestor_id"].to_numpy()
        names = (
            df["taxon_label"].astype(str).to_numpy()
            if "taxon_label" in df.columns
            else None
        )

        if (
            "origin_time_delta" not in df.columns
            and "origin_time" in df.columns
        ):
            df = alifestd_mark_origin_time_delta_asexual(df, mutate=True)
        branch_lengths = (
            df["origin_time_delta"].to_numpy()
            if "origin_time_delta" in df.columns
            else None
        )

        super().__init__(ancestor_ids, names, branch_lengths)
        self.tree = tree  # type: ignore[assignment]
        self._phylogeny_df = df[["id", "ancestor_id"]]

    @staticmethod
    def check_dependencies() -> bool:
        return True

    @staticmethod
    def tree_type() -> type:
        return AlifestdIplotxShimPandas


class AlifestdIplotxShimPolars(AlifestdIplotxShimNumpy):
    """Iplotx ``TreeDataProvider`` for *polars* alife-standard dataframes.

    The dataframe must be asexual with contiguous ids and topologically
    sorted rows.

    Parameters
    ----------
    tree : polars.DataFrame
        Polars phylogeny dataframe in alife standard format.
    """

    def __init__(self, tree: pl.DataFrame) -> None:
        if isinstance(tree, AlifestdIplotxShimPolars):
            self.__dict__.update(tree.__dict__)
            return

        tree_df = alifestd_try_add_ancestor_id_col_polars(tree)
        if not alifestd_has_contiguous_ids_polars(tree_df):
            raise NotImplementedError(
                "AlifestdIplotxShimPolars requires contiguous ids "
                "(id == row index)."
            )

        ancestor_ids = tree_df["ancestor_id"].to_numpy()

        names = None
        if "taxon_label" in tree_df.columns:
            names = tree_df["taxon_label"].cast(pl.Utf8).to_numpy()

        # Convert to pandas for origin_time_delta calculation
        if (
            "origin_time_delta" not in tree_df.columns
            and "origin_time" in tree_df.columns
        ):
            pdf = tree_df.to_pandas()
            pdf = alifestd_mark_origin_time_delta_asexual(pdf, mutate=True)
            branch_lengths = pdf["origin_time_delta"].to_numpy()
        elif "origin_time_delta" in tree_df.columns:
            branch_lengths = tree_df["origin_time_delta"].to_numpy()
        else:
            branch_lengths = None

        super().__init__(ancestor_ids, names, branch_lengths)
        self.tree = tree  # type: ignore[assignment]

    @staticmethod
    def check_dependencies() -> bool:
        try:
            import polars  # noqa: F401

            return True
        except ImportError:  # pragma: no cover
            return False

    @staticmethod
    def tree_type() -> type:
        return AlifestdIplotxShimPolars


def alifestd_to_iplotx_pandas(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
) -> AlifestdIplotxShimPandas:
    """Wrap a pandas phylogeny DataFrame for use with iplotx.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Asexual phylogeny in alife standard format with contiguous ids
        and topologically sorted rows.
    mutate : bool, default False
        If True, allow modification of the input dataframe.

    Returns
    -------
    AlifestdIplotxShimPandas
        An iplotx-compatible tree provider that can be passed directly
        to ``iplotx.tree()``.
    """
    return AlifestdIplotxShimPandas(phylogeny_df, mutate=mutate)


def alifestd_to_iplotx_polars(
    phylogeny_df: pl.DataFrame,
) -> AlifestdIplotxShimPolars:
    """Wrap a polars phylogeny DataFrame for use with iplotx.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        Asexual phylogeny in alife standard format with contiguous ids
        and topologically sorted rows.

    Returns
    -------
    AlifestdIplotxShimPolars
        An iplotx-compatible tree provider that can be passed directly
        to ``iplotx.tree()``.
    """
    return AlifestdIplotxShimPolars(phylogeny_df)

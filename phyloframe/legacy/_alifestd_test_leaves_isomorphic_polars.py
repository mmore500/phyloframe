import argparse
import logging
import os
import sys

import numpy as np
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._jit import jit
from .._auxlib._jit_numpy_bool_t import jit_numpy_bool_t
from .._auxlib._jit_numpy_int64_t import jit_numpy_int64_t
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_assign_contiguous_ids_polars import (
    alifestd_assign_contiguous_ids_polars,
)
from ._alifestd_collapse_unifurcations_polars import (
    alifestd_collapse_unifurcations_polars,
)
from ._alifestd_count_leaf_nodes_polars import (
    alifestd_count_leaf_nodes_polars,
)
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_leaves_polars import alifestd_mark_leaves_polars
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


@jit(nopython=True)
def _walk_leaves_isomorphic(
    ancestor_ids1: np.ndarray,
    ancestor_ids2: np.ndarray,
    leaf_ids1: np.ndarray,
    leaf_ids2: np.ndarray,
) -> jit_numpy_bool_t:
    """Walk ids backward and propagate the leaf-induced id mapping toward the
    roots, returning False if any node's ancestor mapping is inconsistent.

    Both phylogenies are assumed to be topologically sorted with contiguous
    ids. ``leaf_ids1[i]`` and ``leaf_ids2[i]`` give the corresponding ids in
    df1 and df2, respectively, for the ``i``th matched leaf.
    """
    n = len(ancestor_ids1)
    id_map = np.zeros(n, dtype=jit_numpy_int64_t)
    id_map_set = np.zeros(n, dtype=jit_numpy_bool_t)
    id_map[leaf_ids1] = leaf_ids2
    id_map_set[leaf_ids1] = True

    # iterate over ids from back to front; after collapse_unifurcations every
    # inner node has a leaf descendant that populates its mapping before the
    # walk reaches it.
    for id1 in range(n - 1, -1, -1):
        ancestor_id1 = ancestor_ids1[id1]
        id2 = id_map[id1]
        ancestor_id2 = ancestor_ids2[id2]
        if not id_map_set[ancestor_id1]:
            id_map[ancestor_id1] = ancestor_id2
            id_map_set[ancestor_id1] = True
        elif id_map[ancestor_id1] != ancestor_id2:
            return False
    return True


def _canonicalize(df: pl.LazyFrame) -> pl.LazyFrame:
    """Validate preconditions and prepare ``df`` for the leaf-isomorphism walk.

    Adds ``ancestor_id`` (if absent and derivable from ``ancestor_list``),
    drops ``ancestor_list``, collapses unifurcations, recompacts ids so
    the jit walk can index by id, and ensures ``is_leaf`` is present.
    """
    df = alifestd_try_add_ancestor_id_col_polars(df)
    if "ancestor_id" not in df.collect_schema().names():
        raise NotImplementedError("ancestor_id column required")
    if not alifestd_is_topologically_sorted_polars(df):
        raise NotImplementedError("topological sort not yet supported")
    if not alifestd_has_contiguous_ids_polars(df):
        raise NotImplementedError("non-contiguous ids not yet supported")
    df = (
        df.select(pl.exclude("ancestor_list"))
        .pipe(
            alifestd_collapse_unifurcations_polars,
            drop_topological_sensitivity=True,
        )
        .pipe(alifestd_assign_contiguous_ids_polars)
        .lazy()
    )
    if "is_leaf" not in df.collect_schema().names():
        df = alifestd_mark_leaves_polars(df)
    return df


def alifestd_test_leaves_isomorphic_polars(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    taxon_label: str,
) -> bool:
    """Test if phylogenetic relationships between leaf nodes are topologically
    isomorphic between two phylogenies.

    See Also
    --------
    alifestd_test_leaves_isomorphic_asexual :
        Pandas-based implementation.
    """
    if (
        df1.lazy().limit(1).collect().is_empty()
        and df2.lazy().limit(1).collect().is_empty()
    ):
        return True

    df1 = df1.lazy().pipe(_canonicalize)
    df2 = df2.lazy().pipe(_canonicalize)

    n_leaves1 = alifestd_count_leaf_nodes_polars(df1)
    n_leaves2 = alifestd_count_leaf_nodes_polars(df2)
    if n_leaves1 != n_leaves2:
        return False

    logging.info(
        "- alifestd_test_leaves_isomorphic_polars: collecting leaf ids...",
    )
    # sorting both leaf sets by taxon label aligns matching leaves
    # element-wise, side-stepping a join (and the column-aliasing it would
    # require to disambiguate "id" between the two frames).
    leaf_cols = ["id"] if taxon_label == "id" else ["id", taxon_label]
    leaves1_sorted = (
        df1.filter(pl.col("is_leaf"))
        .sort(taxon_label)
        .select(leaf_cols)
        .collect()
    )
    leaves2_sorted = (
        df2.filter(pl.col("is_leaf"))
        .sort(taxon_label)
        .select(leaf_cols)
        .collect()
    )

    if leaves1_sorted[taxon_label].n_unique() != n_leaves1:
        raise ValueError("taxon labels in df1 must be unique among leaves")
    if not leaves1_sorted[taxon_label].equals(leaves2_sorted[taxon_label]):
        return False

    logging.info(
        "- alifestd_test_leaves_isomorphic_polars: collecting ancestor ids...",
    )
    ancestor_ids1 = df1.select("ancestor_id").collect().to_series().to_numpy()
    ancestor_ids2 = df2.select("ancestor_id").collect().to_series().to_numpy()
    if len(ancestor_ids1) != len(ancestor_ids2):
        return False

    return bool(
        _walk_leaves_isomorphic(
            ancestor_ids1.astype(np.int64, copy=True),
            ancestor_ids2.astype(np.int64, copy=True),
            leaves1_sorted["id"].to_numpy().astype(np.int64),
            leaves2_sorted["id"].to_numpy().astype(np.int64),
        ),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()})

Test if phylogenetic relationships between leaf nodes are topologically
isomorphic between two phylogenies.

Return code 0 indicates isomorphism; 1 indicates non-isomorphism.

Note that this CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_test_leaves_isomorphic_asexual :
    CLI entrypoint for Pandas-based implementation.
"""


def _read_phylogeny(path: str) -> pl.DataFrame:
    ext = os.path.splitext(path.replace("csv.gz", "csvgz"))[1]
    return {
        ".csv": pl.read_csv,
        ".csvgz": pl.read_csv,
        ".fea": pl.read_ipc,
        ".feather": pl.read_ipc,
        ".pqt": pl.read_parquet,
        ".parquet": pl.read_parquet,
    }[ext](path)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=format_cli_description(_raw_description),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "first_phylogeny",
        type=str,
        help="Path to first alife-standard phylogeny file",
    )
    parser.add_argument(
        "second_phylogeny",
        type=str,
        help="Path to second alife-standard phylogeny file",
    )
    parser.add_argument(
        "-l",
        "--taxon-label",
        type=str,
        help="Name of column to use as taxon label.",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args = parser.parse_args()

    logging.info(f"reading first phylogeny from {args.first_phylogeny}...")
    first_df = _read_phylogeny(args.first_phylogeny)

    logging.info(f"reading second phylogeny from {args.second_phylogeny}...")
    second_df = _read_phylogeny(args.second_phylogeny)

    with log_context_duration(
        "phyloframe.legacy._alifestd_test_leaves_isomorphic_polars",
        logging.info,
    ):
        result = alifestd_test_leaves_isomorphic_polars(
            first_df,
            second_df,
            taxon_label=args.taxon_label,
        )

    exit_code = [1, 0][result]

    logging.info(f"exiting with return code {exit_code} for result {result}")
    sys.exit(exit_code)

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
    id_map: np.ndarray,
    id_map_set: np.ndarray,
) -> jit_numpy_bool_t:
    """Walk ids backward and propagate the leaf-induced id mapping toward the
    roots, returning False if any node's ancestor mapping is inconsistent.

    Both phylogenies are assumed to be topologically sorted with contiguous
    ids. ``id_map`` should be initialized so that ``id_map[id1] = id2`` for
    every leaf in df1 with matching taxon label in df2; ``id_map_set`` is a
    parallel boolean mask indicating which entries of ``id_map`` are valid.
    """
    n = jit_numpy_int64_t(len(ancestor_ids1))
    for offset in range(n):
        id1 = n - 1 - offset
        if not id_map_set[id1]:
            return False
        ancestor_id1 = ancestor_ids1[id1]
        id2 = id_map[id1]
        ancestor_id2 = ancestor_ids2[id2]
        if id_map_set[ancestor_id1]:
            if id_map[ancestor_id1] != ancestor_id2:
                return False
        else:
            id_map[ancestor_id1] = ancestor_id2
            id_map_set[ancestor_id1] = True
    return True


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
    df1 = df1.lazy()
    df2 = df2.lazy()

    if df1.limit(1).collect().is_empty() and df2.limit(1).collect().is_empty():
        return True

    df1 = alifestd_try_add_ancestor_id_col_polars(df1)
    df2 = alifestd_try_add_ancestor_id_col_polars(df2)

    if "ancestor_list" in df1.collect_schema().names():
        df1 = df1.drop("ancestor_list")
    if "ancestor_list" in df2.collect_schema().names():
        df2 = df2.drop("ancestor_list")

    if not alifestd_is_topologically_sorted_polars(
        df1
    ) or not alifestd_is_topologically_sorted_polars(df2):
        raise NotImplementedError("topological sort not yet supported")

    if not alifestd_has_contiguous_ids_polars(
        df1
    ) or not alifestd_has_contiguous_ids_polars(df2):
        raise NotImplementedError("non-contiguous ids not yet supported")

    logging.info(
        "- alifestd_test_leaves_isomorphic_polars: "
        "collapsing unifurcations...",
    )
    df1 = alifestd_collapse_unifurcations_polars(
        df1,
        ignore_topological_sensitivity=True,
        drop_topological_sensitivity=False,
    )
    df2 = alifestd_collapse_unifurcations_polars(
        df2,
        ignore_topological_sensitivity=True,
        drop_topological_sensitivity=False,
    )

    # collapse_unifurcations leaves gapped ids; recompact so the jit walk
    # below can index by id.
    df1 = alifestd_assign_contiguous_ids_polars(df1).lazy()
    df2 = alifestd_assign_contiguous_ids_polars(df2).lazy()

    logging.info(
        "- alifestd_test_leaves_isomorphic_polars: marking leaves...",
    )
    df1 = alifestd_mark_leaves_polars(df1)
    df2 = alifestd_mark_leaves_polars(df2)

    if taxon_label == "id":
        df1 = df1.with_columns(taxon_label=pl.col("id"))
        df2 = df2.with_columns(taxon_label=pl.col("id"))
        taxon_label = "taxon_label"

    logging.info(
        "- alifestd_test_leaves_isomorphic_polars: collecting leaf ids...",
    )
    leaves1 = (
        df1.filter(pl.col("is_leaf")).select(["id", taxon_label]).collect()
    )
    leaves2 = (
        df2.filter(pl.col("is_leaf"))
        .select([pl.col("id").alias("id2"), pl.col(taxon_label)])
        .collect()
    )

    if leaves1[taxon_label].n_unique() != leaves1.height:
        raise ValueError("taxon labels in df1 must be unique among leaves")
    if leaves2[taxon_label].n_unique() != leaves2.height:
        raise ValueError("taxon labels in df2 must be unique among leaves")

    leaf_pairs = leaves1.join(leaves2, on=taxon_label, how="inner")
    if leaf_pairs.height != leaves1.height or leaves1.height != leaves2.height:
        return False

    logging.info(
        "- alifestd_test_leaves_isomorphic_polars: collecting ancestor ids...",
    )
    ancestor_ids1 = df1.select("ancestor_id").collect().to_series().to_numpy()
    ancestor_ids2 = df2.select("ancestor_id").collect().to_series().to_numpy()
    if len(ancestor_ids1) != len(ancestor_ids2):
        return False

    n = len(ancestor_ids1)
    id_map = np.zeros(n, dtype=np.int64)
    id_map_set = np.zeros(n, dtype=np.bool_)
    leaf_ids1 = leaf_pairs["id"].to_numpy()
    leaf_ids2 = leaf_pairs["id2"].to_numpy()
    id_map[leaf_ids1] = leaf_ids2
    id_map_set[leaf_ids1] = True

    return bool(
        _walk_leaves_isomorphic(
            ancestor_ids1.astype(np.int64, copy=True),
            ancestor_ids2.astype(np.int64, copy=True),
            id_map,
            id_map_set,
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

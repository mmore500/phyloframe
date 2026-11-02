import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import pandas as pd

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._jit import jit
from .._auxlib._jit_TypingError import jit_TypingError
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_make_ancestor_list_col import alifestd_make_ancestor_list_col
from ._alifestd_topological_sensitivity_warned import (
    alifestd_topological_sensitivity_warned,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_regraft_splice_chronological_consistency_asexual_fast_path(
    ancestor_ids: np.ndarray,
    origin_times: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Implementation detail for
    `alifestd_regraft_splice_chronological_consistency_asexual` and
    `alifestd_regraft_splice_chronological_consistency_polars`.

    Requires contiguous ids and topologically sorted rows. Returns
    `ancestor_ids` extended with spliced nodes, which take ids following
    original ids, and, for each spliced node, the id of the taxon it was
    spliced for (whose origin time it takes).
    """
    num_taxa = len(ancestor_ids)
    # at most one node is spliced per taxon
    parents = np.empty(2 * num_taxa, dtype=np.int64)
    parents[:num_taxa] = ancestor_ids
    sources = np.empty(num_taxa, dtype=np.int64)
    num_spliced = 0

    for idx, ancestor_id in enumerate(ancestor_ids):
        origin_time = origin_times[idx]
        # walk up already-spliced lineage, where origin times are
        # nondecreasing, to find edge child <- ancestor that straddles taxon
        child, ancestor = idx, ancestor_id
        while ancestor != child:
            ancestor_time = (
                origin_times[ancestor]
                if ancestor < num_taxa
                else origin_times[sources[ancestor - num_taxa]]
            )
            if not ancestor_time > origin_time:
                break
            child, ancestor = ancestor, parents[ancestor]

        if child == idx:
            continue  # taxon is root or chronologically consistent

        spliced = num_taxa + num_spliced
        sources[num_spliced] = idx
        num_spliced += 1
        # if lineage exhausted, spliced node becomes new root
        parents[spliced] = spliced if ancestor == child else ancestor
        parents[child] = spliced
        parents[idx] = spliced

    return (
        parents[: num_taxa + num_spliced].copy(),
        sources[:num_spliced].copy(),
    )


def _alifestd_regraft_splice_chronological_consistency_asexual_impl(
    ancestor_ids: np.ndarray,
    origin_times: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Implementation detail for
    `alifestd_regraft_splice_chronological_consistency_asexual` and
    `alifestd_regraft_splice_chronological_consistency_polars`.

    Dispatches to jit-compiled fast path, falling back to its pure-Python
    version for origin times numba can't handle (e.g., str).
    """
    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    fast_path = (
        _alifestd_regraft_splice_chronological_consistency_asexual_fast_path
    )
    try:
        return fast_path(ancestor_ids, origin_times)
    except jit_TypingError:  # pragma: no cover
        return fast_path.__wrapped__(ancestor_ids, origin_times)


@alifestd_topological_sensitivity_warned(
    insert=True,
    delete=False,
    update=True,
)
def alifestd_regraft_splice_chronological_consistency_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
) -> pd.DataFrame:
    """For any taxa with origin time preceding its parent's origin time,
    splice in a new node where its lineage passes through its origin time and
    reattach the taxon there.

    Like `alifestd_regraft_chronological_consistency_asexual`, resolves
    origin time inversions by adjusting topology rather than origin times.
    However, rather than regrafting each inconsistent taxon onto its nearest
    eligible ancestor node, regrafts it onto its nearest eligible ancestor
    edge. So, the taxon remains grouped with the clade it originally
    belonged to, as sister to it.

    Specifically, each inconsistent taxon is regrafted, with its subtree,
    onto a new node spliced into the edge between the nearest taxon along its
    lineage with origin time not exceeding its own and that taxon's child
    along its lineage. Spliced nodes take the origin time of the taxon they
    are spliced for (i.e., a zero-length branch subtends the regrafted
    taxon). Taxa that precede their entire lineage are regrafted onto a new
    root spliced above their lineage's root. So, the number of roots is
    unchanged.

    Phylogeny must be asexual, topologically sorted, and have contiguous ids;
    otherwise ``NotImplementedError`` is raised. Original taxa keep their ids
    and rows. Spliced nodes are appended as new rows, with ids following
    original ids and null values for columns other than `id`,
    `ancestor_id`, `ancestor_list`, and `origin_time`. So, output may not be
    topologically sorted. If present, `ancestor_list` is updated for
    regrafted and spliced taxa.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_regraft_splice_chronological_consistency_polars :
        Polars-based implementation.
    alifestd_regraft_chronological_consistency_asexual :
        Regrafts onto nearest eligible ancestor node, without splicing.
    alifestd_coerce_chronological_consistency :
        Resolves origin time inversions by adjusting origin times.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    if "ancestor_id" not in phylogeny_df.columns:
        raise NotImplementedError(
            "alifestd_regraft_splice_chronological_consistency_asexual only "
            "supports asexual phylogenies.",
        )

    if phylogeny_df.empty:
        return phylogeny_df

    if not alifestd_has_contiguous_ids(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")

    if not alifestd_is_topologically_sorted(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    origin_times = phylogeny_df["origin_time"].to_numpy()
    (
        regrafted,
        sources,
    ) = _alifestd_regraft_splice_chronological_consistency_asexual_impl(
        ancestor_ids,
        origin_times,
    )

    num_taxa = len(phylogeny_df)
    id_dtype = phylogeny_df["id"].dtype
    ancestor_id_dtype = phylogeny_df["ancestor_id"].dtype
    is_regrafted = regrafted[:num_taxa] != ancestor_ids
    phylogeny_df["ancestor_id"] = regrafted[:num_taxa].astype(
        ancestor_id_dtype,
    )

    if len(sources) == 0:
        return phylogeny_df

    spliced_df = pd.DataFrame(
        {
            "id": np.arange(num_taxa, len(regrafted)).astype(id_dtype),
            "ancestor_id": regrafted[num_taxa:].astype(ancestor_id_dtype),
            "origin_time": pd.Series(
                origin_times[sources],
                dtype=phylogeny_df["origin_time"].dtype,
            ),
        },
    )

    if "ancestor_list" in phylogeny_df.columns:
        phylogeny_df.loc[
            is_regrafted, "ancestor_list"
        ] = alifestd_make_ancestor_list_col(
            phylogeny_df.loc[is_regrafted, "id"],
            phylogeny_df.loc[is_regrafted, "ancestor_id"],
        )
        spliced_df["ancestor_list"] = alifestd_make_ancestor_list_col(
            spliced_df["id"],
            spliced_df["ancestor_id"],
        )

    return pd.concat([phylogeny_df, spliced_df], ignore_index=True)


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

For any taxa with origin time preceding its parent's, splice in a new node where its lineage passes through its origin time and reattach the taxon there.

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires asexual phylogeny with contiguous ids and topologically sorted rows.

- Spliced nodes are appended as new rows, so output may not be topologically sorted.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_regraft_splice_chronological_consistency_polars :
    Entrypoint for high-performance Polars-based implementation.
"""


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        description=format_cli_description(_raw_description),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = _add_parser_base(
        parser=parser,
        dfcli_module="phyloframe.legacy._alifestd_regraft_splice_chronological_consistency_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    add_bool_arg(
        parser,
        "ignore-topological-sensitivity",
        default=False,
        help="suppress topological sensitivity warning (default: False)",
    )
    add_bool_arg(
        parser,
        "drop-topological-sensitivity",
        default=False,
        help="drop topology-sensitive columns from output (default: False)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_regraft_splice_chronological_consistency_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_regraft_splice_chronological_consistency_asexual,
                    ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                    drop_topological_sensitivity=args.drop_topological_sensitivity,
                ),
            ),
        )

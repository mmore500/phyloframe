import argparse
import functools
import logging
import os

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
def _alifestd_regraft_chronological_consistency_asexual_fast_path(
    ancestor_ids: np.ndarray,
    origin_times: np.ndarray,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_regraft_chronological_consistency_asexual` and
    `alifestd_regraft_chronological_consistency_polars`.

    Requires contiguous ids and topologically sorted rows. Returns a regrafted
    copy of `ancestor_ids`, with each taxon reattached to its nearest
    ancestor that does not originate after it. Taxa with no such ancestor
    become roots. Roots are left unchanged.
    """
    result = ancestor_ids.copy()
    for idx in range(len(result)):
        ancestor_id = result[idx]
        # ancestors are already regrafted, so origin times are nondecreasing
        # down their lineage and the first eligible ancestor is the nearest
        while (
            ancestor_id != idx
            and origin_times[ancestor_id] > origin_times[idx]
        ):
            if result[ancestor_id] == ancestor_id:  # exhausted lineage
                ancestor_id = idx  # ... so, become root
            else:
                ancestor_id = result[ancestor_id]

        result[idx] = ancestor_id

    return result


def _alifestd_regraft_chronological_consistency_asexual_impl(
    ancestor_ids: np.ndarray,
    origin_times: np.ndarray,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_regraft_chronological_consistency_asexual` and
    `alifestd_regraft_chronological_consistency_polars`.

    Dispatches to jit-compiled fast path, falling back to its pure-Python
    version for origin times numba can't handle (e.g., str).
    """
    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    fast_path = _alifestd_regraft_chronological_consistency_asexual_fast_path
    try:
        return fast_path(ancestor_ids, origin_times)
    except jit_TypingError:  # pragma: no cover
        # this does get exercised in tests, but not in coverage
        # because jit is disabled for better visibility of jit code
        return fast_path.__wrapped__(ancestor_ids, origin_times)


@alifestd_topological_sensitivity_warned(
    insert=False,
    delete=False,
    update=True,
)
def alifestd_regraft_chronological_consistency_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
) -> pd.DataFrame:
    """For any taxa with origin time preceding its parent's origin time,
    reattach taxon to its nearest ancestor that it does not precede.

    Whereas `alifestd_coerce_chronological_consistency` resolves origin time
    inversions by adjusting origin times, this function resolves them by
    adjusting topology. Origin times are left unchanged.

    Each inconsistent taxon is regrafted, with its subtree, onto the nearest
    taxon along its lineage with origin time not exceeding the taxon's own
    origin time. Taxa that precede their entire lineage become roots. Root
    taxa are never adjusted.

    Phylogeny must be asexual, topologically sorted, and have contiguous ids;
    otherwise ``NotImplementedError`` is raised.
    Output remains topologically sorted. If present, `ancestor_list` is
    updated for regrafted taxa.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_regraft_chronological_consistency_polars :
        Polars-based implementation.
    alifestd_coerce_chronological_consistency :
        Resolves origin time inversions by adjusting origin times.
    alifestd_find_chronological_inconsistency :
        Finds taxa with origin time inversions.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    if "ancestor_id" not in phylogeny_df.columns:
        raise NotImplementedError(
            "alifestd_regraft_chronological_consistency_asexual only "
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
    regrafted = _alifestd_regraft_chronological_consistency_asexual_impl(
        ancestor_ids,
        phylogeny_df["origin_time"].to_numpy(),
    )

    phylogeny_df["ancestor_id"] = regrafted.astype(
        phylogeny_df["ancestor_id"].dtype,
    )

    if "ancestor_list" in phylogeny_df.columns:
        is_regrafted = regrafted != ancestor_ids
        phylogeny_df.loc[
            is_regrafted, "ancestor_list"
        ] = alifestd_make_ancestor_list_col(
            phylogeny_df.loc[is_regrafted, "id"],
            phylogeny_df.loc[is_regrafted, "ancestor_id"],
        )

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

For any taxa with origin time preceding its parent's, reattach taxon to its nearest ancestor that it does not precede (or make it a root, if none).

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires asexual phylogeny with contiguous ids and topologically sorted rows.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_regraft_chronological_consistency_polars :
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
        dfcli_module="phyloframe.legacy._alifestd_regraft_chronological_consistency_asexual",
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
        "phyloframe.legacy._alifestd_regraft_chronological_consistency_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_regraft_chronological_consistency_asexual,
                    ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                    drop_topological_sensitivity=args.drop_topological_sensitivity,
                ),
            ),
        )

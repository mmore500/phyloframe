import argparse
import functools
import logging
import numbers
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import opytional as opyt
import pandas as pd

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._jit import jit
from .._auxlib._jit_TypingError import jit_TypingError
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_find_chronological_inconsistency import (
    alifestd_find_chronological_inconsistency,
)
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_parse_ancestor_ids import alifestd_parse_ancestor_ids
from ._alifestd_topological_sort import alifestd_topological_sort
from ._alifestd_try_add_ancestor_list_col import (
    alifestd_try_add_ancestor_list_col,
)


@jit(nopython=True)
def _alifestd_coerce_chronological_consistency_asexual_fast_path(
    ancestor_ids: np.ndarray,
    origin_times: np.ndarray,
    epsilon: numbers.Number,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_coerce_chronological_consistency` and
    `alifestd_coerce_chronological_consistency_polars`.

    Requires contiguous ids and topologically sorted rows. Returns a corrected
    copy of `origin_times`, with each taxon's origin time raised to at least
    its parent's origin time plus `epsilon`. Roots are left unchanged.
    """
    result = origin_times.copy()
    for idx, ancestor_id in enumerate(ancestor_ids):
        if ancestor_id == idx:
            continue  # handle root cases

        result[idx] = max(result[idx], result[ancestor_id] + epsilon)

    return result


def alifestd_coerce_chronological_consistency(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    epsilon: typing.Optional[numbers.Number] = None,
) -> pd.DataFrame:
    """For any taxa with origin time preceding its parent's origin time plus
    `epsilon`, set origin time to parent's origin time plus
    `epsilon`.

    With the default `epsilon` of None, taxa are only required not to
    precede their parents; unlike `epsilon=0`, this also supports
    non-numeric origin times (e.g., str). A positive `epsilon` requires each taxon to
    originate at least `epsilon` after its parent(s). Root taxa are
    never adjusted.

    If a correction may be needed and the phylogeny is not topologically
    sorted, the corrected phylogeny will be returned sorted in topological
    order.

    For topologically sorted asexual phylogenies with an `ancestor_id`
    column, contiguous ids, and numeric origin times, an optimized
    implementation is used.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_coerce_chronological_consistency_polars :
        Polars-based implementation.
    alifestd_graft_chronological_consistency_asexual :
        Resolves origin time inversions by grafting onto nearest
        eligible ancestor node.
    alifestd_splice_chronological_consistency_asexual :
        Resolves origin time inversions by grafting onto a node spliced
        into nearest eligible ancestor edge.
    alifestd_find_chronological_inconsistency :
        Finds taxa with origin time inversions.
    """

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    is_topologically_sorted = alifestd_is_topologically_sorted(phylogeny_df)
    if (
        is_topologically_sorted
        and "ancestor_id" in phylogeny_df.columns
        and alifestd_has_contiguous_ids(phylogeny_df)
        and not phylogeny_df.empty
    ):
        # optimized implementation for asexual phylogenies w/ contiguous ids
        for unwrap in lambda x: x, lambda x: x.__wrapped__:
            fast_path = unwrap(
                _alifestd_coerce_chronological_consistency_asexual_fast_path,
            )
            try:
                origin_times = phylogeny_df["origin_time"].to_numpy()
                # promote up front: writes into an int array truncate, e.g.,
                # 3 + 0.5 stays 3; numpy rules keep int32/float32 as is
                dtype = np.result_type(
                    origin_times.dtype, opyt.or_value(epsilon, 0)
                )
                phylogeny_df["origin_time"] = fast_path(
                    phylogeny_df["ancestor_id"].to_numpy().astype(np.int64),
                    origin_times.astype(dtype),
                    opyt.or_value(epsilon, 0),
                )
                return phylogeny_df.set_index("id", drop=False)
            except jit_TypingError:  # pragma: no cover
                continue
            except (TypeError, ValueError):
                break  # e.g., origin_times are strings or sequences

    # slower fallback implementation
    if not is_topologically_sorted:
        if (
            alifestd_find_chronological_inconsistency(
                phylogeny_df, epsilon=epsilon
            )
            is None
        ):
            return phylogeny_df.set_index("id", drop=False)
        phylogeny_df = alifestd_topological_sort(phylogeny_df, mutate=True)

    has_ancestor_list = "ancestor_list" in phylogeny_df
    if not has_ancestor_list:
        phylogeny_df = alifestd_try_add_ancestor_list_col(
            phylogeny_df, mutate=True
        )

    if epsilon is not None:
        dtype = np.result_type(phylogeny_df["origin_time"].dtype, epsilon)
        phylogeny_df["origin_time"] = phylogeny_df["origin_time"].astype(dtype)

    phylogeny_df = phylogeny_df.set_index("id", drop=False)

    # adapted from https://stackoverflow.com/a/31569794
    origin_time_loc = phylogeny_df.columns.get_loc("origin_time")
    for pos, (_idx, row) in enumerate(phylogeny_df.iterrows()):
        for ancestor_id in alifestd_parse_ancestor_ids(row["ancestor_list"]):
            ancestor_loc = phylogeny_df.index.get_loc(ancestor_id)
            ancestor_time = phylogeny_df.iat[ancestor_loc, origin_time_loc]
            if epsilon is not None:
                ancestor_time += epsilon
            offspring_time = phylogeny_df.iat[pos, origin_time_loc]
            phylogeny_df.iat[pos, origin_time_loc] = max(
                offspring_time, ancestor_time
            )

    if not has_ancestor_list:
        phylogeny_df.drop(columns=["ancestor_list"], inplace=True)

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

For any taxa with origin time preceding its parent's, set origin time to parent's origin time.

Use `--epsilon` to require that each taxon originate at least that much after its parent(s).

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.
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
        dfcli_module="phyloframe.legacy._alifestd_coerce_chronological_consistency",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--epsilon",
        default=None,
        type=float,
        help=(
            "minimum origin time gap enforced between each taxon and its "
            "parent(s) (default: None)"
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_coerce_chronological_consistency",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_coerce_chronological_consistency,
                    epsilon=args.epsilon,
                ),
            ),
        )

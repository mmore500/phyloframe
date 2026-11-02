import argparse
import functools
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._jit_TypingError import jit_TypingError
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_coerce_chronological_consistency import (
    _alifestd_coerce_chronological_consistency_asexual_fast_path,
)
from ._alifestd_find_chronological_inconsistency_polars import (
    alifestd_find_chronological_inconsistency_polars,
)
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_coerce_chronological_consistency_polars(
    phylogeny_df: pl.DataFrame,
    *,
    epsilon: float = 0,
) -> pl.DataFrame:
    """For any taxa with origin time preceding its parent's origin time plus
    `epsilon`, set origin time to parent's origin time plus
    `epsilon`.

    With the default `epsilon` of zero, taxa are only required not to
    precede their parents. A positive `epsilon` requires each taxon to
    originate at least `epsilon` after its parent. Root taxa are never
    adjusted.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids, numeric
        `origin_time`, and topologically sorted rows.
    epsilon : float, default 0
        Minimum origin time gap enforced between each taxon and its parent.

    Returns
    -------
    polars.DataFrame
        The phylogeny with corrected `origin_time` column.

    See Also
    --------
    alifestd_coerce_chronological_consistency :
        Pandas-based implementation.
    """

    logging.info(
        "- alifestd_coerce_chronological_consistency_polars: "
        "adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    schema = phylogeny_df.lazy().collect_schema()
    if "ancestor_id" not in schema.names():
        raise NotImplementedError("sexual phylogenies not supported")

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.lazy().collect()

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")

    if (
        epsilon == 0
        and alifestd_find_chronological_inconsistency_polars(phylogeny_df)
        is None
    ):
        return phylogeny_df.lazy().collect()

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    logging.info(
        "- alifestd_coerce_chronological_consistency_polars: "
        "extracting columns...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
        .astype(np.int64)
    )
    origin_times = (
        phylogeny_df.lazy()
        .select("origin_time")
        .collect()
        .to_series()
        .to_numpy()
    )

    logging.info(
        "- alifestd_coerce_chronological_consistency_polars: "
        "coercing origin times...",
    )
    try:
        # promote dtype so that, e.g., int origin_time + 0.5 isn't truncated
        dtype = np.result_type(origin_times.dtype, epsilon)
        corrected = (
            _alifestd_coerce_chronological_consistency_asexual_fast_path(
                ancestor_ids, origin_times.astype(dtype), epsilon
            )
        )
    except (TypeError, jit_TypingError):
        raise NotImplementedError("non-numeric origin_time not supported")

    return (
        phylogeny_df.lazy()
        .collect()
        .with_columns(pl.Series("origin_time", corrected))
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

For any taxa with origin time preceding its parent's, set origin time to parent's origin time.

Use `--epsilon` to require that each taxon originate at least that much after its parent.

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires asexual phylogeny with contiguous ids and topologically sorted rows.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_coerce_chronological_consistency :
    CLI entrypoint for Pandas-based implementation.
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
        dfcli_module="phyloframe.legacy._alifestd_coerce_chronological_consistency_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--epsilon",
        default=0,
        type=float,
        help=(
            "minimum origin time gap enforced between each taxon and its "
            "parent (default: 0)"
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_coerce_chronological_consistency_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_coerce_chronological_consistency_polars,
                    epsilon=args.epsilon,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e

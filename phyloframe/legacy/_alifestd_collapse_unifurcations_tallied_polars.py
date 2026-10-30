import argparse
import functools
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_collapse_unifurcations_polars import (
    alifestd_collapse_unifurcations_polars,
)
from ._alifestd_mark_unifurcation_length_polars import (
    alifestd_mark_unifurcation_length_polars,
)
from ._alifestd_topological_sensitivity_warned_polars import (
    alifestd_topological_sensitivity_warned_polars,
)


@alifestd_topological_sensitivity_warned_polars(
    insert=False,
    delete=True,
    update=True,
)
def alifestd_collapse_unifurcations_tallied_polars(
    phylogeny_df: pl.DataFrame,
    *,
    tally_as: str = "collapsed_unifurcations",
) -> pl.DataFrame:
    """Pare record to bypass organisms with one ancestor and one descendant,
    tallying bypassed organisms into column `collapsed_unifurcations`.

    Each remaining node's tally is incremented by the number of
    unifurcations collapsed between it and its new ancestor, plus any
    tallies those unifurcations already held (e.g., from an earlier
    collapse). So, repeated collapses accumulate correct tallies.

    If the tally column is absent, tallies start from zero. The tally column
    name can be changed via the ``tally_as`` parameter. Null or NaN tallies
    are not supported; ``NotImplementedError`` is raised.

    Phylogeny must be asexual, topologically sorted, and have
    contiguous ids; otherwise ``NotImplementedError`` is raised.

    See Also
    --------
    alifestd_collapse_unifurcations_tallied_asexual :
        Pandas-based implementation.
    alifestd_collapse_unifurcations_polars :
        Collapses unifurcations without tallying them.
    alifestd_mark_unifurcation_length_polars :
        Measures unifurcations that would be collapsed.
    """
    if tally_as not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = phylogeny_df.with_columns(
            pl.lit(0, dtype=pl.UInt32).alias(tally_as),
        )
    schema = phylogeny_df.lazy().collect_schema()

    tally = pl.col(tally_as)
    is_missing = tally.is_null()
    if schema[tally_as].is_float():
        is_missing = is_missing | tally.is_nan()
    if phylogeny_df.lazy().select(is_missing.any()).collect().item():
        raise NotImplementedError("NaN or null values not yet supported")

    with log_context_duration(
        "alifestd_mark_unifurcation_length_polars", logging.info
    ):
        phylogeny_df = alifestd_mark_unifurcation_length_polars(
            phylogeny_df, mark_as="_tally_length", weights=tally + 1
        )
    phylogeny_df = phylogeny_df.with_columns(
        (tally + pl.col("_tally_length")).cast(schema[tally_as]),
    ).drop("_tally_length")

    with log_context_duration(
        "alifestd_collapse_unifurcations_polars", logging.info
    ):
        return alifestd_collapse_unifurcations_polars(
            phylogeny_df, ignore_topological_sensitivity=True
        )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Pare record to bypass organisms with one ancestor and one descendant, tallying bypassed organisms into column `collapsed_unifurcations`.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires 'ancestor_id' column to be present in input DataFrame.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_collapse_unifurcations_tallied_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_collapse_unifurcations_tallied_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--tally-as",
        default="collapsed_unifurcations",
        type=str,
        help="tally column name (default: collapsed_unifurcations)",
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

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_collapse_unifurcations_tallied_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_collapse_unifurcations_tallied_polars,
                    tally_as=args.tally_as,
                    ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                    drop_topological_sensitivity=args.drop_topological_sensitivity,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e

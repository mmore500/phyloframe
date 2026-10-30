import argparse
import functools
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import pandas as pd

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_collapse_unifurcations import alifestd_collapse_unifurcations
from ._alifestd_mark_unifurcation_length_asexual import (
    alifestd_mark_unifurcation_length_asexual,
)
from ._alifestd_topological_sensitivity_warned import (
    alifestd_topological_sensitivity_warned,
)


@alifestd_topological_sensitivity_warned(
    insert=False,
    delete=True,
    update=True,
)
def alifestd_collapse_unifurcations_tallied_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    tally_as: str = "collapsed_unifurcations",
) -> pd.DataFrame:
    """Pare record to bypass organisms with one ancestor and one descendant,
    tallying bypassed organisms into column `collapsed_unifurcations`.

    Each remaining node's tally is incremented by the number of
    unifurcations collapsed between it and its new ancestor, plus any
    tallies those unifurcations already held (e.g., from an earlier
    collapse). So, repeated collapses accumulate correct tallies.

    If the tally column is absent, tallies start from zero. The tally column
    name can be changed via the ``tally_as`` parameter. NaN tallies are not
    supported; ``NotImplementedError`` is raised.

    Phylogeny must be asexual, topologically sorted, and have
    contiguous ids; otherwise ``NotImplementedError`` is raised.

    Input dataframe is not mutated by this operation unless ``mutate``
    is set True. If mutate set True, operation does not occur in place;
    still use return value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_collapse_unifurcations_tallied_polars :
        Polars-based implementation.
    alifestd_collapse_unifurcations :
        Collapses unifurcations without tallying them.
    alifestd_mark_unifurcation_length_asexual :
        Measures unifurcations that would be collapsed.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if tally_as not in phylogeny_df.columns:
        phylogeny_df[tally_as] = 0
    if phylogeny_df[tally_as].isna().any():
        raise NotImplementedError("NaN values not yet supported")

    phylogeny_df["_tally_weight"] = phylogeny_df[tally_as] + 1
    with log_context_duration(
        "alifestd_mark_unifurcation_length_asexual", logging.info
    ):
        phylogeny_df = alifestd_mark_unifurcation_length_asexual(
            phylogeny_df,
            mutate=True,
            mark_as="_tally_length",
            weights="_tally_weight",
        )
    phylogeny_df[tally_as] = (
        phylogeny_df[tally_as] + phylogeny_df["_tally_length"]
    ).astype(phylogeny_df[tally_as].dtype)
    phylogeny_df.drop(columns=["_tally_length", "_tally_weight"], inplace=True)

    with log_context_duration("alifestd_collapse_unifurcations", logging.info):
        return alifestd_collapse_unifurcations(
            phylogeny_df, mutate=True, ignore_topological_sensitivity=True
        )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Pare record to bypass organisms with one ancestor and one descendant, tallying bypassed organisms into column `collapsed_unifurcations`.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_collapse_unifurcations_tallied_polars :
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
        dfcli_module="phyloframe.legacy._alifestd_collapse_unifurcations_tallied_asexual",
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
    with log_context_duration(
        "phyloframe.legacy._alifestd_collapse_unifurcations_tallied_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_collapse_unifurcations_tallied_asexual,
                    tally_as=args.tally_as,
                    ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                    drop_topological_sensitivity=args.drop_topological_sensitivity,
                ),
            ),
        )

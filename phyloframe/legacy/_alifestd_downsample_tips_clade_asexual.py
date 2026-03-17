import argparse
import functools
import logging
import os
import sys
import typing

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
from ._alifestd_mark_sample_tips_clade_asexual import (
    alifestd_mark_sample_tips_clade_asexual,
)
from ._alifestd_prune_extinct_lineages_asexual import (
    alifestd_prune_extinct_lineages_asexual,
)
from ._alifestd_topological_sensitivity_warned import (
    alifestd_topological_sensitivity_warned,
)


@alifestd_topological_sensitivity_warned(
    insert=False,
    delete=True,
    update=False,
)
def alifestd_downsample_tips_clade_asexual(
    phylogeny_df: pd.DataFrame,
    n_downsample: int,
    mutate: bool = False,
    seed: typing.Optional[int] = None,
) -> pd.DataFrame:
    """Create a subsample phylogeny containing at most `n_downsample` tips,
    comprising a single clade within the original phylogeny. Candidate clades
    are sampled proportionally to their size.

    If `n_downsample` is greater than the number of tips in the phylogeny,
    the whole phylogeny is returned.

    Only supports asexual phylogenies.
    """
    phylogeny_df = alifestd_mark_sample_tips_clade_asexual(
        phylogeny_df,
        n_downsample,
        mutate=mutate,
        seed=seed,
        mark_as="extant",
    )

    if phylogeny_df.empty:
        return phylogeny_df.drop(columns=["extant"])

    return alifestd_prune_extinct_lineages_asexual(
        phylogeny_df, mutate=True
    ).drop(columns=["extant"])


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Create a subsample phylogeny containing at most `-n` tips, comprising a single clade within the original phylogeny.
Candidate clades are sampled proportionally to their tip count.

If `-n` is greater than the number of tips in the phylogeny, the whole phylogeny is returned.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires 'ancestor_id' column to be present in input DataFrame.
Otherwise, no action is taken.

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
        dfcli_module="phyloframe.legacy._alifestd_downsample_tips_clade_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "-n",
        default=sys.maxsize,
        type=int,
        help="Number of tips to subsample.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        dest="seed",
        help="Integer seed for deterministic behavior.",
        type=int,
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
        "phyloframe.legacy._alifestd_downsample_tips_clade_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_downsample_tips_clade_asexual,
                    n_downsample=args.n,
                    seed=args.seed,
                    ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                    drop_topological_sensitivity=args.drop_topological_sensitivity,
                ),
            ),
            overridden_arguments="ignore",  # seed is overridden
        )

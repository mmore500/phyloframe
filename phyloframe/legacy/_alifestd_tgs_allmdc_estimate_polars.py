import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._identity import identity
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._warn_experimental import warn_experimental
from ._impl import tgs_allmdc_impl, tgs_prepare_polars, tgs_validate_args


@warn_experimental
def alifestd_tgs_allmdc_estimate_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_masks: typing.Sequence[typing.Union[str, pl.Expr]],
    trait_names: typing.Optional[typing.Sequence[str]] = None,
    neutral_traits: typing.Sequence[typing.Union[int, str]] = (),
    clade_size_groups_per_doubling: int = 2,
    calibrate: bool = False,
    calibration_neighborhood_fractions: typing.Sequence[float] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    ),
    num_calibration_partitions: int = 4096,
    num_bootstrap: int = 0,
    loss_rate: float = 0.0,
    selection_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
    max_growth_exponent: float = 600.0,
    max_shift_iterations: int = 50,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = identity,
) -> pl.DataFrame:
    """Estimate per-trait gain rate and net selection from every branch,
    pendant and inner, i.e., from mutationally defined clades (MDCs) of
    every size, from how often each branch's clade carries the trait as a
    function of branch length and clade size.

    Branches subtending multi-leaf clades are fit with a survival shift per
    clade-size group, shared across traits, for the time a carrier clade
    needs to grow out to its sampled leaves. See
    `alifestd_tgs_allmdc_estimate_asexual` for the model.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format, with column
        `origin_time`.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    trait_masks : Sequence[str or polars.Expr]
        Column names or boolean expressions marking taxa with each trait.
        Only leaf values are used. Null values are not supported.
    trait_names : Sequence[str], optional
        Labels for output rows; defaults to column names for traits given as
        str, else trait indices.
    neutral_traits : Sequence[int or str], default ()
        Indices or names of traits known to be neutral. Gain ratios are
        reported relative to their mean gain rate (leaving the focal trait
        out). Optional unless `calibrate` is set, which requires at least
        two.
    clade_size_groups_per_doubling : int, default 2
        Number of log-scaled clade-size groups per doubling of clade size;
        see `alifestd_tgs_allmdc_estimate_asexual`.
    calibrate : bool, default False
        If True, calibrate branch lengths against `neutral_traits` gains
        within each clade-size group, on the single-leaf group's neutral
        clock.
    calibration_neighborhood_fractions : Sequence[float], default (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        Candidate calibration neighborhood sizes, as fractions of each
        group's branches.
    num_calibration_partitions : int, default 4096
        Maximum number of partitions of each group's branches for
        calibration and binning; a power of two.
    num_bootstrap : int, default 0
        Number of bootstrap replicates, resampling branches with
        replacement and refitting.
    loss_rate : float, default 0.0
        Known per-unit-time trait loss rate.
    selection_starts : Sequence[float], default (0.0, -0.3, -1.8)
        Starting values for selection, in units of 1 / mean branch length.
    max_growth_exponent : float, default 600.0
        Cap on carrier growth exponents.
    max_shift_iterations : int, default 50
        Maximum rounds of alternately fitting traits and clade-size shifts.
    seed : int, optional
        Seed for bootstrap resampling.
    parallel_backend : str, optional
        joblib backend to run bootstrap replicates in parallel, e.g., "loky"
        (processes) or "threading".
    progress_wrap : Callable, default identity
        Pass tqdm or equivalent to display fit iteration and bootstrap
        progress.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column, if ids are
        non-contiguous or not topologically sorted, or if a trait mask has
        null values.
    ValueError
        If `calibrate` is set with fewer than two distinct neutral traits,
        if `trait_names` and `trait_masks` differ in length, if
        `clade_size_groups_per_doubling` is less than 1, if
        `calibration_neighborhood_fractions` is empty or has a value outside
        (0, 1], or if `num_calibration_partitions` is not a power of two.

    Returns
    -------
    polars.DataFrame
        One row per trait; see
        `alifestd_tgs_allmdc_estimate_asexual` for columns.

    See Also
    --------
    alifestd_tgs_allmdc_estimate_asexual :
        Pandas-based implementation.
    alifestd_tgs_pendantmdc_estimate_polars :
        Leaf pendant branches only.
    """
    trait_names, neutral_traits = tgs_validate_args(
        trait_masks=trait_masks,
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibration_neighborhood_fractions=calibration_neighborhood_fractions,
        num_calibration_partitions=num_calibration_partitions,
    )
    prepared = tgs_prepare_polars(
        phylogeny_df,
        trait_masks=trait_masks,
        trait_names=trait_names,
        caller="alifestd_tgs_allmdc_estimate_polars",
    )

    logging.info(
        "- alifestd_tgs_allmdc_estimate_polars: "
        "fitting gain and selection...",
    )
    res = tgs_allmdc_impl(
        prepared,
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        clade_size_groups_per_doubling=clade_size_groups_per_doubling,
        calibrate=calibrate,
        calibration_neighborhood_fractions=calibration_neighborhood_fractions,
        num_calibration_partitions=num_calibration_partitions,
        num_bootstrap=num_bootstrap,
        loss_rate=loss_rate,
        selection_starts=selection_starts,
        max_growth_exponent=max_growth_exponent,
        max_shift_iterations=max_shift_iterations,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
    return pl.from_pandas(res)


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Estimate per-trait gain rate and net selection from how often each branch's
clade carries each trait as a function of branch length and clade size,
using pendant and inner branches, with a survival shift per clade-size group
shared across traits.

Output is one row per trait, with columns `trait`, `num_leaves`,
`num_trait_present`, `num_clades`, `num_clade_gains`, `gain_rate`,
`selection`, `gain_ratio`, `selection_lrt_naive`, and a `clade_size_shift_*`
column per clade-size group; with `--num-bootstrap`, also `selection_se`,
`selection_z`, `selection_p`, `log_gain_ratio_se`, `gain_ratio_z`,
`gain_ratio_p`, `num_bootstrap_degenerate`, and `clade_size_shift_*_se`.

Data is assumed to be in alife standard format, with column `origin_time`.
Only supports asexual phylogenies.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

- Output is a per-trait summary table, not a phylogeny.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_tgs_allmdc_estimate_asexual :
    Pandas-based implementation.
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
        dfcli_module=(
            "phyloframe.legacy._alifestd_tgs_allmdc_estimate_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--trait-masks",
        nargs="+",
        required=True,
        type=str,
        help="Names of boolean columns marking taxa with each trait.",
    )
    parser.add_argument(
        "--neutral-traits",
        nargs="*",
        default=[],
        type=str,
        help="Names of trait columns known to be neutral (default: none).",
    )
    parser.add_argument(
        "--clade-size-groups-per-doubling",
        default=2,
        type=int,
        help=(
            "Number of log-scaled clade-size groups per doubling of clade "
            "size (default: 2)."
        ),
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Measure branch length by neutral trait gains.",
    )
    parser.add_argument(
        "--calibration-neighborhood-fractions",
        default=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        nargs="+",
        type=float,
        help=(
            "Candidate calibration neighborhood sizes, as fractions of each "
            "group's branches (default: 0.005 0.01 0.025 0.05 0.1 0.25 0.5 "
            "1.0)."
        ),
    )
    parser.add_argument(
        "--num-calibration-partitions",
        default=4096,
        type=int,
        help=(
            "Maximum number of branch partitions per clade-size group, a "
            "power of two (default: 4096)."
        ),
    )
    parser.add_argument(
        "--num-bootstrap",
        default=0,
        type=int,
        help="Number of bootstrap replicates (default: 0).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        default=None,
        type=int,
        help="Seed for bootstrap resampling (default: None).",
    )
    parser.add_argument(
        "--loss-rate",
        default=0.0,
        type=float,
        help="Known per-unit-time trait loss rate (default: 0.0).",
    )
    parser.add_argument(
        "--selection-starts",
        default=(0.0, -0.3, -1.8),
        nargs="+",
        type=float,
        help=(
            "Starting values for selection, in units of 1 / mean branch "
            "length (default: 0.0 -0.3 -1.8)."
        ),
    )
    parser.add_argument(
        "--max-growth-exponent",
        default=600.0,
        type=float,
        help="Cap on carrier growth exponents (default: 600.0).",
    )
    parser.add_argument(
        "--max-shift-iterations",
        default=50,
        type=int,
        help=(
            "Maximum rounds of alternately fitting traits and clade-size "
            "shifts (default: 50)."
        ),
    )
    parser.add_argument(
        "--parallel-backend",
        default=None,
        type=str,
        help="joblib parallel backend for bootstrap (default: None).",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_tgs_allmdc_estimate_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_tgs_allmdc_estimate_polars,
                    trait_masks=args.trait_masks,
                    neutral_traits=args.neutral_traits,
                    clade_size_groups_per_doubling=(
                        args.clade_size_groups_per_doubling
                    ),
                    calibrate=args.calibrate,
                    calibration_neighborhood_fractions=(
                        args.calibration_neighborhood_fractions
                    ),
                    num_calibration_partitions=args.num_calibration_partitions,
                    num_bootstrap=args.num_bootstrap,
                    seed=args.bootstrap_seed,
                    loss_rate=args.loss_rate,
                    selection_starts=args.selection_starts,
                    max_growth_exponent=args.max_growth_exponent,
                    max_shift_iterations=args.max_shift_iterations,
                    parallel_backend=args.parallel_backend,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e

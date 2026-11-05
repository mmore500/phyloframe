import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._identity import identity
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._warn_experimental import warn_experimental
from ._impl import tgs_impl, tgs_prepare_polars, tgs_validate_args


@warn_experimental
def alifestd_tgs_pendantmdc_estimate_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_transitions: typing.Sequence[str],
    neutral_traits: typing.Sequence[str] = (),
    calibrate: bool = False,
    calibrate_neighborhood_fractions: typing.Sequence[float] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    ),
    calibrate_num_partitions: int = 4096,
    calibrate_max_rate_ratio: float = np.inf,
    num_bootstrap: int = 0,
    loss_rate: float = 0.0,
    selection_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
    growth_exponent_bound: float = 600.0,
    min_leaf_origin_time: float = -np.inf,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = identity,
) -> pl.DataFrame:
    """Estimate per-trait gain rate and net selection from leaf pendant
    branches only, i.e., from single-leaf mutationally defined clades
    (MDCs), from how the fraction of leaves carrying each trait varies with
    pendant branch length.

    Each leaf's pendant branch starts without the trait (unless the trait is
    shared across the parent's whole clade, in which case the leaf is
    dropped for that trait), so pendant length is an exposure window. Trait
    presence versus pendant length is fit to

        p = gain_rate * (exp(c * length) - 1) / c,  c = selection - loss_rate

    which is linear in length at neutrality and saturates under a cost.

    Uses pendant branches only, in that:

    - only gains on a leaf's own pendant branch are counted; leaves whose
      trait is shared by their parent's whole clade (i.e., gained on an
      inner branch) are dropped for that trait, so trait-defined clades
      contribute nothing;
    - inner branches contribute no exposure, and branching within carrier
      clades (the rate at which they diversify) is not modeled; and
    - each leaf is scored independently, given its pendant length, rather
      than through a full tree likelihood (e.g., BiSSE).

    Calibration also uses pendant branches only, comparing neutral traits'
    pendant gains across leaves of similar pendant length and origin time.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format, with column
        `origin_time`.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    trait_transitions : Sequence[str]
        Names of integer columns, one per trait, giving each node's trait
        state transition on the branch from its parent: +1 for a gain, -1
        for a loss, 0 for none (and, at a root, its initial state). Make
        them from trait states at every node with
        `alifestd_mark_lineage_delta_polars`, with `basal_value=False`
        to give each root's initial state. For leaf data, under parsimony
        without loss, a node has the trait if and only if every leaf of its
        clade does, so gains fall on the stems of mutationally defined
        clades (see `alifestd_mask_trait_mdcs_strict_polars`). Loss events,
        i.e., -1 values, are not yet supported. Output rows are labeled by
        these column names.
    neutral_traits : Sequence[str], default ()
        Names, among `trait_transitions`, of traits known to be neutral.
        Used to calibrate pendant lengths (with `calibrate`), which requires
        at least two, and as the reference for gain ratios (relative to
        their mean gain rate, leaving the focal trait out).
        Without neutral traits, `gain_ratio` and its statistics are nan;
        gain rate and selection are still estimated.
    calibrate : bool, default False
        If True, measure pendant length by `neutral_traits` gains rather
        than by branch length, so that distortion of branch lengths shared
        with neutral traits (e.g., from tree reconstruction) cancels. See
        `alifestd_tgs_pendantmdc_estimate_asexual` for details.
    calibrate_neighborhood_fractions : Sequence[float], default (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        Candidate calibration neighborhood sizes, as fractions of leaves.
        With `calibrate`, the candidate whose calibration best predicts
        held-out neutral traits is used. Larger neighborhoods smooth more;
        smaller ones resolve finer distortion but are noisier. A fraction of
        1.0 makes one neighborhood of all leaves, i.e., no calibration.
    calibrate_num_partitions : int, default 4096
        Number of equal-occupancy partitions of leaves, by ranked pendant
        length and origin time, among which calibration neighborhoods are
        formed; a power of two. Also bins leaves for fitting, so that fit
        cost is bounded independent of leaf count. Fewer are used if needed
        to keep at least 4 leaves in each.
    calibrate_max_rate_ratio : float, default np.inf
        Cap on the neutral gain rate ratios that rescale pendant lengths in
        calibration, bounding the inflation of lengths in neighborhoods
        with little exposure; uncapped by default. Must be positive.
    num_bootstrap : int, default 0
        Number of bootstrap replicates, resampling leaves with replacement
        and refitting, for standard errors, z-scores, and p-values. The
        calibration neighborhood size is chosen once, on the full data, and
        held fixed across replicates.
    loss_rate : float, default 0.0
        Known per-unit-time trait loss rate.
    selection_starts : Sequence[float], default (0.0, -0.3, -1.8)
        Starting values for selection, in units of 1 / mean pendant length;
        the best of the resulting fits is kept. See `tgs_fit`.
    growth_exponent_bound : float, default 600.0
        Bound on carrier growth exponents, (selection - loss rate) *
        pendant length, keeping optimizer trial steps finite. See `tgs_fit`.
    min_leaf_origin_time : float, default -inf
        Leaves originating earlier are excluded.
    seed : int, optional
        Seed for bootstrap resampling.
    parallel_backend : str, optional
        joblib backend to run bootstrap replicates in parallel, e.g., "loky"
        (processes) or "threading".
    progress_wrap : Callable, default identity
        Pass tqdm or equivalent to display bootstrap progress.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column, if ids are
        non-contiguous or not topologically sorted, or if a
        `trait_transitions` column is not integer, has null values, or has
        loss events, i.e., -1 values.
    ValueError
        If `calibrate` is set with fewer than two distinct neutral traits,
        if a `trait_transitions` column is missing or has values other than
        +1, -1, or 0, or gives states other than 0 and 1, if a
        `neutral_traits` name is not among `trait_transitions`, or if
        `calibrate_neighborhood_fractions` is empty or has a value outside
        (0, 1], if `calibrate_num_partitions` is not a power of two, or if
        `calibrate_max_rate_ratio` is not positive.

    Returns
    -------
    polars.DataFrame
        One row per trait; see
        `alifestd_tgs_pendantmdc_estimate_asexual` for columns.

    See Also
    --------
    alifestd_tgs_pendantmdc_estimate_asexual :
        Pandas-based implementation.
    """
    trait_names, neutral_traits = tgs_validate_args(
        trait_transitions=trait_transitions,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_num_partitions=calibrate_num_partitions,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
    )
    exposure_df = tgs_prepare_polars(
        phylogeny_df,
        trait_transitions=trait_transitions,
        caller="alifestd_tgs_pendantmdc_estimate_polars",
    )

    logging.info(
        "- alifestd_tgs_pendantmdc_estimate_polars: "
        "fitting gain and selection...",
    )
    res = tgs_impl(
        exposure_df,
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_num_partitions=calibrate_num_partitions,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        num_bootstrap=num_bootstrap,
        loss_rate=loss_rate,
        selection_starts=selection_starts,
        growth_exponent_bound=growth_exponent_bound,
        min_leaf_origin_time=min_leaf_origin_time,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
    return pl.from_pandas(res)


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Estimate per-trait gain rate and net selection from how often leaves carry
each trait as a function of pendant branch length.

Output is one row per trait, with columns `trait`, `num_leaves`,
`num_trait_present`, `gain_rate`, `selection`, `gain_ratio`,
`selection_lrt_naive`, and `gain_ratio_lrt_naive`; with `--num-bootstrap`,
also `selection_se`, `selection_z`, `selection_p`, `log_gain_ratio_se`,
`gain_ratio_z`, `gain_ratio_p`, and `num_bootstrap_degenerate`.

Data is assumed to be in alife standard format, with column `origin_time`.
Only supports asexual phylogenies.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

- Output is a per-trait summary table, not a phylogeny.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_tgs_pendantmdc_estimate_asexual :
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
            "phyloframe.legacy._alifestd_tgs_pendantmdc_estimate_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--trait-transitions",
        nargs="+",
        required=True,
        type=str,
        help=(
            "Names of integer columns giving each trait's state transition "
            "on the branch into each node (+1 gain, -1 loss, 0 none)."
        ),
    )
    parser.add_argument(
        "--neutral-traits",
        nargs="*",
        default=[],
        type=str,
        help=(
            "Names of trait transition columns of traits known to be neutral "
            "(default: none)."
        ),
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Measure pendant length by neutral trait gains.",
    )
    parser.add_argument(
        "--calibrate-neighborhood-fractions",
        default=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        nargs="+",
        type=float,
        help=(
            "Candidate calibration neighborhood sizes, as fractions of "
            "leaves (default: 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1.0)."
        ),
    )
    parser.add_argument(
        "--calibrate-num-partitions",
        default=4096,
        type=int,
        help=(
            "Number of leaf partitions for calibration neighborhoods, a "
            "power of two (default: 4096)."
        ),
    )
    parser.add_argument(
        "--calibrate-max-rate-ratio",
        default=np.inf,
        type=float,
        help=(
            "Cap on calibration's neutral gain rate ratios "
            "(default: uncapped)."
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
            "Starting values for selection, in units of 1 / mean pendant "
            "length (default: 0.0 -0.3 -1.8)."
        ),
    )
    parser.add_argument(
        "--growth-exponent-bound",
        default=600.0,
        type=float,
        help="Bound on carrier growth exponents (default: 600.0).",
    )
    parser.add_argument(
        "--min-leaf-origin-time",
        default=-np.inf,
        type=float,
        help="Exclude leaves originating earlier (default: -inf).",
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
            "phyloframe.legacy._alifestd_tgs_pendantmdc_estimate_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_tgs_pendantmdc_estimate_polars,
                    trait_transitions=args.trait_transitions,
                    neutral_traits=args.neutral_traits,
                    calibrate=args.calibrate,
                    calibrate_neighborhood_fractions=(
                        args.calibrate_neighborhood_fractions
                    ),
                    calibrate_num_partitions=args.calibrate_num_partitions,
                    calibrate_max_rate_ratio=args.calibrate_max_rate_ratio,
                    num_bootstrap=args.num_bootstrap,
                    seed=args.bootstrap_seed,
                    loss_rate=args.loss_rate,
                    selection_starts=args.selection_starts,
                    growth_exponent_bound=args.growth_exponent_bound,
                    min_leaf_origin_time=args.min_leaf_origin_time,
                    parallel_backend=args.parallel_backend,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e

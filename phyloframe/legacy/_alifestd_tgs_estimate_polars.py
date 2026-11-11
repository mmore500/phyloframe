import argparse
import functools
import gc
import logging
import os
import typing
import warnings

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import polars as pl
from tqdm import tqdm

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._identity import identity
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._log_memory_usage import log_memory_usage
from .._auxlib._warn_experimental import warn_experimental
from ._impl import (
    tgs_build_exposure_df_polars,
    tgs_fit_traits,
    tgs_select_calibration_neighborhoods,
    tgs_tally_exposure_episodes,
    tgs_validate_args,
)


@warn_experimental
def alifestd_tgs_estimate_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_transitions: typing.Union[str, typing.Sequence[str]],
    neutral_traits: typing.Union[str, typing.Sequence[str]],
    episode_mask: typing.Optional[str] = None,
    calibration_partition_by: typing.Optional[str] = None,
    calibration_compare_by_bins: typing.Optional[str] = None,
    calibration_compare_by_vals: typing.Union[str, typing.Sequence[str]] = (
        "branch_length",
        "origin_time",
    ),
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
    calibrate_max_rate_ratio: float = np.inf,
    length_rtol: float = 1e-2,
    loss_rate: float = 0.0,
    selection_coef_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
    growth_exponent_bound: float = 600.0,
    progress_wrap: typing.Callable = identity,
) -> pl.DataFrame:
    """Estimate per-trait gain rate and selection coefficient (tgs, for
    trait gain/selection) from how often branches gain each trait.

    Each branch whose parent lacks a trait is an exposure episode, which
    gains the trait with probability

        gain_rate * (exp(c * length) - 1) / c * exp(c * survival)

    where `length` is the exposure episode's duration and `survival` is the
    time from its end node to the latest leaf below it (zero for a leaf).
    Survival describes how long a carrier lineage must persist to be seen
    at every leaf of its clade. Growth rate `c` is the selection
    coefficient minus the loss rate, `selection_coef - loss_rate`.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format; see Input
        Schema.
    trait_transitions : str or Sequence[str]
        Names of integer columns, one per trait, with each node's trait
        state transition on the branch from its parent: +1 for a gain and 0
        for none.

        See `alifestd_mark_lineage_delta_polars` to make transitions from
        trait states.
    neutral_traits : str or Sequence[str]
        Names, among `trait_transitions`, of at least two traits known to
        be neutral, used to calibrate episode lengths and as the reference
        for gain ratios.
    episode_mask : str, optional
        Name of a boolean column marking, at each exposure episode's end
        node, whether to fit that episode, e.g., `is_leaf` for leaf pendant
        branches only.
    calibration_partition_by : str, optional
        Name of a column of calibration partition ids, e.g., from
        `alifestd_mark_clade_size_geom_classes_polars`. Bins are only
        compared within a partition.
    calibration_compare_by_bins : str, optional
        Name of a column of calibration bin ids, at each exposure episode's
        end node, e.g., from `alifestd_mark_joint_quantile_bins_polars`.

        Calibration rescales each episode's length by the neutral gain rate
        in its bin's neighborhood relative to overall, removing distortion
        that neutral traits share (e.g., from tree reconstruction). If not
        given, each partition is one bin.
    calibration_compare_by_vals : str or Sequence[str], default ("branch_length", "origin_time")
        Columns by which bins are compared: a bin's neighborhood is its
        nearest bins in its partition by mean rank of these columns.

        May be episode columns `branch_length` (the episode's length),
        `origin_time`, `clade_size`, or `survival_duration`, or other
        columns of `phylogeny_df`, at each episode's end node.
    calibrate_neighborhood_fractions : Sequence[float], default (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        Candidate calibration neighborhood sizes, as fractions of a
        partition's bins.

        The candidate that best predicts each neutral trait's gains, when
        calibrating against the other neutral traits, is used.
    calibrate_max_rate_ratio : float, default np.inf
        Cap on the neutral gain rate ratios that rescale episode lengths.
    length_rtol : float, default 1e-2
        Relative tolerance within which episodes' calibrated lengths
        and survival durations must agree to be fit together, as one
        granule; 0 for one granule per episode.
    loss_rate : float, default 0.0
        Known per-unit-time trait loss rate.
    selection_coef_starts : Sequence[float], default (0.0, -0.3, -1.8)
        Optimizer starting values for the selection coefficient, in units
        of 1 / mean episode length.
    growth_exponent_bound : float, default 600.0
        Bound on `c * (length + survival)`, so likelihoods stay finite.
    progress_wrap : Callable, default identity
        Pass tqdm or equivalent to display progress over traits.

    Returns
    -------
    polars.DataFrame
        One row per trait; see Output Schema.

    Raises
    ------
    ValueError
        If there are no exposure episodes, if an argument is invalid, or if
        a trait is gained more than once along a lineage.
    NotImplementedError
        For unsupported input, e.g., fewer than two neutral traits, a
        non-asexual phylogeny, unifurcations, null values, non-finite
        origin times, exposure episodes with zero or negative lengths,
        trait losses, or a trait present at a root.

    Warns
    -----
    UserWarning
        For traits whose estimates are undefined (nan), or if `origin_time`
        values lose precision as float64.

    Input Schema
    ------------
    'id' : integer
        Taxon identifier; contiguous and topologically sorted.
    'ancestor_id' : integer
        Parent's taxon identifier (or own, for a root).

        Non-root nodes with one child (unifurcations) are not supported.
    'origin_time' : float
        Time of each taxon's origin, from which episode lengths and survival
        durations are taken; must be finite.
    `trait_transitions` columns : integer
        See Parameters.
    `episode_mask` column : boolean, optional
        See Parameters.
    `calibration_partition_by` column : optional
        See Parameters.
    `calibration_compare_by_bins` column : optional
        See Parameters.
    `calibration_compare_by_vals` columns : optional
        See Parameters.

    Output Schema
    -------------
    'trait' : string
        Trait, named by its `trait_transitions` column.
    'num_leaf_episodes' : integer
        Leaf branches that started without the trait.
    'num_leaf_gains' : integer
        Of those, the number that gained the trait.
    'num_inner_episodes' : integer
        Inner branches that started without the trait.
    'num_inner_gains' : integer
        Of those, the number that gained the trait.
    'gain_rate' : float
        Estimated rate of trait gain, per unit episode length.
    'selection_coef' : float
        Estimated rate at which a carrier lineage's expected number of
        sampled descendants changes relative to a non-carrier's (negative
        for a cost); nan without gains.
    'gain_ratio' : float
        Gain rate relative to the mean gain rate of the other neutral
        traits.
    'selection_coef_lrt_naive' : float
        Likelihood-ratio statistic for selection coefficient = 0.
    'gain_ratio_lrt_naive' : float
        Likelihood-ratio statistic for gain ratio = 1.

    Warnings
    --------
    Statistical inference tends to overestimate significance. Nested
    exposure episodes share leaves, so their outcomes are not independent,
    but likelihood-ratio statistics treat them as independent.

    See Also
    --------
    alifestd_mark_clade_size_geom_classes_polars :
        Marks clade size classes, e.g., as calibration partitions.
    alifestd_mark_joint_quantile_bins_polars :
        Marks joint quantile bins, e.g., as calibration bins.
    alifestd_mark_max_descendant_origin_time_polars :
        Marks each node's latest descendant, e.g., for survival durations.
    alifestd_mask_trait_mdcs_strict_polars :
        Marks trait gains from leaf trait data.
    alifestd_tgs_estimate_bootstrap_polars :
        Adds bootstrap standard errors, z-scores, and p-values.

    Notes
    -----
    Calibration rescales episode lengths only. Survival durations are used
    as measured on the tree.

    With `loss_rate` 0, a neutral trait should be gained equally often per
    unit episode length, whatever the survival duration. If neutral traits
    are instead gained more or less often on episodes with longer survival
    durations (e.g., from distorted inner branch lengths), selection
    coefficients may be biased. To correct for this, additionally calibrate
    by survival duration; see Examples.

    Leaf pendant branches have zero survival duration, so disagreement
    between estimates with and without `episode_mask="is_leaf"` may suggest
    distorted inner branch lengths.

    Examples
    --------
    Take phylogeny `df` with trait transition columns `site0` to `site3`,
    the last two for traits known to be neutral:

    >>> trait_transitions = ["site0", "site1", "site2", "site3"]
    >>> neutral_traits = ["site2", "site3"]

    To calibrate within clade size classes, by bins of similar branch length
    and origin time:

    >>> estimates_df = (
    ...     alifestd_mark_origin_time_delta_polars(
    ...         df, mark_as="branch_length"
    ...     )
    ...     .pipe(
    ...         alifestd_mark_clade_size_geom_classes_polars,
    ...         weight_by=neutral_traits,
    ...         min_class_weight=20,
    ...     )
    ...     .pipe(
    ...         alifestd_mark_joint_quantile_bins_polars,
    ...         bin_by=["branch_length", "origin_time"],
    ...         partition_by="clade_size_geom_class",
    ...     )
    ...     .pipe(
    ...         alifestd_tgs_estimate_polars,
    ...         trait_transitions=trait_transitions,
    ...         neutral_traits=neutral_traits,
    ...         calibration_partition_by="clade_size_geom_class",
    ...         calibration_compare_by_bins="joint_quantile_bin",
    ...     )
    ... )

    To additionally calibrate by survival duration, mark it and add it to the
    columns that bins are made from and compared by:

    >>> estimates_df = (
    ...     alifestd_mark_origin_time_delta_polars(
    ...         df, mark_as="branch_length"
    ...     )
    ...     .pipe(alifestd_mark_max_descendant_origin_time_polars)
    ...     .with_columns(
    ...         survival_duration=pl.col("max_descendant_origin_time")
    ...         - pl.col("origin_time"),
    ...     )
    ...     .pipe(
    ...         alifestd_mark_clade_size_geom_classes_polars,
    ...         weight_by=neutral_traits,
    ...         min_class_weight=20,
    ...     )
    ...     .pipe(
    ...         alifestd_mark_joint_quantile_bins_polars,
    ...         bin_by=["branch_length", "origin_time", "survival_duration"],
    ...         partition_by="clade_size_geom_class",
    ...     )
    ...     .pipe(
    ...         alifestd_tgs_estimate_polars,
    ...         trait_transitions=trait_transitions,
    ...         neutral_traits=neutral_traits,
    ...         calibration_partition_by="clade_size_geom_class",
    ...         calibration_compare_by_bins="joint_quantile_bin",
    ...         calibration_compare_by_vals=[
    ...             "branch_length",
    ...             "origin_time",
    ...             "survival_duration",
    ...         ],
    ...     )
    ... )
    """
    return _alifestd_tgs_estimate_polars_impl(**locals())[0]


def _alifestd_tgs_estimate_polars_impl(
    phylogeny_df: pl.DataFrame,
    *,
    trait_transitions: typing.Union[str, typing.Sequence[str]],
    neutral_traits: typing.Union[str, typing.Sequence[str]],
    episode_mask: typing.Optional[str] = None,
    calibration_partition_by: typing.Optional[str] = None,
    calibration_compare_by_bins: typing.Optional[str] = None,
    calibration_compare_by_vals: typing.Union[str, typing.Sequence[str]] = (
        "branch_length",
        "origin_time",
    ),
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
    calibrate_max_rate_ratio: float = np.inf,
    length_rtol: float = 1e-2,
    loss_rate: float = 0.0,
    selection_coef_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
    growth_exponent_bound: float = 600.0,
    progress_wrap: typing.Callable = identity,
) -> typing.Tuple[pl.DataFrame, pl.DataFrame, typing.Callable]:
    """Implementation detail for `alifestd_tgs_estimate_polars` and
    `alifestd_tgs_estimate_bootstrap_polars`.

    Returns `alifestd_tgs_estimate_polars`'s estimates, the exposure
    episodes they were fit to (see `tgs_build_exposure_df_polars`), and a
    partial of `tgs_fit_traits`, taking `weights` and `progress_wrap`, with
    which bootstrap replicates refit.
    """
    log_memory_usage(logging.info)

    if isinstance(trait_transitions, str):
        trait_transitions = [trait_transitions]
    if isinstance(neutral_traits, str):
        neutral_traits = [neutral_traits]
    if isinstance(calibration_compare_by_vals, str):
        calibration_compare_by_vals = [calibration_compare_by_vals]

    tgs_validate_args(
        trait_transitions=trait_transitions,
        neutral_traits=neutral_traits,
        calibration_compare_by_vals=calibration_compare_by_vals,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        length_rtol=length_rtol,
        growth_exponent_bound=growth_exponent_bound,
    )
    neutral_trait_ids = sorted(
        {list(trait_transitions).index(trait) for trait in neutral_traits}
    )

    exposure_df = tgs_build_exposure_df_polars(
        phylogeny_df,
        trait_transitions=trait_transitions,
        episode_mask=episode_mask,
        calibration_partition_by=calibration_partition_by,
        calibration_compare_by_bins=calibration_compare_by_bins,
        calibration_compare_by_vals=calibration_compare_by_vals,
    )
    logging.info(
        f"- alifestd_tgs_estimate_polars: fitting {len(trait_transitions)} "
        f"traits to {len(exposure_df)} exposure episodes...",
    )

    fit_traits = functools.partial(
        tgs_fit_traits,
        exposure_df,
        num_traits=len(trait_transitions),
        neutral_trait_ids=neutral_trait_ids,
        calibration_neighborhoods=tgs_select_calibration_neighborhoods(
            exposure_df,
            neighborhood_fractions=calibrate_neighborhood_fractions,
            neutral_trait_ids=neutral_trait_ids,
            max_rate_ratio=calibrate_max_rate_ratio,
        ),
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        length_rtol=length_rtol,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
        growth_exponent_bound=growth_exponent_bound,
    )
    estimates_df = pl.concat(
        [
            tgs_tally_exposure_episodes(
                exposure_df, trait_names=trait_transitions
            ),
            fit_traits(
                weights=np.ones(len(exposure_df)), progress_wrap=progress_wrap
            ),
        ],
        how="horizontal",
    )
    gc.collect()
    log_memory_usage(logging.info)

    undefined_estimate_traits = estimates_df.filter(
        pl.col("gain_ratio_lrt_naive").is_nan()
    )["trait"].to_list()
    if undefined_estimate_traits:
        warnings.warn(
            f"traits {undefined_estimate_traits} have undefined (nan) "
            "estimates, as they or their neutral reference have no gains",
        )
    return estimates_df, exposure_df, fit_traits


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Estimate per-trait gain rate and selection coefficient from how often branches gain each trait.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.


Input Schema: Required Columns
==============================
'id' : integer
    Taxon identifier; contiguous and topologically sorted.

'ancestor_id' : integer
    Parent's taxon identifier (or own, for a root).

'origin_time' : float
    Time of each taxon's origin.

`--trait-transitions` columns : integer
    Each node's trait state transition on the branch from its parent: +1
    for a gain and 0 for none.


Output Schema
=============
One row per trait, not a phylogeny.

'trait' : string
    Trait, named by its transition column.

'num_leaf_episodes', 'num_leaf_gains' : integer
    Leaf branches that started without the trait, and of those, the number
    that gained it.

'num_inner_episodes', 'num_inner_gains' : integer
    Likewise, for inner branches.

'gain_rate', 'selection_coef', 'gain_ratio' : float
    Estimated gain rate, selection coefficient, and gain rate relative to
    the other neutral traits'.

'selection_coef_lrt_naive', 'gain_ratio_lrt_naive' : float
    Naive likelihood-ratio statistics for selection coefficient = 0 and for
    gain ratio = 1.


Additional Notes
================
- Requires contiguous, topologically sorted ids, and no unifurcations
  (non-root nodes with one child).

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_tgs_estimate_bootstrap_polars :
    Adds bootstrap standard errors, z-scores, and p-values.
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
        dfcli_module=("phyloframe.legacy._alifestd_tgs_estimate_polars"),
        dfcli_version=get_phyloframe_version(),
    )
    return _add_estimate_arguments(parser)


def _add_estimate_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Implementation detail for `_create_parser` here and in
    `_alifestd_tgs_estimate_bootstrap_polars`."""
    parser.add_argument(
        "--trait-transitions",
        nargs="+",
        required=True,
        type=str,
        help=(
            "Names of integer columns giving each trait's state transition "
            "on the branch into each node (+1 gain, 0 none)."
        ),
    )
    parser.add_argument(
        "--neutral-traits",
        nargs="+",
        required=True,
        type=str,
        help=(
            "Names of trait transition columns of traits known to be neutral; "
            "at least two."
        ),
    )
    parser.add_argument(
        "--episode-mask",
        default=None,
        type=str,
        help=(
            "Name of boolean column marking, at each exposure episode's end "
            "node, whether to fit it, e.g., is_leaf for leaf pendant "
            "branches only (default: none, i.e., fit every episode)."
        ),
    )
    parser.add_argument(
        "--calibration-partition-by",
        default=None,
        type=str,
        help=(
            "Name of integer column of calibration partitions; bins are "
            "compared only within a partition (default: none, i.e., one "
            "partition)."
        ),
    )
    parser.add_argument(
        "--calibration-compare-by-bins",
        default=None,
        type=str,
        help=(
            "Name of integer column of calibration bins (default: none, "
            "i.e., one bin per partition)."
        ),
    )
    parser.add_argument(
        "--calibration-compare-by-vals",
        default=("branch_length", "origin_time"),
        nargs="+",
        type=str,
        help=(
            "Columns by which calibration bins are compared (default: "
            "branch_length origin_time)."
        ),
    )
    parser.add_argument(
        "--length-rtol",
        default=1e-2,
        type=float,
        help=(
            "Relative tolerance within which episodes of similar calibrated "
            "length are fit together, as one granule (default: 1e-2)."
        ),
    )
    parser.add_argument(
        "--calibrate-neighborhood-fractions",
        default=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        nargs="+",
        type=float,
        help=(
            "Candidate calibration neighborhood sizes, as fractions of a "
            "partition's bins (default: 0.005 0.01 0.025 0.05 0.1 0.25 0.5 "
            "1.0)."
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
        "--loss-rate",
        default=0.0,
        type=float,
        help="Known per-unit-time trait loss rate (default: 0.0).",
    )
    parser.add_argument(
        "--selection-coef-starts",
        default=(0.0, -0.3, -1.8),
        nargs="+",
        type=float,
        help=(
            "Starting values for the selection coefficient, in units of "
            "1 / mean episode length (default: 0.0 -0.3 -1.8)."
        ),
    )
    parser.add_argument(
        "--growth-exponent-bound",
        default=600.0,
        type=float,
        help="Bound on carrier growth exponents (default: 600.0).",
    )
    return parser


def _get_estimate_kwargs(
    args: argparse.Namespace,
) -> typing.Dict[str, typing.Any]:
    """Implementation detail for the CLI here and in
    `_alifestd_tgs_estimate_bootstrap_polars`.

    Returns estimator keyword arguments from parsed `args` (see
    `_add_estimate_arguments`).
    """
    return dict(
        trait_transitions=args.trait_transitions,
        neutral_traits=args.neutral_traits,
        episode_mask=args.episode_mask,
        calibration_partition_by=args.calibration_partition_by,
        calibration_compare_by_bins=args.calibration_compare_by_bins,
        calibration_compare_by_vals=args.calibration_compare_by_vals,
        calibrate_neighborhood_fractions=args.calibrate_neighborhood_fractions,
        calibrate_max_rate_ratio=args.calibrate_max_rate_ratio,
        length_rtol=args.length_rtol,
        loss_rate=args.loss_rate,
        selection_coef_starts=args.selection_coef_starts,
        growth_exponent_bound=args.growth_exponent_bound,
    )


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    with log_context_duration(
        "phyloframe.legacy._alifestd_tgs_estimate_polars",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=functools.partial(
                alifestd_tgs_estimate_polars,
                **_get_estimate_kwargs(args),
                progress_wrap=functools.partial(
                    tqdm, desc="alifestd_tgs_estimate_polars"
                ),
            ),
        )

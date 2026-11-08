import argparse
import functools
import logging
import os
import typing

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
from .._auxlib._warn_experimental import warn_experimental
from ._impl import (
    tgs_build_exposure_df_polars,
    tgs_estimate_tgs_from_exposures,
    tgs_tally_exposure_episodes,
    tgs_validate_args,
)


@warn_experimental
def alifestd_tgs_pendantmdc_estimate_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_transitions: typing.Sequence[str],
    neutral_traits: typing.Sequence[str],
    calibration_partition_by: typing.Optional[str] = None,
    calibration_compare_by_bins: typing.Optional[str] = None,
    calibration_compare_by_vals: typing.Sequence[str] = (
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
    num_bootstrap: int = 0,
    loss_rate: float = 0.0,
    selection_coef_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
    growth_exponent_bound: float = 600.0,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = identity,
) -> pl.DataFrame:
    """Estimate per-trait gain rate and selection coefficient from leaf pendant
    branches only, from how the fraction of leaves carrying each trait
    varies with pendant branch length.

    Here, an MDC is a mutationally defined clade: a maximal clade whose
    leaves all carry the trait, which, under parsimony without loss, gained
    it on its stem branch. This estimator uses single-leaf MDCs only (gains
    on pendant branches); see
    `alifestd_tgs_allmdc_estimate_polars` for MDCs of
    every size.

    Each leaf's pendant branch starts without the trait (unless the trait is
    shared across the parent's whole clade, in which case the leaf is
    dropped for that trait), so pendant length is an exposure window. Trait
    presence versus pendant length is fit to

        p = gain_rate * (exp(c * length) - 1) / c,
        c = selection_coef - loss_rate

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
    pendant gains across calibration bins of leaves (see
    `calibration_compare_by_bins`).

    See `tgs_fit` for the model and its relation to BiSSE.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format; see Input
        Schema.
    trait_transitions : Sequence[str]
        Names of integer columns, one per trait, giving each node's trait
        state transition on the branch from its parent (see Input Schema).
        Output rows are labeled by these column names.
    neutral_traits : Sequence[str]
        Names, among `trait_transitions`, of traits known to be neutral. At
        least two are required for selecting the calibration configuration.
        Used to calibrate pendant lengths against one another; their mean gain
        rate, leaving the focal trait out, is the reference for gain ratios.
    calibration_partition_by : str, optional
        Name of a column of integer partition ids (see Input Schema): bins
        are compared only with bins of the same partition, never across
        partitions, with neighborhood size selected separately within each, e.g., by clade size (see
        `alifestd_mark_clade_size_classes_polars`), as neutral gains per
        unit length may differ between leaves and inner nodes. If not given,
        all bins are compared with one another.
    calibration_compare_by_bins : str, optional
        Name of a column of integer calibration bin ids, at each exposure
        episode's end node (see Input Schema), e.g., from
        `alifestd_mark_joint_quantile_bins_polars` (see Notes). Calibration
        rescales each episode's length by the neutral gain rate in its bin's
        neighborhood relative to overall, removing distortion that neutral
        traits share (e.g., from tree reconstruction). If not given, each
        partition is one bin, so without `calibration_partition_by` too,
        there is no calibration.
    calibration_compare_by_vals : Sequence[str], default ("branch_length", "origin_time")
        Columns by which bins are compared: a bin's neighborhood is its
        nearest bins in its partition by mean rank of these columns.
        Exposure episode columns take precedence: `branch_length`;
        `origin_time`, of the exposure episode's end node; `clade_size`, its
        number of leaves; and `survival_duration`, from its end node to its
        latest leaf. Other names are numeric columns of `phylogeny_df`, at
        each episode's end node.
    calibrate_neighborhood_fractions : Sequence[float], default (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        Candidate calibration neighborhood sizes, as fractions of a
        partition's bins. The candidate used is the one whose calibration
        best predicts where each neutral trait was gained, when calibrating
        against only the other neutral traits. Larger neighborhoods smooth
        more; smaller ones resolve finer distortion but are noisier. Pass
        (1.0,), one neighborhood of each partition's bins, to disable
        calibration within partitions.
    calibrate_max_rate_ratio : float, default np.inf
        Cap on the neutral gain rate ratios that rescale pendant lengths in
        calibration, bounding the inflation of lengths in neighborhoods
        with little exposure; uncapped by default. Must be positive.
    length_rtol : float, default 1e-2
        Relative tolerance within which leaves' calibrated pendant lengths
        must agree for them to be merged for fitting, bounding fit cost; 0
        for no merging.
    num_bootstrap : int, default 0
        Number of bootstrap replicates, resampling leaves with replacement
        and refitting, for standard errors, z-scores, and p-values.
        Standard guidance suggests about 200 replicates for standard errors
        (Efron and Tibshirani 1993, An Introduction to the Bootstrap).
        Calibration neighborhood size is held constant.
    loss_rate : float, default 0.0
        Known per-unit-time trait loss rate.
    selection_coef_starts : Sequence[float], default (0.0, -0.3, -1.8)
        Starting values for the selection coefficient, in units of 1 / mean
        pendant length; the best of the resulting fits is kept. See `tgs_fit`.
    growth_exponent_bound : float, default 600.0
        Bound on carrier growth exponents, the selection coefficient minus
        loss rate, times pendant length, keeping optimizer trial steps
        finite. See `tgs_fit`.
    seed : int, optional
        Seed for bootstrap resampling.
    parallel_backend : str, optional
        joblib backend to run bootstrap replicates in parallel, e.g., "loky"
        (processes) or "threading". Replicate fits are mostly GIL-bound
        Python (scipy's optimizer loop), so "loky" is typically faster.
    progress_wrap : Callable, default identity
        Pass tqdm or equivalent to display progress over traits, for the
        point estimate, and over bootstrap replicates.

    Returns
    -------
    polars.DataFrame
        One row per trait; see Output Schema.

    Raises
    ------
    ValueError
        If there are no exposure episodes (i.e., leaf pendant branches to fit),
        if an argument is invalid, if a column has invalid values, or if a
        trait has gains but no calibrated exposure (calibration too fine).
    polars.exceptions.ColumnNotFoundError
        If a column is missing.
    NotImplementedError
        If fewer than two neutral traits are given, or for unsupported input
        (see Input Schema), e.g., a non-asexual phylogeny, null values,
        trait losses, or a trait present at a root.

    Input Schema
    ------------
    'id' : integer
        Taxon identifier; contiguous and topologically sorted.
    'ancestor_id' : integer
        Parent's taxon identifier (or own, for a root).
    'origin_time' : float
        Time of each taxon's origin, from which branch lengths are taken.
    `trait_transitions` columns : integer
        Each node's trait state transition on the branch from its parent:
        +1 for a gain, -1 for a loss (not yet supported), 0 for none, and,
        at a root, its initial state, which must be 0 (a trait present at a
        root is uninformative, so not supported); see Notes.
    `calibration_partition_by`, `calibration_compare_by_bins` columns : integer, optional
        Each node's calibration partition and bin ids, for the exposure
        episode ending at it; see Parameters.
    `calibration_compare_by_vals` columns : numeric, optional
        Values by which calibration bins are compared; see Parameters.

    Output Schema
    -------------
    One row per trait:

    'trait' : string
        Trait, named by its `trait_transitions` column.
    'num_leaf_episodes' : integer
        Leaf pendant branches that started without the trait.
    'num_leaf_gains' : integer
        Of those, the number that gained the trait.
    'gain_rate' : float
        Estimated rate of trait gain, per unit pendant length.
    'selection_coef' : float
        Estimated selection coefficient, the rate at which a carrier
        lineage's expected number of sampled descendants changes relative
        to a non-carrier's (negative for a cost); nan without carriers.
    'gain_ratio' : float
        Gain rate relative to the mean gain rate of the other neutral
        traits.
    'selection_coef_lrt_naive' : float
        Likelihood-ratio statistic for selection coefficient = 0 (see
        Warnings).
    'gain_ratio_lrt_naive' : float
        Likelihood-ratio statistic for gain ratio = 1 (see Warnings).

    With bootstrap replicates, also:

    'selection_coef_se', 'log_gain_ratio_se' : float
        Bootstrap standard errors (se), i.e., standard deviations across
        replicates, of the selection coefficient and log gain ratio.
    'selection_coef_z', 'gain_ratio_z' : float
        Corresponding z-scores, against 0 and 1, respectively.
    'selection_coef_p', 'gain_ratio_p' : float
        Corresponding two-sided p-values, by normal approximation.
    'num_bootstrap_degenerate' : integer
        Replicates excluded for resampling no carriers of the trait, or of
        its neutral reference.

    Notes
    -----
    Trait transitions can be made from trait states at every node with
    `alifestd_mark_lineage_delta_polars`, with `basal_value=False`, so that
    a trait present at a root is caught rather than silently read as absent;
    for leaf data, see `alifestd_mask_trait_mdcs_strict_polars`.

    Unifurcations are collapsed (see `alifestd_collapse_unifurcations_polars`),
    so pendant branches span from leaves to their stem node, the nearest
    ancestor with at least two children. Root leaves have no pendant branch,
    so are excluded.

    To calibrate by bins of leaves of similar pendant length and origin
    time, mark them, measuring pendant lengths from each leaf's stem node
    (see `alifestd_mark_stem_node_id_polars`):

        estimates_df = (
            alifestd_mark_stem_node_id_polars(df)
            .pipe(alifestd_mark_leaves_polars)
            .with_columns(
                branch_length=pl.col("origin_time")
                - pl.col("origin_time").gather(pl.col("stem_node_id")),
            )
            .pipe(
                alifestd_mark_joint_quantile_bins_polars,
                bin_by=["branch_length", "origin_time"],
                partition_by="is_leaf",
            )
            .pipe(
                alifestd_tgs_pendantmdc_estimate_polars,
                trait_transitions=trait_transitions,
                neutral_traits=neutral_traits,
                calibration_compare_by_bins="joint_quantile_bin",
            )
        )

    The joint quantile bins reject null or nan values, so fill or drop rows
    with an unknown `origin_time` (e.g., a disconnected root) first.

    To exclude some leaves, e.g., those originating before a given time,
    prune them first: mark the leaves to keep in a boolean column, and pass
    it as `criterion` to `alifestd_prune_extinct_lineages_polars`. Pruning a
    leaf lengthens its sister's pendant branch if the sister is then an only
    child.

    Under sparse sampling, the selection coefficient estimates the change
    rate of a reconstructed carrier lineage's weight, roughly the
    difference in per-lineage net growth; magnitudes may be biased, but
    signs and rankings are informative. With few carriers, cost and loss
    are confounded, so `loss_rate` is fixed rather than fit.

    `gain_ratio_lrt_naive` refits the selection coefficient with the trait's
    gain rate held at its neutral reference's mean.

    Warnings
    --------
    Statistical inference tends to overestimate significance. Leaves are
    correlated through shared ancestry, but likelihood testing treats them
    as independent, and the gain ratio's test treats the neutral reference
    rate as known, so `selection_coef_lrt_naive` and `gain_ratio_lrt_naive`
    overstate significance; consider dividing each by its mean over neutral
    traits before a chi-squared test with one degree of freedom. Likewise,
    bootstrap sampling assumes leaves are independent, so its replicates may
    be underdispersed, making standard errors too small.

    See Also
    --------
    alifestd_mark_joint_quantile_bins_polars :
        Marks joint quantile bins, e.g., as calibration bins.
    alifestd_mask_trait_singletons_polars :
        Marks leaves dropped for each trait.
    alifestd_tgs_allmdc_estimate_polars :
        MDCs of every size, not only leaf pendant branches.
    """
    trait_names, neutral_traits = tgs_validate_args(
        trait_transitions=trait_transitions,
        neutral_traits=neutral_traits,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        length_rtol=length_rtol,
        calibration_compare_by_vals=calibration_compare_by_vals,
    )
    exposure_df = tgs_build_exposure_df_polars(
        phylogeny_df,
        trait_transitions=trait_transitions,
        caller="alifestd_tgs_pendantmdc_estimate_polars",
        calibration_partition_by=calibration_partition_by,
        calibration_compare_by_bins=calibration_compare_by_bins,
        calibration_compare_by_vals=calibration_compare_by_vals,
    )
    # leaf pendant branches only
    exposure_df = exposure_df.filter(pl.col("clade_size") == 1)
    return tgs_estimate_tgs_from_exposures(
        exposure_df,
        counts_df=tgs_tally_exposure_episodes(
            trait_names=trait_names, exposure_df=exposure_df
        ).select("trait", "num_leaf_episodes", "num_leaf_gains"),
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        length_rtol=length_rtol,
        num_bootstrap=num_bootstrap,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
        growth_exponent_bound=growth_exponent_bound,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Estimate per-trait gain rate and selection coefficient from how often leaves
carry each trait as a function of pendant branch length.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.


Input Schema: Required Columns
==============================
'id' : integer
    Taxon identifier; contiguous and topologically sorted.

'ancestor_id' : integer
    Parent's taxon identifier (or own, for a root).

'origin_time' : float
    Time of each taxon's origin, from which branch lengths are taken.

`--trait-transitions` columns : integer
    Each node's trait state transition on the branch from its parent: +1
    for a gain, 0 for none, and, at a root, its initial state.


Input Schema: Optional Columns
==============================
`--calibration-partition-by`, `--calibration-compare-by-bins` columns : integer
    Each node's calibration partition and bin (see
    phyloframe.legacy._alifestd_mark_clade_size_classes_polars and
    phyloframe.legacy._alifestd_mark_joint_quantile_bins_polars).

`--calibration-compare-by-vals` columns : numeric
    Values by which calibration bins are compared, if not exposure episode
    columns (e.g., branch_length).


Output Schema
=============
One row per trait, not a phylogeny.

'trait' : string
    Trait, named by its transition column.

'num_leaf_episodes', 'num_leaf_gains' : integer
    Leaf pendant branches that started without the trait, and of those, the
    number that gained it.

'gain_rate', 'selection_coef', 'gain_ratio' : float
    Estimated gain rate, selection coefficient, and gain rate relative to
    the other neutral traits'.

'selection_coef_lrt_naive', 'gain_ratio_lrt_naive' : float
    Naive likelihood-ratio statistics for selection coefficient = 0 and for
    gain ratio = 1.

With `--num-bootstrap`, also 'selection_coef_se', 'selection_coef_z',
'selection_coef_p', 'log_gain_ratio_se', 'gain_ratio_z', 'gain_ratio_p', and
'num_bootstrap_degenerate'; see the Python docstring.


Additional Notes
================
- Requires contiguous, topologically sorted ids.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_tgs_allmdc_estimate_polars :
    MDCs of every size, not only leaf pendant branches.
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
        nargs="+",
        required=True,
        type=str,
        help=(
            "Names of trait transition columns of traits known to be neutral; "
            "at least two."
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
            "Relative tolerance for merging calibrated branch lengths for "
            "fitting (default: 1e-2)."
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
        "--selection-coef-starts",
        default=(0.0, -0.3, -1.8),
        nargs="+",
        type=float,
        help=(
            "Starting values for the selection coefficient, in units of "
            "1 / mean pendant length (default: 0.0 -0.3 -1.8)."
        ),
    )
    parser.add_argument(
        "--growth-exponent-bound",
        default=600.0,
        type=float,
        help="Bound on carrier growth exponents (default: 600.0).",
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
                    calibrate_neighborhood_fractions=(
                        args.calibrate_neighborhood_fractions
                    ),
                    calibration_partition_by=args.calibration_partition_by,
                    calibration_compare_by_bins=(
                        args.calibration_compare_by_bins
                    ),
                    calibration_compare_by_vals=(
                        args.calibration_compare_by_vals
                    ),
                    length_rtol=args.length_rtol,
                    calibrate_max_rate_ratio=args.calibrate_max_rate_ratio,
                    num_bootstrap=args.num_bootstrap,
                    seed=args.bootstrap_seed,
                    loss_rate=args.loss_rate,
                    selection_coef_starts=args.selection_coef_starts,
                    growth_exponent_bound=args.growth_exponent_bound,
                    parallel_backend=args.parallel_backend,
                    progress_wrap=functools.partial(
                        tqdm, desc="alifestd_tgs_pendantmdc_estimate_polars"
                    ),
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e

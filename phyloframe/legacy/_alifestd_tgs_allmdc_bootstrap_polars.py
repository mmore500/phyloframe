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
from ._alifestd_tgs_allmdc_estimate_polars import (
    _add_parser_arguments,
    _alifestd_tgs_allmdc_build_exposure_df,
    _cli_kwargs,
)
from ._impl._tgs_bootstrap_tgs_from_exposures import (
    tgs_bootstrap_tgs_from_exposures,
)


@warn_experimental
def alifestd_tgs_allmdc_bootstrap_polars(
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
    loss_rate: float = 0.0,
    selection_coef_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
    growth_exponent_bound: float = 600.0,
    num_bootstrap: int = 200,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = identity,
) -> pl.DataFrame:
    """Fit each trait's gain rate and selection coefficient to bootstrap
    resamples of exposure episodes, as for `alifestd_tgs_allmdc_estimate_polars`.

    Each replicate resamples exposure episodes with replacement and refits. Calibration
    is set up once, from the original exposure episodes, as for the point estimate, so
    calibration neighborhood size is held constant across replicates. Pass
    the result, with the point estimates from
    `alifestd_tgs_allmdc_estimate_polars` run with the same arguments, to
    `alifestd_tgs_summarize_bootstrap_polars` for standard errors, z-scores,
    and p-values.

    Parameters
    ----------
    phylogeny_df, trait_transitions, neutral_traits, calibration_partition_by, calibration_compare_by_bins, calibration_compare_by_vals, calibrate_neighborhood_fractions, calibrate_max_rate_ratio, length_rtol, loss_rate, selection_coef_starts, growth_exponent_bound
        As for `alifestd_tgs_allmdc_estimate_polars`.
    num_bootstrap : int, default 200
        Number of bootstrap replicates. Standard guidance suggests about 200
        replicates for standard errors (Efron and Tibshirani 1993, An
        Introduction to the Bootstrap).
    seed : int, optional
        Seed for bootstrap resampling; if None, drawn from numpy's global
        random state.
    parallel_backend : str, optional
        joblib backend to run replicates in parallel, e.g., "loky"
        (processes) or "threading"; if None, runs them sequentially.
        Replicate fits are mostly GIL-bound Python (scipy's optimizer loop),
        so "loky" is typically faster.
    progress_wrap : Callable, default identity
        Pass tqdm or equivalent to display progress over replicates.

    Returns
    -------
    polars.DataFrame
        One row per replicate and trait; see Output Schema.

    Raises
    ------
    As for `alifestd_tgs_allmdc_estimate_polars`.

    Output Schema
    -------------
    'replicate' : integer
        Bootstrap replicate index.
    'trait' : string
        Trait, named by its `trait_transitions` column.
    'gain_rate' : float
        The replicate's fitted gain rate.
    'selection_coef' : float
        The replicate's fitted selection coefficient; nan without carriers.
    'selection_coef_lrt_naive' : float
        The replicate's likelihood-ratio statistic for selection
        coefficient = 0.
    'gain_ratio' : float
        The replicate's gain rate relative to the mean gain rate of the
        other neutral traits.
    'gain_ratio_lrt_naive' : float
        The replicate's likelihood-ratio statistic for gain ratio = 1.

    Warnings
    --------
    Replicates resample exposure episodes independently, ignoring correlation through
    shared ancestry, so their spread may understate uncertainty.

    See Also
    --------
    alifestd_tgs_allmdc_estimate_polars :
        Point estimates.
    alifestd_tgs_summarize_bootstrap_polars :
        Standard errors, z-scores, and p-values from these replicates.
    """
    (
        trait_names,
        neutral_traits,
        exposure_df,
    ) = _alifestd_tgs_allmdc_build_exposure_df(
        phylogeny_df,
        trait_transitions=trait_transitions,
        neutral_traits=neutral_traits,
        calibration_partition_by=calibration_partition_by,
        calibration_compare_by_bins=calibration_compare_by_bins,
        calibration_compare_by_vals=calibration_compare_by_vals,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        length_rtol=length_rtol,
    )
    return tgs_bootstrap_tgs_from_exposures(
        exposure_df,
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        length_rtol=length_rtol,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
        growth_exponent_bound=growth_exponent_bound,
        num_bootstrap=num_bootstrap,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Fit each trait's gain rate and selection coefficient to bootstrap resamples
of exposure episodes, as for phyloframe.legacy._alifestd_tgs_allmdc_estimate_polars,
which takes the same input columns and options.

Output Schema
=============
One row per replicate and trait, not a phylogeny.

'replicate' : integer
    Bootstrap replicate index.

'trait' : string
    Trait, named by its transition column.

'gain_rate', 'selection_coef', 'gain_ratio' : float
    The replicate's fitted gain rate, selection coefficient, and gain ratio.

'selection_coef_lrt_naive', 'gain_ratio_lrt_naive' : float
    The replicate's naive likelihood-ratio statistics.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_tgs_allmdc_estimate_polars :
    Point estimates.
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
            "phyloframe.legacy._alifestd_tgs_allmdc_bootstrap_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    parser = _add_parser_arguments(parser)
    parser.add_argument(
        "--num-bootstrap",
        default=200,
        type=int,
        help="Number of bootstrap replicates (default: 200).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        default=None,
        type=int,
        help="Seed for bootstrap resampling (default: None).",
    )
    parser.add_argument(
        "--parallel-backend",
        default=None,
        type=str,
        help="joblib parallel backend for replicates (default: None).",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_tgs_allmdc_bootstrap_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_tgs_allmdc_bootstrap_polars,
                    **_cli_kwargs(args),
                    num_bootstrap=args.num_bootstrap,
                    seed=args.bootstrap_seed,
                    parallel_backend=args.parallel_backend,
                    progress_wrap=functools.partial(
                        tqdm, desc="alifestd_tgs_allmdc_bootstrap_polars"
                    ),
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e

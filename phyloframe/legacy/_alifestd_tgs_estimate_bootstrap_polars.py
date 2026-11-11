import argparse
import functools
import gc
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl
from tqdm import tqdm

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._identity import identity
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._log_memory_usage import log_memory_usage
from .._auxlib._run_weighted_bootstrap import run_weighted_bootstrap
from .._auxlib._warn_experimental import warn_experimental
from ._alifestd_tgs_estimate_polars import (
    _add_estimate_arguments,
    _alifestd_tgs_estimate_polars_impl,
    _get_estimate_kwargs,
)
from ._impl import tgs_summarize_bootstrap


@warn_experimental
def alifestd_tgs_estimate_bootstrap_polars(
    phylogeny_df: pl.DataFrame,
    *,
    num_bootstrap: int,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = identity,
    **kwargs: typing.Any,
) -> pl.DataFrame:
    """Estimate per-trait gain rate and selection coefficient, as
    `alifestd_tgs_estimate_polars` does, with bootstrap standard errors,
    z-scores, and p-values.

    Bootstrap replicates resample exposure episodes with replacement and
    refit, holding calibration neighborhoods fixed at those of the point
    estimate.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format; see Input
        Schema of `alifestd_tgs_estimate_polars`.
    num_bootstrap : int
        Number of bootstrap replicates; must be positive.
    seed : int, optional
        Seed for bootstrap resampling.
    parallel_backend : str, optional
        joblib backend to run bootstrap replicates in parallel, e.g.,
        "loky"; if None, runs them sequentially.
    progress_wrap : Callable, default identity
        Pass tqdm or equivalent to display progress over traits and
        bootstrap replicates.
    **kwargs
        Passed to `alifestd_tgs_estimate_polars`, e.g.,
        `trait_transitions_candidate` and `trait_transitions_neutral`.

    Returns
    -------
    polars.DataFrame
        One row per trait; see Output Schema.

    Raises
    ------
    ValueError
        If `num_bootstrap` is not positive, or as for
        `alifestd_tgs_estimate_polars`.
    NotImplementedError
        As for `alifestd_tgs_estimate_polars`.

    Warns
    -----
    UserWarning
        For traits with bootstrap replicates excluded for an undefined
        estimate, or as for `alifestd_tgs_estimate_polars`.

    Output Schema
    -------------
    The columns of `alifestd_tgs_estimate_polars`'s Output Schema, then:

    'selection_coef_se', 'log_gain_ratio_se' : float
        Standard deviations of the selection coefficient and log gain ratio
        across bootstrap replicates.
    'selection_coef_z', 'gain_ratio_z' : float
        Corresponding z-scores, against 0 and 1, respectively.
    'selection_coef_p', 'gain_ratio_p' : float
        Corresponding two-sided p-values, by normal approximation.
    'num_bootstrap_degenerate' : integer
        Replicates excluded for resampling no gains of the trait, or of its
        neutral reference.

    Warnings
    --------
    Bootstrap resampling treats exposure episodes as independent, but
    nested episodes share leaves, so standard errors tend to be too small.

    See Also
    --------
    alifestd_tgs_estimate_polars :
        Point estimates only.
    """
    if not num_bootstrap > 0:
        raise ValueError("num_bootstrap must be positive")

    estimates_df, exposure_df, fit_traits = _alifestd_tgs_estimate_polars_impl(
        phylogeny_df, progress_wrap=progress_wrap, **kwargs
    )

    logging.info(
        f"- alifestd_tgs_estimate_bootstrap_polars: fitting {num_bootstrap} "
        "bootstrap replicates...",
    )
    replicate_fits = run_weighted_bootstrap(
        # standard errors take replicates' estimates only
        functools.partial(fit_traits, calc_lrts=False, progress_wrap=identity),
        num_items=len(exposure_df),
        num_replicates=num_bootstrap,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
    del exposure_df, fit_traits  # clear memory
    gc.collect()
    log_memory_usage(logging.info)

    return pl.concat(
        [
            estimates_df,
            tgs_summarize_bootstrap(
                estimates_df, replicate_fits=replicate_fits
            ),
        ],
        how="horizontal",
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Estimate per-trait gain rate and selection coefficient, with bootstrap standard errors, z-scores, and p-values.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Input schema is as for phyloframe.legacy._alifestd_tgs_estimate_polars.


Output Schema
=============
One row per trait, not a phylogeny, with the columns of
phyloframe.legacy._alifestd_tgs_estimate_polars, then:

'selection_coef_se', 'log_gain_ratio_se' : float
    Bootstrap standard errors of the selection coefficient and log gain
    ratio.

'selection_coef_z', 'gain_ratio_z', 'selection_coef_p', 'gain_ratio_p' : float
    Corresponding z-scores and two-sided normal p-values.

'num_bootstrap_degenerate' : integer
    Replicates excluded for resampling no gains of the trait, or of its
    neutral reference.


Additional Notes
================
- Requires contiguous, topologically sorted ids, and no unifurcations
  (non-root nodes with one child).

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_tgs_estimate_polars :
    Point estimates only.
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
            "phyloframe.legacy._alifestd_tgs_estimate_bootstrap_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    parser = _add_estimate_arguments(parser)
    parser.add_argument(
        "--num-bootstrap",
        required=True,
        type=int,
        help="Number of bootstrap replicates; must be positive.",
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
        help="joblib parallel backend for bootstrap (default: None).",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    with log_context_duration(
        "phyloframe.legacy._alifestd_tgs_estimate_bootstrap_polars",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=functools.partial(
                alifestd_tgs_estimate_bootstrap_polars,
                **_get_estimate_kwargs(args),
                num_bootstrap=args.num_bootstrap,
                seed=args.bootstrap_seed,
                parallel_backend=args.parallel_backend,
                progress_wrap=functools.partial(
                    tqdm, desc="alifestd_tgs_estimate_bootstrap_polars"
                ),
            ),
        )

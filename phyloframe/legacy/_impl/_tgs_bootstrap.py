import functools
import typing

import pandas as pd

from ..._auxlib._run_seeded_replicates import run_seeded_replicates


def tgs_bootstrap(
    *,
    fit_replicate: typing.Callable,
    fit_kwargs: dict,
    num_bootstrap: int,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns fits to bootstrap resamples, indexed by replicate and trait
    index, from `fit_replicate` (e.g., `tgs_fit_bootstrap_replicate`),
    called with each replicate's `seed` and `fit_kwargs`.
    """
    # each replicate draws as many leaves (or branches) as there are, with
    # replacement, so one may be drawn zero, one, or several times; weighting
    # each by its draw count in the fit is the same as refitting to the
    # resample, without building the resampled dataset
    replicate_fits = run_seeded_replicates(
        functools.partial(fit_replicate, fit_kwargs=fit_kwargs),
        num_replicates=num_bootstrap,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
    return pd.concat(
        replicate_fits,
        keys=range(num_bootstrap),
        names=["replicate", "trait_index"],
    )

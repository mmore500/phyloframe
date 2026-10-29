import functools
import typing

import pandas as pd

from ..._auxlib._run_seeded_replicates import run_seeded_replicates
from ._tgs_fit_bootstrap_replicate import tgs_fit_bootstrap_replicate


def tgs_bootstrap(
    *,
    fit_kwargs: dict,
    num_bootstrap: int,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendant_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns fits to bootstrap resamples of leaves, indexed by replicate and
    trait index; see `tgs_fit_traits`.
    """
    # each replicate draws as many leaves as there are, with replacement, so
    # a leaf may be drawn zero, one, or several times; weighting each leaf by
    # its draw count in the fit is the same as refitting to the resampled
    # leaves, without building the resampled dataset
    replicate_fits = run_seeded_replicates(
        functools.partial(tgs_fit_bootstrap_replicate, fit_kwargs=fit_kwargs),
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

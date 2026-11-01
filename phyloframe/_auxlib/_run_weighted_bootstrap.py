import functools
import typing

import numpy as np

from ._identity import identity
from ._run_seeded_replicates import run_seeded_replicates
from ._sample_bootstrap_counts import sample_bootstrap_counts


def _run_weighted_bootstrap_replicate(
    *, seed: int, func: typing.Callable, num_items: int
) -> typing.Any:
    """Implementation detail for `run_weighted_bootstrap`."""
    weights = sample_bootstrap_counts(
        num_items, rng=np.random.default_rng(seed)
    )
    return func(weights=weights.astype(np.float64))


def run_weighted_bootstrap(
    func: typing.Callable[..., typing.Any],
    *,
    num_items: int,
    num_replicates: int,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = identity,
) -> typing.List[typing.Any]:
    """Call `func(weights=...)` once per bootstrap replicate, and return the
    results in replicate order.

    Each replicate draws `num_items` items with replacement, so an item may
    be drawn zero, one, or several times; `weights` holds each item's draw
    count, as floats. Weighting by draw count is the same as refitting to
    the resample, without building the resampled data. See
    `sample_bootstrap_counts` and, for seeding and parallelism,
    `run_seeded_replicates`.

    Parameters
    ----------
    func : Callable
        Called as `func(weights=weights)`. Must be picklable for
        process-based backends (e.g., a `functools.partial` of a module-level
        function).
    num_items : int
        Number of items resampled.
    num_replicates : int
        Number of replicates.
    seed : int, optional
        Seed for deriving replicate seeds.
    parallel_backend : str, optional
        joblib backend to run replicates in parallel.
    progress_wrap : Callable, default identity
        Wraps the iterable of replicate jobs, e.g., `tqdm.tqdm`.

    Returns
    -------
    list
        Result of each replicate.
    """
    return run_seeded_replicates(
        functools.partial(
            _run_weighted_bootstrap_replicate, func=func, num_items=num_items
        ),
        num_replicates=num_replicates,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )

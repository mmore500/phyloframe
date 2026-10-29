import typing

import joblib
import numpy as np
import opytional as opyt


def run_seeded_replicates(
    func: typing.Callable[..., typing.Any],
    *,
    num_replicates: int,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = lambda x: x,
) -> typing.List[typing.Any]:
    """Call `func(seed=...)` once per replicate, each with its own seed, and
    return the results in replicate order.

    Replicate seeds are derived from `seed` via `np.random.SeedSequence`,
    so results do not depend on how or in what order replicates run.

    Parameters
    ----------
    func : Callable
        Called as `func(seed=replicate_seed)`, with an integer seed. Must
        be picklable for process-based backends.
    num_replicates : int
        Number of replicates.
    seed : int, optional
        Seed for deriving replicate seeds; drawn from numpy's global random
        state if not given.
    parallel_backend : str, optional
        joblib backend to run replicates in parallel, e.g., "loky"
        (processes) or "threading"; runs sequentially if not given.
    progress_wrap : Callable, default identity
        Wraps the iterable of replicate jobs, e.g., `tqdm.tqdm`.

    Returns
    -------
    list
        Result of each replicate.
    """
    seed = opyt.or_else(seed, lambda: np.random.randint(2**31))
    replicate_seeds = np.random.SeedSequence(seed).generate_state(
        num_replicates
    )
    jobs = (
        joblib.delayed(func)(seed=int(replicate_seed))
        for replicate_seed in replicate_seeds
    )
    with joblib.parallel_backend(parallel_backend or "sequential", n_jobs=-1):
        return joblib.Parallel()(progress_wrap(jobs))

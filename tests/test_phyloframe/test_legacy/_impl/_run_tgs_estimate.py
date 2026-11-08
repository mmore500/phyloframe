import typing

import pandas as pd
import polars as pl


def run_tgs_estimate(
    estimator: typing.Callable,
    bootstrapper: typing.Callable,
    phylogeny_df: pl.DataFrame,
    *,
    num_bootstrap: int = 0,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Run a tgs estimator, returning pandas; with `num_bootstrap`, run its
    bootstrap counterpart instead, for bootstrap summary columns too."""
    if num_bootstrap:
        estimates_df = bootstrapper(
            phylogeny_df,
            **kwargs,
            num_bootstrap=num_bootstrap,
            seed=seed,
            parallel_backend=parallel_backend,
        )
    else:
        estimates_df = estimator(phylogeny_df, **kwargs)
    return estimates_df.to_pandas()

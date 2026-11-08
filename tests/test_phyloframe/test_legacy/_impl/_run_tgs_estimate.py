import typing

import pandas as pd
import polars as pl

from phyloframe.legacy import alifestd_tgs_summarize_bootstrap_polars


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
    """Run a tgs estimator, returning pandas; with `num_bootstrap`, also run
    its bootstrap counterpart and append the bootstrap summary."""
    estimates_df = estimator(phylogeny_df, **kwargs)
    if num_bootstrap:
        estimates_df = alifestd_tgs_summarize_bootstrap_polars(
            estimates_df,
            bootstrapper(
                phylogeny_df,
                **kwargs,
                num_bootstrap=num_bootstrap,
                seed=seed,
                parallel_backend=parallel_backend,
            ),
        )
    return estimates_df.to_pandas()

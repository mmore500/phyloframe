import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import alifestd_tgs_summarize_bootstrap_polars


def _estimates() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trait": ["a", "b"],
            "num_leaf_episodes": [10, 20],
            "gain_rate": [1e-3, 2e-3],
            "selection_coef": [-0.05, 0.0],
            "gain_ratio": [0.5, 2.0],
            "selection_coef_lrt_naive": [9.0, 0.1],
            "gain_ratio_lrt_naive": [3.0, 4.0],
        },
    )


def _replicates(order=("a", "b")) -> pl.DataFrame:
    estimates = _estimates()
    return pl.concat(
        estimates.select(
            pl.lit(replicate, dtype=pl.Int64).alias("replicate"),
            "trait",
            "gain_rate",
            pl.col("selection_coef") + delta,
            "selection_coef_lrt_naive",
            "gain_ratio",
            "gain_ratio_lrt_naive",
        )
        for replicate, delta in enumerate((-0.01, 0.0, 0.01))
    ).sort(pl.col("trait").replace_strict(list(order), range(len(order))))


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_summary_appended(order):
    out = alifestd_tgs_summarize_bootstrap_polars(
        _estimates(), _replicates(order)
    )
    assert out.columns[: _estimates().width] == _estimates().columns
    np.testing.assert_allclose(out["selection_coef_se"], 0.01)
    np.testing.assert_allclose(out["selection_coef_z"], [-5.0, 0.0])
    assert out["num_bootstrap_degenerate"].to_list() == [0, 0]
    # gain ratios are constant across replicates, so have zero spread
    np.testing.assert_allclose(out["log_gain_ratio_se"], 0.0)


def test_unknown_trait_raises():
    with pytest.raises(ValueError, match="absent"):
        alifestd_tgs_summarize_bootstrap_polars(
            _estimates().head(1), _replicates()
        )

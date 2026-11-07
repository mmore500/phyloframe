import numpy as np
import polars as pl

from phyloframe.legacy._impl import tgs_build_estimates_df


def _counts() -> pl.DataFrame:
    return pl.DataFrame({"trait": ["a", "b"], "num_leaves": [10, 20]})


def _point() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "gain_rate": [1e-3, 2e-3],
            "selection_coef": [-0.05, 0.0],
            "selection_coef_lrt_naive": [9.0, 0.1],
            "gain_ratio": [0.5, 2.0],
            "gain_ratio_lrt_naive": [3.0, 4.0],
        },
    )


_COLUMNS = [
    "trait",
    "num_leaves",
    "gain_rate",
    "selection_coef",
    "gain_ratio",
    "selection_coef_lrt_naive",
    "gain_ratio_lrt_naive",
]


def test_no_bootstrap():
    out = tgs_build_estimates_df(
        _counts(),
        point_estimate_df=_point(),
        bootstrap_fits_df=None,
    )
    assert out.columns == _COLUMNS
    assert out["gain_ratio"].to_list() == [0.5, 2.0]


def test_summary_appended():
    point = _point()
    replicate_fits = pl.concat(
        point.select(
            pl.lit(replicate).alias("replicate"),
            pl.int_range(pl.len()).alias("trait_index"),
            pl.all(),
        ).with_columns(pl.col("selection_coef") + delta)
        for replicate, delta in enumerate((-0.01, 0.0, 0.01))
    )
    out = tgs_build_estimates_df(
        _counts(),
        point_estimate_df=point,
        bootstrap_fits_df=replicate_fits,
    )
    assert out.columns[: len(_COLUMNS)] == _COLUMNS
    np.testing.assert_allclose(out["selection_coef_se"], 0.01)

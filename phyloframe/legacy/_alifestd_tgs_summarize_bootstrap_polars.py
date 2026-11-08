import polars as pl

from .._auxlib._warn_experimental import warn_experimental
from ._impl._tgs_summarize_bootstrap import tgs_summarize_bootstrap


@warn_experimental
def alifestd_tgs_summarize_bootstrap_polars(
    estimates_df: pl.DataFrame,
    replicates_df: pl.DataFrame,
) -> pl.DataFrame:
    """Append bootstrap standard errors, z-scores, and p-values to trait
    gain/selection (tgs) estimates.

    Takes point estimates from `alifestd_tgs_pendantmdc_estimate_polars` or
    `alifestd_tgs_allmdc_estimate_polars` and replicate fits from the
    matching `alifestd_tgs_pendantmdc_bootstrap_polars` or
    `alifestd_tgs_allmdc_bootstrap_polars`, run with the same arguments.

    Replicates resampling no carriers of a trait, or of its neutral
    reference, leave its selection coefficient or gain ratio undefined, so
    are excluded from that trait's statistics, with a warning.

    Parameters
    ----------
    estimates_df : polars.DataFrame
        Point estimates, one row per trait; see Input Schema.
    replicates_df : polars.DataFrame
        Replicate fits, one row per replicate and trait; see Input Schema.

    Returns
    -------
    polars.DataFrame
        `estimates_df` with summary columns appended; see Output Schema.

    Raises
    ------
    ValueError
        If `replicates_df` names a trait absent from `estimates_df`.

    Input Schema
    ------------
    `estimates_df` columns:

    'trait' : string
        Trait name.
    'selection_coef', 'gain_ratio' : float
        Point estimates.

    `replicates_df` columns:

    'replicate' : integer
        Bootstrap replicate index.
    'trait' : string
        Trait name.
    'selection_coef', 'gain_ratio' : float
        Replicate fits.

    Output Schema
    -------------
    `estimates_df` columns, then:

    'selection_coef_se' : float
        Bootstrap standard error of the selection coefficient, over
        replicates with a defined selection coefficient; nan with fewer
        than two.
    'selection_coef_z' : float
        Selection coefficient over its standard error.
    'selection_coef_p' : float
        Two-sided normal p-value of `selection_coef_z`.
    'log_gain_ratio_se' : float
        Bootstrap standard error of the log gain ratio, over replicates
        with a defined gain ratio; nan with fewer than two.
    'gain_ratio_z' : float
        Log gain ratio over its standard error.
    'gain_ratio_p' : float
        Two-sided normal p-value of `gain_ratio_z`.
    'num_bootstrap_degenerate' : integer
        Replicates excluded for an undefined selection coefficient or gain
        ratio.

    See Also
    --------
    alifestd_tgs_allmdc_bootstrap_polars :
        Replicate fits of `alifestd_tgs_allmdc_estimate_polars`.
    alifestd_tgs_pendantmdc_bootstrap_polars :
        Replicate fits of `alifestd_tgs_pendantmdc_estimate_polars`.
    """
    bootstrap_fits_df = replicates_df.join(
        estimates_df.select("trait").with_row_index("trait_index"),
        on="trait",
        how="left",
    ).with_columns(pl.col("trait_index").cast(pl.Int64))
    if bootstrap_fits_df["trait_index"].has_nulls():
        raise ValueError(
            "replicates_df names traits absent from estimates_df",
        )
    return pl.concat(
        [
            estimates_df,
            tgs_summarize_bootstrap(
                point_estimate_df=estimates_df,
                bootstrap_fits_df=bootstrap_fits_df,
            ),
        ],
        how="horizontal",
    )

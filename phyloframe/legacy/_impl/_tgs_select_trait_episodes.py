import polars as pl
import polars.selectors as cs


def tgs_count_exposure_traits(exposure_df: pl.DataFrame) -> int:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns the number of traits in `exposure_df` (see
    `tgs_calc_exposure_df`).
    """
    return exposure_df.select(cs.matches(r"^trait\d+_starts_absent$")).width


def tgs_select_trait_episodes(
    exposure_df: pl.DataFrame, *, trait: int
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns one row per exposure episode of `trait` (index), i.e., per
    branch of `exposure_df` (see `tgs_calc_exposure_df`) starting without
    the trait; see Output Schema. Callers process one trait at a time, so
    that memory scales with branches, not branches times traits.

    Output Schema
    -------------
    'exposure_row' : integer
        Row index of the episode's branch in `exposure_df`.
    'gain' : boolean
        Whether the branch gained the trait.
    """
    return exposure_df.select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias("exposure_row"),
        pl.col(f"trait{trait}_gain").alias("gain"),
    ).filter(exposure_df[f"trait{trait}_starts_absent"])

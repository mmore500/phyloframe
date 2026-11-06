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


def tgs_unpivot_exposure_episodes(exposure_df: pl.DataFrame) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns one row per exposure episode, i.e., per trait and branch of
    `exposure_df` (see `tgs_calc_exposure_df`) starting without the trait,
    trait-major, with columns `trait` (index), `branch` (row index into
    `exposure_df`), and `gain`, whether the branch gained the trait.
    """
    schema = {"trait": pl.Int64, "branch": pl.Int64, "gain": pl.Boolean}
    return pl.concat(
        [
            pl.DataFrame(schema=schema),  # in case there are no traits
            *(
                exposure_df.select(
                    pl.lit(trait, dtype=pl.Int64).alias("trait"),
                    pl.int_range(pl.len(), dtype=pl.Int64).alias("branch"),
                    pl.col(f"trait{trait}_gain").alias("gain"),
                ).filter(exposure_df[f"trait{trait}_starts_absent"])
                for trait in range(tgs_count_exposure_traits(exposure_df))
            ),
        ],
    )

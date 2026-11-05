def tgs_validate_allmdc_args(
    *,
    calibrate_size_partition_ratio: float,
    calibrate_size_partition_min_neutral_gains: int,
) -> None:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.
    Acronym tgs stands for trait gain/selection.

    Validates arguments specific to the allmdc estimator; see also
    `tgs_validate_args`.
    """
    if not calibrate_size_partition_ratio > 1:
        raise ValueError("calibrate_size_partition_ratio must exceed 1")
    if calibrate_size_partition_min_neutral_gains < 1:
        raise ValueError(
            "calibrate_size_partition_min_neutral_gains must be at least 1"
        )

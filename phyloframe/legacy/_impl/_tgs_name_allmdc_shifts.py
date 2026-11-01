import typing


def tgs_name_allmdc_shifts(
    group_bounds: typing.Sequence[typing.Tuple[int, int]],
) -> typing.List[str]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns output column names for the shifts of clade-size groups after
    the first (single-leaf) group, given each group's (fewest, most)
    leaves, e.g., `clade_size_shift_2` and `clade_size_shift_4_5`.
    """
    return [
        f"clade_size_shift_{lower}" + (f"_{upper}" if upper != lower else "")
        for lower, upper in group_bounds[1:]
    ]

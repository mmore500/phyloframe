import typing


def tgs_find_reference_traits(
    *, neutral_traits: typing.Sequence[int], trait: int
) -> typing.List[int]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars`.
    Acronym tgs stands for trait gain/selection.

    Returns `trait`'s neutral reference: the neutral traits other than
    itself, so that a neutral trait is not compared against itself.
    """
    return sorted(set(neutral_traits) - {trait})

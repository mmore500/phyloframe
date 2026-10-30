import typing


def tgs_find_reference_traits(
    *, neutral_traits: typing.Sequence[int], trait: int
) -> typing.List[int]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendant_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns `trait`'s neutral reference: the neutral traits other than
    itself, so that a neutral trait is not compared against itself.
    """
    return sorted(set(neutral_traits) - {trait})

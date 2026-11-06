import pandas as pd

from phyloframe.legacy._impl import tgs_extract_trait_masks


def test_masks():
    exposure_df = pd.DataFrame(
        {
            "clade_size": [1, 1, 2],
            "trait0_starts_absent": [True, False, True],
            "trait0_gain": [True, False, False],
            "trait1_starts_absent": [True, True, True],
            "trait1_gain": [False, False, True],
        },
    )
    starts_absent, gains = tgs_extract_trait_masks(exposure_df)
    assert starts_absent.dtype == gains.dtype == bool
    assert starts_absent.tolist() == [[1, 1], [0, 1], [1, 1]]
    assert gains.tolist() == [[1, 0], [0, 0], [0, 1]]


def test_empty():
    exposure_df = pd.DataFrame(
        {"trait0_starts_absent": [], "trait0_gain": []}, dtype=bool
    )
    starts_absent, gains = tgs_extract_trait_masks(exposure_df)
    assert starts_absent.shape == gains.shape == (0, 1)

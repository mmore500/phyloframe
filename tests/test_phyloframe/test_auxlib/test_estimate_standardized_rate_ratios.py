import numpy as np
import pytest

from phyloframe._auxlib._estimate_standardized_rate_ratios import (
    estimate_standardized_rate_ratios,
)


def test_uniform_rate():
    exposure = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = estimate_standardized_rate_ratios(
        num_events=0.1 * exposure,
        exposure=exposure,
        group_ids=np.array([0, 1]),
        pseudocount=0.0,
    )
    np.testing.assert_allclose(res, np.ones((2, 2)))


def test_recovers_distortion():
    true_exposure = np.array([10.0, 10.0, 10.0, 10.0])
    measured = true_exposure * np.array([2.0, 2.0, 0.5, 0.5])
    num_events = 0.01 * true_exposure
    ratio = estimate_standardized_rate_ratios(
        num_events=num_events,
        exposure=measured,
        group_ids=[0, 0, 1, 1],
        pseudocount=0.0,
    )
    corrected = measured * ratio
    # corrected exposure is proportional to true exposure
    np.testing.assert_allclose(corrected / corrected[0], np.ones(4))


def test_groups_pool_bins():
    ratio = estimate_standardized_rate_ratios(
        num_events=[1.0, 3.0, 0.0],
        exposure=[1.0, 1.0, 2.0],
        group_ids=[5, 5, 7],
        pseudocount=0.0,
    )
    # group 5 rate 2.0, group 7 rate 0.0, overall rate 1.0
    np.testing.assert_allclose(ratio, [2.0, 2.0, 0.0])


def test_pseudocount_smooths_empty_group():
    ratio = estimate_standardized_rate_ratios(
        num_events=[1.0, 3.0, 0.0],
        exposure=[1.0, 1.0, 2.0],
        group_ids=[5, 5, 7],
    )
    # group 5: (4 + 1) / (2 + 1); group 7: (0 + 1) / (2 + 1)
    np.testing.assert_allclose(ratio, [5 / 3, 5 / 3, 1 / 3])
    assert (ratio > 0).all()


def test_pseudocount_vanishes_with_data():
    ratio = estimate_standardized_rate_ratios(
        num_events=[1e6, 0.0, 0.0],
        exposure=[1e6, 1e6, 0.0],
        group_ids=[0, 1, 1],
    )
    np.testing.assert_allclose(ratio, [2.0, 0.0, 0.0], atol=1e-5)


def test_no_exposure_group():
    ratio = estimate_standardized_rate_ratios(
        num_events=[1.0, 0.0],
        exposure=[1.0, 0.0],
        group_ids=[0, 1],
        pseudocount=0.0,
    )
    np.testing.assert_allclose(ratio, [1.0, 1.0])


@pytest.mark.parametrize("pseudocount", [0.0, 1.0])
def test_no_events(pseudocount: float):
    ratio = estimate_standardized_rate_ratios(
        num_events=[0.0, 0.0],
        exposure=[1.0, 2.0],
        group_ids=[0, 1],
        pseudocount=pseudocount,
    )
    np.testing.assert_allclose(ratio, [1.0, 1.0])


def test_preserves_shape():
    num_events = np.ones((3, 4))
    group_ids = np.arange(12).reshape(3, 4) % 2
    ratio = estimate_standardized_rate_ratios(
        num_events=num_events, exposure=np.ones((3, 4)), group_ids=group_ids
    )
    assert ratio.shape == (3, 4)
    assert ratio == pytest.approx(np.ones((3, 4)))


def test_keyword_only():
    with pytest.raises(TypeError):
        estimate_standardized_rate_ratios([1.0], [1.0], [0])

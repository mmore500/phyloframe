import typing

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_clade_size_geom_classes_polars,
    alifestd_mark_joint_quantile_bins_polars,
    alifestd_mark_max_descendant_origin_time_polars,
    alifestd_mark_origin_time_delta_polars,
    alifestd_tgs_estimate_polars,
)

_TRAITS = ["focal", "n1", "n2", "n3", "n4"]
_FOCAL_SELECTION_COEF = -0.04


def _make_clades(
    survival_bias: float, num_clades: int = 30_000, seed: int = 1
) -> pl.DataFrame:
    """Root with clades hanging off it, each a stem node A with two child
    nodes B, each with two leaves on short pendant branches.

    Each inner branch gains a trait under the full model, with its chance
    of gain also scaled by exp(`survival_bias` * survival duration) for
    every trait, neutral ones included; leaves gain it under the pendant
    model. Trait `focal` has twice the neutral gain rate and a cost; traits
    `n1` to `n4` are neutral.
    """
    rng = np.random.default_rng(seed)
    num_nodes = 1 + 7 * num_clades
    a = np.arange(1, num_nodes, 7)
    b = np.stack([a + 1, a + 4], axis=1)
    leaves = np.stack([a + 2, a + 3, a + 5, a + 6], axis=1)
    ancestor_id = np.zeros(num_nodes, dtype=int)
    ancestor_id[b] = a[:, None]
    ancestor_id[leaves[:, :2]] = b[:, [0]]
    ancestor_id[leaves[:, 2:]] = b[:, [1]]

    a_length = rng.uniform(0.5, 30, num_clades)
    b_length = rng.uniform(5, 40, (num_clades, 2))
    leaf_length = rng.uniform(0.5, 5, (num_clades, 4))
    origin_time = np.zeros(num_nodes)
    origin_time[a] = a_length
    origin_time[b] = a_length[:, None] + b_length
    origin_time[leaves[:, :2]] = origin_time[b[:, [0]]] + leaf_length[:, :2]
    origin_time[leaves[:, 2:]] = origin_time[b[:, [1]]] + leaf_length[:, 2:]
    b_survival = np.stack(
        [leaf_length[:, :2].max(axis=1), leaf_length[:, 2:].max(axis=1)],
        axis=1,
    )
    a_survival = (b_length + b_survival).max(axis=1)

    columns = {
        "id": np.arange(num_nodes),
        "ancestor_id": ancestor_id,
        "origin_time": origin_time,
    }
    for trait, (gain, selection_coef) in zip(
        _TRAITS, [(2e-3, _FOCAL_SELECTION_COEF)] + [(1e-3, 0.0)] * 4
    ):

        def p(length: np.ndarray) -> np.ndarray:
            if selection_coef == 0:
                return gain * length
            return gain * np.expm1(selection_coef * length) / selection_coef

        survival_rate = selection_coef + survival_bias
        a_gain = rng.random(num_clades) < p(a_length) * np.exp(
            survival_rate * a_survival
        )
        b_gain = (
            rng.random((num_clades, 2))
            < p(b_length) * np.exp(survival_rate * b_survival)
        ) & ~a_gain[:, None]
        leaf_gain = (
            rng.random((num_clades, 4)) < p(leaf_length)
        ) & ~np.repeat(a_gain[:, None] | b_gain, 2, axis=1)
        transitions = np.zeros(num_nodes, dtype=np.int8)
        transitions[a] = a_gain
        transitions[b] = b_gain
        transitions[leaves] = leaf_gain
        columns[trait] = transitions

    return pl.DataFrame(columns)


def _estimate(
    phylogeny_df: pl.DataFrame, *, calibrate_by: typing.Sequence[str]
) -> pl.DataFrame:
    """Estimate with calibration bins made from, and compared by,
    `calibrate_by`, as the estimator's Examples show."""
    return (
        alifestd_mark_origin_time_delta_polars(
            phylogeny_df, mark_as="branch_length"
        )
        .pipe(alifestd_mark_max_descendant_origin_time_polars)
        .with_columns(
            survival_duration=pl.col("max_descendant_origin_time")
            - pl.col("origin_time"),
        )
        .pipe(
            alifestd_mark_clade_size_geom_classes_polars,
            weight_by=_TRAITS[1:],
            min_class_weight=20,
        )
        .pipe(
            alifestd_mark_joint_quantile_bins_polars,
            bin_by=calibrate_by,
            partition_by="clade_size_geom_class",
        )
        .pipe(
            alifestd_tgs_estimate_polars,
            trait_transitions=_TRAITS,
            neutral_traits=_TRAITS[1:],
            calibration_partition_by="clade_size_geom_class",
            calibration_compare_by_bins="joint_quantile_bin",
            calibration_compare_by_vals=calibrate_by,
        )
    )


_WITHOUT_SURVIVAL = ["branch_length", "origin_time"]
_WITH_SURVIVAL = ["branch_length", "origin_time", "survival_duration"]


def test_survival_duration_marked_as_documented():
    # leaves have zero survival duration; inner nodes, time to latest leaf
    phylogeny_df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 1],
            "origin_time": [0.0, 1.0, 3.0, 6.0],
        },
    )
    res = alifestd_mark_max_descendant_origin_time_polars(phylogeny_df)
    survival_duration = res["max_descendant_origin_time"] - res["origin_time"]
    assert survival_duration.to_list() == [6.0, 5.0, 0.0, 0.0]


def test_survival_calibration_reduces_neutral_survival_bias():
    # inner branches' gains are depressed with survival duration for every
    # trait, so neutral traits appear costly
    phylogeny_df = _make_clades(survival_bias=-0.05)
    without = _estimate(phylogeny_df, calibrate_by=_WITHOUT_SURVIVAL)
    with_ = _estimate(phylogeny_df, calibrate_by=_WITH_SURVIVAL)

    neutral_without = without["selection_coef"][1:].mean()
    neutral_with = with_["selection_coef"][1:].mean()
    assert neutral_without < -0.003
    # binning by survival duration brings neutral traits closer to neutral
    assert abs(neutral_with) < 0.75 * abs(neutral_without)

    for res in without, with_:
        assert res["selection_coef"][0] == pytest.approx(
            _FOCAL_SELECTION_COEF, abs=0.01
        )
        assert res["gain_ratio"][0] == pytest.approx(2.0, rel=0.25)


def test_survival_calibration_harmless_without_bias():
    phylogeny_df = _make_clades(survival_bias=0.0)
    without = _estimate(phylogeny_df, calibrate_by=_WITHOUT_SURVIVAL)
    with_ = _estimate(phylogeny_df, calibrate_by=_WITH_SURVIVAL)

    for res in without, with_:
        assert abs(res["selection_coef"][1:].mean()) < 0.002
        assert res["selection_coef"][0] == pytest.approx(
            _FOCAL_SELECTION_COEF, abs=0.01
        )
    np.testing.assert_allclose(
        with_["selection_coef"], without["selection_coef"], atol=2e-3
    )

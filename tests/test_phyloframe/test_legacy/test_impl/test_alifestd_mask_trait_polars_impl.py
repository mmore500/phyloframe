import numpy as np
import polars as pl
import pytest

from phyloframe.legacy._impl import alifestd_mask_trait_polars_impl


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 0, 0], "t": [False, True, False]}
    )


def test_dispatches_to_fast_path():
    seen = {}

    def fast_path(ancestor_ids: np.ndarray, trait_mask: np.ndarray):
        seen["args"] = (ancestor_ids.tolist(), trait_mask.tolist())
        return trait_mask

    res = alifestd_mask_trait_polars_impl(
        _df(), trait_mask="t", mark_as="m", fast_path=fast_path, caller="x"
    )
    assert seen["args"] == ([0, 0, 0], [False, True, False])
    assert res["m"].to_list() == [False, True, False]


def test_empty():
    res = alifestd_mask_trait_polars_impl(
        _df().clear(),
        trait_mask="t",
        mark_as="m",
        fast_path=lambda *a: pytest.fail("not called"),
        caller="x",
    )
    assert res["m"].to_list() == []


def test_null_mask_raises():
    df = _df().with_columns(pl.Series("t", [None, True, False]))
    with pytest.raises(NotImplementedError):
        alifestd_mask_trait_polars_impl(
            df, trait_mask="t", mark_as="m", fast_path=None, caller="x"
        )

import polars as pl
import polars.selectors as cs


def mask_persistence_validate_values(selected: pl.DataFrame) -> None:
    """Implementation detail for `alifestd_mask_persistence_polars`.

    Raises NotImplementedError for null mask values; for non-numeric, null,
    or non-finite criterion values; and for cherry or trunk values that
    decrease from ancestor to descendant.
    """
    for name in "candidate_mask", "extant_mask":
        if selected[name].has_nulls():
            raise NotImplementedError(f"null {name} values not supported")

    is_finite = pl.all().is_finite().fill_null(False)
    is_decreasing = pl.all() < pl.all().gather(selected["ancestor_id"])
    for name in "criterion_cherry", "criterion_trunk", "criterion_tiebreak":
        fields = selected.select(name).unnest(cs.struct())  # column per field
        if not all(dtype.is_numeric() for dtype in fields.dtypes):
            raise NotImplementedError(
                f"non-numeric {name} values not supported",
            )
        if not all(fields.select(is_finite.all()).row(0)):
            raise NotImplementedError(
                f"null or non-finite {name} values not supported",
            )
        if name != "criterion_tiebreak" and any(
            fields.select(is_decreasing.any()).row(0)
        ):
            raise NotImplementedError(
                f"{name} values decreasing from ancestor to descendant "
                "not supported",
            )

import polars as pl


def alifestd_make_empty_polars(
    ancestor_id: bool = False,
) -> pl.DataFrame:
    """Create an alife standard phylogeny dataframe with zero rows."""
    schema = {
        "id": pl.Int64,
    }
    if ancestor_id:
        schema["ancestor_id"] = pl.Int64

    return pl.DataFrame(schema=schema)

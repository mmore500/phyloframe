import polars as pl

from ._alifestd_from_avida_spop import (
    _parse_spop_ancestor_list,
    _parse_spop_text,
)
from ._alifestd_make_empty_polars import alifestd_make_empty_polars


def alifestd_from_avida_spop_polars(
    spop_text: str,
    *,
    create_ancestor_list: bool = True,
) -> pl.DataFrame:
    """Convert Avida ``.spop`` population snapshot text to a phylogeny
    dataframe.

    Parses the text content of an Avida ``.spop`` (structured population)
    file and returns a polars DataFrame in alife standard format.

    Parameters
    ----------
    spop_text : str
        Full text content of an Avida ``.spop`` file.
    create_ancestor_list : bool, default True
        If True, include an ``ancestor_list`` column in the result.

    Returns
    -------
    pl.DataFrame
        Phylogeny dataframe in alife standard format.

    See Also
    --------
    alifestd_from_avida_spop :
        Pandas-based implementation.

    Raises
    ------
    ValueError
        If the ``#format`` header is missing from the spop text.
    """
    header, avida_data = _parse_spop_text(spop_text)

    if not avida_data["id"]:
        return alifestd_make_empty_polars()

    # Build alife-standard columns.
    result_data = {}
    result_data["id"] = pl.Series(avida_data["id"]).cast(pl.Int64)

    if create_ancestor_list:
        result_data["ancestor_list"] = pl.Series(
            [_parse_spop_ancestor_list(p) for p in avida_data["parents"]],
            dtype=pl.Utf8,
        )

    result_data["origin_time"] = pl.Series(
        avida_data["update_born"],
    ).cast(pl.Int64)

    # Add remaining Avida fields under their original names.
    for field in set(header) - {"id", "parents", "update_born"}:
        result_data[field] = pl.Series(
            avida_data[field],
            dtype=pl.Utf8,
        )

    return pl.DataFrame(result_data)

import argparse
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_splay_polytomies import (
    _alifestd_splay_polytomies_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_splay_polytomies_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    r"""Use a simple splay strategy to resolve polytomies, converting them
    into bifurcations.

    No adjustments to any branch length columns are performed. Nodes added
    to perform the splaying-out will have zero-length subtending branches.
    """

    logging.info(
        "- alifestd_splay_polytomies_polars: adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df

    logging.info(
        "- alifestd_splay_polytomies_polars: checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not yet supported",
        )

    logging.info(
        "- alifestd_splay_polytomies_polars: " "checking topological sort...",
    )
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "topologically unsorted rows not yet supported",
        )

    logging.info(
        "- alifestd_splay_polytomies_polars: extracting arrays...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
        .copy()
    )

    logging.info(
        "- alifestd_splay_polytomies_polars: computing splay...",
    )
    (
        splayed_ancestor_ids,
        new_source_ids,
        new_ids,
        new_ancestor_ids,
    ) = _alifestd_splay_polytomies_fast_path(ancestor_ids)

    df = phylogeny_df.lazy().collect()

    # Update ancestor_ids for existing rows
    df = df.with_columns(ancestor_id=splayed_ancestor_ids)

    if len(new_ids) > 0:
        # Build addendum from source rows
        addendum = df.select(pl.all().gather(new_source_ids))
        addendum = addendum.with_columns(
            id=np.array(new_ids, dtype=np.int64),
            ancestor_id=np.array(new_ancestor_ids, dtype=np.int64),
        )
        df = pl.concat([df, addendum], how="align")

    return df


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Use a simple splay strategy to resolve polytomies into bifurcations.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.
"""


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        description=format_cli_description(_raw_description),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = _add_parser_base(
        parser=parser,
        dfcli_module=("phyloframe.legacy._alifestd_splay_polytomies_polars"),
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_splay_polytomies_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=alifestd_splay_polytomies_polars,
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e

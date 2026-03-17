import argparse
import functools
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
from ._alifestd_mark_csr_children_asexual import (
    _alifestd_mark_csr_children_asexual_fast_path,
)
from ._alifestd_mark_csr_offsets_asexual import (
    _alifestd_mark_csr_offsets_asexual_fast_path,
)


def alifestd_mark_csr_children_polars(
    phylogeny_df: pl.DataFrame,
    *,
    mark_as: str = "csr_children",
) -> pl.DataFrame:
    """Add column `csr_children`, a flat array of child ids grouped by parent
    according to CSR offsets from the `csr_offsets` column.

    The output column name can be changed via the ``mark_as`` parameter.
    """

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):

        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):

        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    logging.info(
        "- alifestd_mark_csr_children_polars: extracting ancestor ids...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )

    if "csr_offsets" in phylogeny_df.lazy().collect_schema().names():
        logging.info(
            "- alifestd_mark_csr_children_polars: "
            "using existing csr_offsets column...",
        )
        csr_offsets = (
            phylogeny_df.lazy()
            .select("csr_offsets")
            .collect()
            .to_series()
            .to_numpy()
            .astype(np.int64)
        )
    else:
        logging.info(
            "- alifestd_mark_csr_children_polars: "
            "computing csr_offsets offsets...",
        )
        csr_offsets = _alifestd_mark_csr_offsets_asexual_fast_path(
            ancestor_ids,
        )

    logging.info(
        "- alifestd_mark_csr_children_polars: "
        "scattering children into flat array...",
    )
    csr_children = _alifestd_mark_csr_children_asexual_fast_path(
        ancestor_ids.astype(np.int64),
        csr_offsets,
    )

    return phylogeny_df.with_columns(
        pl.Series(csr_children).alias(mark_as),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `csr_children`, a flat array of child ids grouped by parent according to CSR offsets.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_csr_children_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="csr_children",
        type=str,
        help="output column name (default: csr_children)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_csr_children_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_csr_children_polars, mark_as=args.mark_as
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e

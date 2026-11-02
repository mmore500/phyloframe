import argparse
import functools
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import polars as pl

from .._auxlib._add_bool_arg import add_bool_arg
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
from ._alifestd_regraft_splice_chronological_consistency_asexual import (
    _alifestd_regraft_splice_chronological_consistency_asexual_impl,
)
from ._alifestd_topological_sensitivity_warned_polars import (
    alifestd_topological_sensitivity_warned_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


@alifestd_topological_sensitivity_warned_polars(
    insert=True,
    delete=False,
    update=True,
)
def alifestd_regraft_splice_chronological_consistency_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    """For any taxa with origin time preceding its parent's origin time,
    splice in a new node where its lineage passes through its origin time and
    reattach the taxon there.

    Like `alifestd_regraft_chronological_consistency_polars`, resolves
    origin time inversions by adjusting topology rather than origin times.
    However, rather than regrafting each inconsistent taxon onto its nearest
    eligible ancestor node, regrafts it onto its nearest eligible ancestor
    edge. So, the taxon remains grouped with the clade it originally
    belonged to, as sister to it.

    Specifically, each inconsistent taxon is regrafted, with its subtree,
    onto a new node spliced into the edge between the nearest taxon along its
    lineage with origin time not exceeding its own and that taxon's child
    along its lineage. Spliced nodes take the origin time of the taxon they
    are spliced for (i.e., a zero-length branch subtends the regrafted
    taxon). Taxa that precede their entire lineage are regrafted onto a new
    root spliced above their lineage's root. So, the number of roots is
    unchanged.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topologically sorted rows.

    Returns
    -------
    polars.DataFrame or polars.LazyFrame
        The phylogeny with regrafted `ancestor_id` column (and
        `ancestor_list` column, if present). Original taxa keep their ids and
        rows. Spliced nodes are appended as new rows, with ids following
        original ids and null values for columns other than `id`,
        `ancestor_id`, `ancestor_list`, and `origin_time`. So, output may not
        be topologically sorted.

    See Also
    --------
    alifestd_regraft_splice_chronological_consistency_asexual :
        Pandas-based implementation.
    alifestd_regraft_chronological_consistency_polars :
        Regrafts onto nearest eligible ancestor node, without splicing.
    alifestd_coerce_chronological_consistency_polars :
        Resolves origin time inversions by adjusting origin times.
    """

    logging.info(
        "- alifestd_regraft_splice_chronological_consistency_polars: "
        "adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    schema = phylogeny_df.lazy().collect_schema()
    if "ancestor_id" not in schema.names():
        raise NotImplementedError("sexual phylogenies not supported")

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    logging.info(
        "- alifestd_regraft_splice_chronological_consistency_polars: "
        "extracting columns...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )
    origin_times = (
        phylogeny_df.lazy()
        .select("origin_time")
        .collect()
        .to_series()
        .to_numpy()
    )

    logging.info(
        "- alifestd_regraft_splice_chronological_consistency_polars: "
        "regrafting and splicing taxa...",
    )
    (
        regrafted,
        sources,
    ) = _alifestd_regraft_splice_chronological_consistency_asexual_impl(
        ancestor_ids,
        origin_times,
    )

    num_taxa = len(ancestor_ids)
    phylogeny_df = phylogeny_df.with_columns(
        pl.Series("ancestor_id", regrafted[:num_taxa]).cast(
            schema["ancestor_id"],
        ),
    )

    if len(sources) == 0:
        return phylogeny_df

    spliced_df = pl.DataFrame(
        {
            "id": pl.Series(np.arange(num_taxa, len(regrafted))).cast(
                schema["id"],
            ),
            "ancestor_id": pl.Series(regrafted[num_taxa:]).cast(
                schema["ancestor_id"],
            ),
            "origin_time": pl.Series(
                origin_times[sources],
                dtype=schema["origin_time"],
            ),
        },
    )
    if isinstance(phylogeny_df, pl.LazyFrame):
        spliced_df = spliced_df.lazy()

    phylogeny_df = pl.concat([phylogeny_df, spliced_df], how="diagonal")

    if "ancestor_list" in schema.names():
        logging.info(
            "- alifestd_regraft_splice_chronological_consistency_polars: "
            "updating ancestor_list...",
        )
        is_regrafted = pl.Series(
            np.concatenate(
                [
                    regrafted[:num_taxa] != ancestor_ids,
                    np.ones(len(sources), dtype=bool),
                ],
            ),
        )
        phylogeny_df = phylogeny_df.with_columns(
            ancestor_list=pl.when(is_regrafted)
            .then(
                pl.when(pl.col("id") == pl.col("ancestor_id"))
                .then(pl.lit("[none]"))
                .otherwise(pl.format("[{}]", pl.col("ancestor_id"))),
            )
            .otherwise(pl.col("ancestor_list")),
        )

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

For any taxa with origin time preceding its parent's, splice in a new node where its lineage passes through its origin time and reattach the taxon there.

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires asexual phylogeny with contiguous ids and topologically sorted rows.

- Spliced nodes are appended as new rows, so output may not be topologically sorted.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_regraft_splice_chronological_consistency_asexual :
    CLI entrypoint for Pandas-based implementation.
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
        dfcli_module="phyloframe.legacy._alifestd_regraft_splice_chronological_consistency_polars",
        dfcli_version=get_phyloframe_version(),
    )
    add_bool_arg(
        parser,
        "ignore-topological-sensitivity",
        default=False,
        help="suppress topological sensitivity warning (default: False)",
    )
    add_bool_arg(
        parser,
        "drop-topological-sensitivity",
        default=False,
        help="drop topology-sensitive columns from output (default: False)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_regraft_splice_chronological_consistency_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_regraft_splice_chronological_consistency_polars,
                    ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                    drop_topological_sensitivity=args.drop_topological_sensitivity,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e

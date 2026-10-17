import argparse
import functools
import gc
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._log_memory_usage import log_memory_usage
from ._alifestd_mark_founder_id_polars import (
    alifestd_mark_founder_id_polars,
)
from ._alifestd_mark_monomorphic_clades_polars import (
    alifestd_mark_monomorphic_clades_polars,
)
from ._alifestd_topological_sensitivity_warned_polars import (
    alifestd_topological_sensitivity_warned_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


@alifestd_topological_sensitivity_warned_polars(
    insert=False,
    delete=True,
    update=False,
)
def alifestd_agg_monomorphic_clades_polars(
    phylogeny_df: pl.DataFrame,
    *,
    by: typing.Union[str, typing.Sequence[str]],
    named_aggs: typing.Mapping[str, pl.Expr] = {},
    trait_mask: typing.Optional[str] = None,
) -> pl.DataFrame:
    """Collapse clades whose members all share identical values in `by`
    column(s), aggregating values over each maximal monomorphic clade.

    Each maximal monomorphic clade is replaced by its root, which becomes a
    leaf. Taxa belonging to no monomorphic clade (i.e., polymorphic inner
    nodes) are kept as their own singleton group. Null values compare equal
    to one another.

    Output retains "id", "ancestor_id", "ancestor_list", and `by` columns,
    an "is_monomorphic_clade" column marking collapsed clades (vs.
    polymorphic singletons), plus aggregation results; other columns are
    discarded. By default, `by` columns are aggregated to the group's shared
    value (i.e., from members marked by `trait_mask`, if provided). Override
    by passing an entry of the same name in `named_aggs`.

    For example, to count leaves and take the root's and the latest origin
    times within each collapsed clade, pass

        named_aggs=dict(
            clade_size=pl.col("is_leaf").sum(),
            origin_time_first=pl.col("origin_time").first(),
            origin_time_last=pl.col("origin_time").max(),
        )

    where "is_leaf" may be added with `alifestd_mark_leaves_polars`. Group
    members are aggregated in topological order, so `first` gives the group
    root's value.

    To instead keep the group root's values for all columns while overriding
    some, take `first` over all columns except structural and `by` columns,
    e.g., for `by="trait"`,

        named_aggs={
            **{
                col: pl.col(col).first()
                for col in phylogeny_df.columns
                if col not in ("id", "ancestor_id", "ancestor_list", "trait")
            },
            "origin_time": pl.col("origin_time").max(),
        }

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny.
    by : str or sequence of str
        Name(s) of column(s) whose values determine monomorphism.
    named_aggs : mapping of str to polars.Expr, default {}
        Aggregations over group members, by output column name, forwarded as
        keyword arguments to `polars.DataFrame.group_by(...).agg`.
    trait_mask : str, optional
        Name of boolean column marking taxa whose `by` values are considered.
        Values of unmarked taxa are ignored. For example, pass "is_leaf" to
        consider only leaf values (see `alifestd_mark_leaves_polars`). Clades
        with no marked taxa are considered monomorphic. If None, all taxa are
        considered.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has non-contiguous ids or is not topologically
        sorted.
    ValueError
        If any of `by` or `trait_mask` is not a column in `phylogeny_df` or
        `named_aggs` would overwrite "id", "ancestor_id", or
        "ancestor_list".

    Returns
    -------
    polars.DataFrame
        The collapsed phylogeny in alife standard format, with aggregation
        columns.

    See Also
    --------
    alifestd_mark_monomorphic_clades_polars :
        Marks monomorphic clades without collapsing them.
    """
    if isinstance(by, str):
        by = [by]
    else:
        by = [*by]

    structure_cols = {"id", "ancestor_id", "ancestor_list"}
    if overwritten := structure_cols & {*named_aggs}:
        raise ValueError(f"named_aggs may not overwrite {overwritten}")

    logging.info(
        "- alifestd_agg_monomorphic_clades_polars: "
        "adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()

    logging.info(
        "- alifestd_agg_monomorphic_clades_polars: marking clades...",
    )
    mono_col = "alifestd_agg_monomorphic_clades_is_monomorphic"
    phylogeny_df = alifestd_mark_monomorphic_clades_polars(
        phylogeny_df, by, mark_as=mono_col, trait_mask=trait_mask
    )

    logging.info(
        "- alifestd_agg_monomorphic_clades_polars: calculating groups...",
    )
    # oldest monomorphic taxon along each lineage is its group's root;
    # taxa outside monomorphic clades are their own founder
    group_col = "alifestd_agg_monomorphic_clades_group"
    phylogeny_df = alifestd_mark_founder_id_polars(
        phylogeny_df, founder_mask=mono_col, mark_as=group_col
    )
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_agg_monomorphic_clades_polars: aggregating...",
    )
    by_aggs = {
        col: (
            pl.col(col).first()
            if trait_mask is None
            else pl.when(pl.col(trait_mask).cast(pl.Boolean).any())
            .then(
                pl.col(col).filter(pl.col(trait_mask).cast(pl.Boolean)).first()
            )
            .otherwise(pl.col(col).first())
        )
        for col in by
    }
    aggregated_df = phylogeny_df.group_by(group_col).agg(
        **{
            **by_aggs,
            "is_monomorphic_clade": pl.col(mono_col).first(),
            **named_aggs,
        },
    )

    logging.info(
        "- alifestd_agg_monomorphic_clades_polars: collapsing...",
    )
    return (
        phylogeny_df.filter(pl.col(group_col) == pl.col("id"))
        .drop(*({*schema_names} - structure_cols), mono_col)
        .join(aggregated_df, on=group_col, how="left", maintain_order="left")
        .drop(group_col)
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Collapse clades whose members all share identical values in `--by` column(s),
aggregating values over each maximal monomorphic clade onto its root.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires 'ancestor_id' column to be present in input DataFrame.
Otherwise, no action is taken.

- Use `--trait-mask` to consider only values of taxa marked by a boolean
column, e.g., `--trait-mask is_leaf` to consider only leaf values.

- Use `--agg` to specify aggregations, named by their output column (e.g.,
via `.alias`), e.g.,
`--agg 'pl.col("is_leaf").sum().alias("clade_size")'`,
`--agg 'pl.col("origin_time").first().alias("origin_time_first")'`, or
`--agg 'pl.col("origin_time").max().alias("origin_time_last")'`.

- Output retains 'id', 'ancestor_id', 'ancestor_list', and `--by` columns,
plus aggregation results; other columns are discarded. `--by` columns are
aggregated to the group's shared value by default.

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
        dfcli_module="phyloframe.legacy._alifestd_agg_monomorphic_clades_polars",
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
    parser.add_argument(
        "--by",
        action="append",
        dest="by",
        type=str,
        default=[],
        help=(
            "Name of column whose values determine monomorphism. "
            "Specify multiple columns by using this flag multiple times."
        ),
    )
    parser.add_argument(
        "--trait-mask",
        default=None,
        type=str,
        help=(
            "Name of boolean column marking taxa whose values are "
            "considered (default: all taxa)."
        ),
    )
    parser.add_argument(
        "--agg",
        action="append",
        dest="aggs",
        type=str,
        default=[],
        help=(
            "Python expression evaluating to a polars aggregation, with "
            "polars available as `pl`. "
            "Specify multiple aggregations by using this flag multiple times."
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    named_aggs = {
        expr.meta.output_name(): expr
        for expr in (eval(agg, {"pl": pl}) for agg in args.aggs)  # nosec B307
    }

    with log_context_duration(
        "phyloframe.legacy._alifestd_agg_monomorphic_clades_polars",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=functools.partial(
                alifestd_agg_monomorphic_clades_polars,
                by=args.by,
                trait_mask=args.trait_mask,
                named_aggs=named_aggs,
                ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                drop_topological_sensitivity=args.drop_topological_sensitivity,
            ),
        )

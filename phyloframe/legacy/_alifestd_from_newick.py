import argparse
import ast
import logging
import os
import pathlib
import types
import typing
import warnings

import numpy as np
import pandas as pd
import polars as pl

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._configure_prod_logging import configure_prod_logging
from .._auxlib._eval_kwargs import eval_kwargs
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._jit import jit
from .._auxlib._jit_parse_float import jit_parse_float
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_make_ancestor_list_col import alifestd_make_ancestor_list_col


@jit(nopython=True)
def _parse_newick_jit(
    chars: np.ndarray,
    n: int,
    dtype_id: np.dtype = np.dtype(np.int64),
) -> typing.Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    """Inner JIT kernel for newick parsing.

    Allocates and populates arrays with tree structure and records
    branch-length substring positions for external float conversion.

    Implementation detail for `_parse_newick`.

    Adapted from
    https://github.com/niemasd/TreeSwift/blob/v1.1.45/treeswift/Tree.py#L1439

    Returns
    -------
    tuple
        (ids, ancestor_ids, label_starts, label_stops, label_quoted,
         bl_starts, bl_stops, bl_node_ids, num_nodes, num_bls).
    """
    # estimate max node count using a translation table: each '(' and ','
    # creates a node, each ';' may start a new tree's root, plus the first
    # root. ';' is an upper bound (trailing/empty trees overcount, which is
    # harmless since arrays are trimmed to num_nodes).
    node_weight = np.zeros(256, dtype=np.int64)
    node_weight[np.uint8(ord(","))] = 1
    node_weight[np.uint8(ord("("))] = 1
    node_weight[np.uint8(ord(";"))] = 1
    max_nodes = 1 + np.sum(node_weight[chars])

    # pre-allocate arrays using heuristic size
    ids = np.empty(max_nodes, dtype=dtype_id)
    ancestor_ids = np.empty(max_nodes, dtype=dtype_id)
    label_starts = np.zeros(max_nodes, dtype=np.int64)
    label_stops = np.zeros(max_nodes, dtype=np.int64)
    label_quoted = np.zeros(max_nodes, dtype=np.uint8)
    bl_starts = np.empty(max_nodes, dtype=np.int64)
    bl_stops = np.empty(max_nodes, dtype=np.int64)
    bl_node_ids = np.empty(max_nodes, dtype=np.int64)

    # character codes
    LPAREN = np.uint8(ord("("))
    RPAREN = np.uint8(ord(")"))
    COMMA = np.uint8(ord(","))
    COLON = np.uint8(ord(":"))
    SEMI = np.uint8(ord(";"))
    SQUOTE = np.uint8(ord("'"))
    SPACE = np.uint8(ord(" "))
    TAB = np.uint8(ord("\t"))
    NEWLINE = np.uint8(ord("\n"))
    CR = np.uint8(ord("\r"))
    LBRACKET = np.uint8(ord("["))
    RBRACKET = np.uint8(ord("]"))

    # translation tables for delimiter detection
    is_bl_term = np.zeros(256, dtype=np.uint8)
    is_bl_term[COMMA] = 1
    is_bl_term[RPAREN] = 1
    is_bl_term[SEMI] = 1
    is_bl_term[LBRACKET] = 1

    is_lbl_term = np.zeros(256, dtype=np.uint8)
    is_lbl_term[COLON] = 1
    is_lbl_term[COMMA] = 1
    is_lbl_term[SEMI] = 1
    is_lbl_term[RPAREN] = 1
    is_lbl_term[LBRACKET] = 1

    # create root node (id 0)
    ids[0] = 0
    ancestor_ids[0] = 0  # root is its own ancestor
    num_nodes = 1
    num_bls = 0

    cur = 0
    i = 0

    while i < n:
        c = chars[i]

        # go to new child: '(' — most frequent structural characters first
        if c == LPAREN:
            child_id = num_nodes
            ids[child_id] = child_id
            ancestor_ids[child_id] = cur
            num_nodes += 1
            cur = child_id

        # go to new sibling: ','
        elif c == COMMA:
            parent = ancestor_ids[cur]
            child_id = num_nodes
            ids[child_id] = child_id
            ancestor_ids[child_id] = parent
            num_nodes += 1
            cur = child_id
            # skip spaces after comma
            while i + 1 < n and chars[i + 1] == SPACE:
                i += 1

        # go to parent: ')'
        elif c == RPAREN:
            cur = ancestor_ids[cur]

        # edge length — parse immediately without re-entering loop
        elif c == COLON:
            i += 1
            ls_start = i
            while i < n and not is_bl_term[chars[i]]:
                i += 1
            bl_starts[num_bls] = ls_start
            bl_stops[num_bls] = i
            bl_node_ids[num_bls] = cur
            num_bls += 1
            i -= 1  # will be incremented at end of loop

        # quoted label — parse inline through closing quote
        elif c == SQUOTE:
            i += 1
            lbl_start = i
            while i < n:
                if chars[i] != SQUOTE:
                    i += 1  # ordinary label character
                elif i + 1 < n and chars[i + 1] == SQUOTE:
                    i += 2  # doubled '' is an escaped literal quote
                else:
                    break  # lone quote closes the label
            label_starts[cur] = lbl_start
            label_stops[cur] = i
            label_quoted[cur] = 1
            # i now points at closing quote; will be incremented at end

        # comment (square brackets)
        elif c == LBRACKET:
            count = 1
            i += 1
            while i < n and count > 0:
                count += chars[i] == LBRACKET
                count -= chars[i] == RBRACKET
                i += 1
            i -= 1  # will be incremented at end of loop

        # end of current tree; if more (non-whitespace) content follows, it
        # begins another tree, so allocate a fresh root for it inline
        # (forest / multi-tree support)
        elif c == SEMI:
            j = i + 1
            while j < n and (
                chars[j] == SPACE
                or chars[j] == TAB
                or chars[j] == NEWLINE
                or chars[j] == CR
            ):
                j += 1
            if j < n and chars[j] != SEMI:
                root_id = num_nodes
                ids[root_id] = root_id
                ancestor_ids[root_id] = root_id
                num_nodes += 1
                cur = root_id
            i = j - 1  # resume at the next content char (loop will +1)

        # unquoted label
        else:
            lbl_start = i
            while i < n and not is_lbl_term[chars[i]]:
                i += 1
            label_starts[cur] = lbl_start
            label_stops[cur] = i
            i -= 1  # will be incremented at end of loop

        i += 1

    return (
        ids,
        ancestor_ids,
        label_starts,
        label_stops,
        label_quoted,
        bl_starts,
        bl_stops,
        bl_node_ids,
        num_nodes,
        num_bls,
    )


def _parse_newick(
    newick: str,
    chars: np.ndarray,
    n: int,
    branch_length_dtype: type = float,
    dtype_id: np.dtype = np.dtype(np.int64),
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse newick string characters into parallel arrays.

    Implementation detail for `alifestd_from_newick`.

    Uses a two-phase approach: an inner numba JIT kernel handles structural
    parsing and records branch-length substring positions, then float
    conversion is performed externally via numpy.

    Adapted from
    https://github.com/niemasd/TreeSwift/blob/v1.1.45/treeswift/Tree.py#L1439

    Parameters
    ----------
    newick : str
        The newick string (used for substring extraction).
    chars : np.ndarray
        Array of uint8 character codes for the newick string.
    n : int
        Length of chars array.
    branch_length_dtype : type, default float
        Dtype for branch length parsing. Strings are converted via this
        dtype (providing validation), then stored as float64 with NaN for
        missing values. Callers handle nullable-int conversion.
    dtype_id : np.dtype, default np.dtype(np.int64)
        Numpy dtype for ``id`` and ``ancestor_id`` arrays.

    Returns
    -------
    tuple of np.ndarray
        (ids, ancestor_ids, branch_lengths, label_start_stops, label_quoted)
        where branch_lengths is float64 with NaN for missing values (callers
        should convert to nullable int if needed), label_start_stops has shape
        (num_nodes, 2) giving the start (inclusive) and stop (exclusive) index
        into `chars` for each node's label, and label_quoted is a bool array
        flagging whether each node's label was quoted.
    """
    (
        ids,
        ancestor_ids,
        label_starts,
        label_stops,
        label_quoted,
        bl_starts,
        bl_stops,
        bl_node_ids,
        num_nodes,
        num_bls,
    ) = _parse_newick_jit(chars, n, dtype_id)

    # trim to actual sizes
    ids = ids[:num_nodes]
    ancestor_ids = ancestor_ids[:num_nodes]

    # branch lengths: parse substrings using the requested dtype, then
    # store as float64 (to support NaN for missing values)
    np_dtype = np.dtype(branch_length_dtype)
    branch_lengths = np.full(num_nodes, np.nan, dtype=np.float64)
    if num_bls:
        # string extraction is sequential (variable-length slices);
        # numeric conversion is vectorized via np.array
        node_ids = bl_node_ids[:num_bls]
        starts = bl_starts[:num_bls]
        stops = bl_stops[:num_bls]
        bl_strings = [newick[starts[k] : stops[k]] for k in range(num_bls)]
        branch_lengths[node_ids] = np.array(bl_strings, dtype=np_dtype)

    # pack label start/stops into a 2D array
    label_start_stops = np.column_stack(
        (label_starts[:num_nodes], label_stops[:num_nodes])
    )

    return (
        ids,
        ancestor_ids,
        branch_lengths,
        label_start_stops,
        label_quoted[:num_nodes],
    )


def _extract_labels(
    newick: str,
    chars: np.ndarray,
    label_start_stops: np.ndarray,
    label_quoted: np.ndarray,
    replace_unquoted_table: dict,
) -> np.ndarray:
    """Extract taxon labels from newick string using index ranges.

    Implementation detail for `alifestd_from_newick`.

    The character mapping in ``replace_unquoted_table`` (a ``str.maketrans``
    table, possibly empty) is applied to *unquoted* labels only, leaving
    quoted labels verbatim. ``label_quoted`` flags which labels were quoted.
    """
    num_nodes = len(label_start_stops)
    labels = np.empty(num_nodes, dtype=object)
    for k, (start, stop) in enumerate(label_start_stops):
        label = newick[start:stop]
        if label_quoted[k]:
            # only quoted labels can contain escaped quotes; collapse the
            # doubled '' back to a single quote
            label = label.replace("''", "'")
        elif replace_unquoted_table:
            # substitutions apply to unquoted labels only
            label = label.translate(replace_unquoted_table)
        labels[k] = label
    return labels


@jit(nopython=True)
def _jit_build_label_buffer(
    chars: np.ndarray,
    label_starts: np.ndarray,
    label_stops: np.ndarray,
    num_nodes: int,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Build a flat byte buffer and offset array for label strings.

    Copies label bytes from `chars` into a contiguous buffer, suitable
    for zero-copy construction of an Arrow/Polars string array.

    Implementation detail for `alifestd_from_newick_polars`.

    Returns
    -------
    tuple of np.ndarray
        (data, offsets) where data is uint8 label bytes and offsets is
        int64 with shape (num_nodes + 1,) giving byte positions.
    """
    # first pass: compute total size
    total = np.int64(0)
    for i in range(num_nodes):
        total += label_stops[i] - label_starts[i]

    data = np.empty(total, dtype=np.uint8)
    offsets = np.empty(num_nodes + 1, dtype=np.int64)
    offsets[0] = 0
    pos = np.int64(0)
    for i in range(num_nodes):
        s = label_starts[i]
        e = label_stops[i]
        length = e - s
        for j in range(length):
            data[pos + j] = chars[s + j]
        pos += length
        offsets[i + 1] = pos

    return data, offsets


@jit(nopython=True)
def _jit_parse_branch_lengths(
    chars: np.ndarray,
    bl_starts: np.ndarray,
    bl_stops: np.ndarray,
    bl_node_ids: np.ndarray,
    num_nodes: int,
    num_bls: int,
) -> np.ndarray:
    """Parse branch length floats directly from character data.

    Avoids Python-level string extraction and numpy string-to-float
    conversion by delegating to ``jit_parse_float`` for each value.

    Implementation detail for `_parse_newick`.

    Returns
    -------
    np.ndarray
        Float64 array of length num_nodes, with NaN for missing values.
    """
    branch_lengths = np.full(num_nodes, np.nan, dtype=np.float64)
    for k in range(num_bls):
        branch_lengths[bl_node_ids[k]] = jit_parse_float(
            chars, bl_starts[k], bl_stops[k]
        )
    return branch_lengths


# Performance (as of 2026-03-15, 200k-node caterpillar tree, JIT-warmed):
#   with branch lengths: phyloframe ~0.4s vs treeswift ~1.4s (~0.3x)
#   without branch lengths: phyloframe ~0.3s vs treeswift ~0.8s (~0.4x)
def alifestd_from_newick(
    newick: str,
    *,
    allow_forest: typing.Optional[bool] = None,
    branch_length_dtype: type = float,
    create_ancestor_list: bool = False,
    dtype_id: typing.Optional[type] = np.int64,
    replace_unquoted: typing.Mapping[str, str] = types.MappingProxyType({}),
) -> pd.DataFrame:
    """Convert a Newick format string to a phylogeny dataframe.

    Parses a Newick tree string and returns a pandas DataFrame in alife
    standard format with columns: id, ancestor_id, taxon_label,
    origin_time_delta, and branch_length. Optionally includes
    ancestor_list.

    Parameters
    ----------
    newick : str
        A phylogeny in Newick format.
    allow_forest : bool or None, default None
        Policy for a Newick string holding multiple ``;``-terminated trees
        (a forest). ``None`` parses the forest but warns; ``True`` parses it
        silently; ``False`` raises ``ValueError`` unless there is a single
        tree.
    branch_length_dtype : type, default float
        Dtype for branch length values. Use ``int`` to get nullable integer
        columns (``pd.Int64Dtype``). Missing branch lengths will be ``pd.NA``
        for integer dtypes or ``NaN`` for float dtypes.
    create_ancestor_list : bool, default False
        If True, include an ``ancestor_list`` column in the result.
    dtype_id : type or None, default np.int64
        Numpy dtype for the ``id`` and ``ancestor_id`` columns. If None, the
        smallest signed integer dtype that can hold all node ids is chosen
        automatically based on the node count of the Newick string.
    replace_unquoted : Mapping[str, str], optional
        Character substitutions to apply to *unquoted* taxon labels only,
        leaving quoted labels verbatim. Keys must be single characters.
        Pass ``{"_": " "}`` to follow the strict Newick convention in which
        an unquoted underscore denotes a space.

    Returns
    -------
    pd.DataFrame
        Phylogeny dataframe in alife standard format.

    Notes
    -----
    By default, unquoted underscores in taxon labels are preserved literally;
    they are *not* converted to spaces. This diverges from the strict Newick
    convention (in which an unquoted ``_`` denotes a space), but matches the
    round-trip behavior of ``alifestd_as_newick_asexual``. Pass
    ``replace_unquoted={"_": " "}`` to follow the strict convention.

    See Also
    --------
    alifestd_from_newick_polars :
        Polars-based implementation.
    alifestd_as_newick_asexual :
        Inverse conversion, from alife standard to Newick format.
    """
    newick = newick.strip()
    if any(len(key) != 1 for key in replace_unquoted):
        raise ValueError("replace_unquoted keys must be single characters")
    replace_unquoted_table = str.maketrans(dict(replace_unquoted))
    if dtype_id is None:
        # the parser assigns one node id per '(' and per ',', plus one root
        # per tree (';'-terminated); size the dtype from that node-count
        # upper bound rather than commas alone to avoid overflow
        node_count = newick.count("(") + newick.count(",") + newick.count(";")
        resolved_dtype_id = np.min_scalar_type(-max(node_count, 1))
    else:
        resolved_dtype_id = np.dtype(dtype_id)

    if not newick:
        columns = {
            "id": pd.Series(dtype=resolved_dtype_id),
            "ancestor_id": pd.Series(dtype=resolved_dtype_id),
            "taxon_label": pd.Series(dtype=str),
            "origin_time_delta": pd.Series(dtype=float),
            "branch_length": pd.Series(dtype=float),
        }
        if create_ancestor_list:
            columns["ancestor_list"] = pd.Series(dtype=str)
        return pd.DataFrame(columns)

    chars = np.frombuffer(newick.encode("ascii"), dtype=np.uint8)
    n = len(chars)

    (
        ids,
        ancestor_ids,
        branch_lengths,
        label_start_stops,
        label_quoted,
    ) = _parse_newick(newick, chars, n, branch_length_dtype, resolved_dtype_id)

    if allow_forest is not True:
        num_roots = int(np.count_nonzero(ancestor_ids == ids))
        if num_roots > 1:
            if allow_forest is False:
                raise ValueError(
                    f"Newick string contains a forest of {num_roots} trees; "
                    "pass allow_forest=True to allow.",
                )
            warnings.warn(
                f"Newick string contains a forest of {num_roots} trees; pass "
                "allow_forest=True to silence this warning or "
                "allow_forest=False to require a single tree.",
            )

    labels = _extract_labels(
        newick,
        chars,
        label_start_stops,
        label_quoted,
        replace_unquoted_table,
    )

    # convert branch lengths to requested dtype with proper null handling
    np_dtype = np.dtype(branch_length_dtype)
    if np.issubdtype(np_dtype, np.integer):
        # use pandas nullable integer: NaN -> pd.NA
        bl_series = pd.array(
            np.where(np.isnan(branch_lengths), pd.NA, branch_lengths),
            dtype=pd.Int64Dtype(),
        )
        otd_series = bl_series.copy()
    else:
        bl_series = branch_lengths
        otd_series = branch_lengths.copy()

    phylogeny_df = pd.DataFrame(
        {
            "id": ids,
            "ancestor_id": ancestor_ids,
            "taxon_label": labels,
            "origin_time_delta": otd_series,
            "branch_length": bl_series,
        },
    )

    if create_ancestor_list:
        phylogeny_df["ancestor_list"] = alifestd_make_ancestor_list_col(
            phylogeny_df["id"],
            phylogeny_df["ancestor_id"],
        )

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()})

Convert Newick format phylogeny data to Alife standard format.

Note that this CLI entrypoint is experimental and may be subject to change.
"""


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=format_cli_description(_raw_description),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="Newick file to convert to Alife standard dataframe format.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Path to write Alife standard dataframe output to.",
    )
    parser.add_argument(
        "--output-engine",
        type=str,
        choices=["pandas", "polars"],
        default="pandas",
        help="DataFrame engine to use for writing the output file. Defaults to 'pandas'.",
    )
    parser.add_argument(
        "--branch-length-dtype",
        type=str,
        choices=["float", "int"],
        default="float",
        help="Dtype for branch length values. Defaults to 'float'.",
    )
    add_bool_arg(
        parser,
        "create-ancestor-list",
        default=False,
        help="Include an ancestor_list column in the output.",
    )
    add_bool_arg(
        parser,
        "allow-forest",
        default=None,
        help=(
            "Allow a multi-tree (forest) Newick string. Unset warns; "
            "--allow-forest is silent; --no-allow-forest requires a single "
            "tree."
        ),
    )
    parser.add_argument(
        "--replace-unquoted",
        type=str,
        default="{}",
        metavar="MAPPING",
        help=(
            "Mapping of single-character substitutions to apply to unquoted "
            "taxon labels only (quoted labels are left verbatim), given as a "
            "Python dict literal. "
            "Example: \"{'_': ' '}\" maps unquoted underscores to spaces."
        ),
    )
    parser.add_argument(
        "--output-kwarg",
        action="append",
        dest="output_kwargs",
        type=str,
        default=[],
        help=(
            "Additional keyword arguments to pass to output engine call. "
            "Provide as 'key=value'. "
            "Specify multiple kwargs by using this flag multiple times. "
            "Arguments will be evaluated as Python expressions. "
            "Example: 'index=False'"
        ),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=get_phyloframe_version(),
    )
    return parser


_dtype_lookup = {"float": float, "int": int}


if __name__ == "__main__":
    configure_prod_logging()

    parser = _create_parser()
    args = parser.parse_args()

    logging.info(f"reading Newick data from {args.input_file}...")
    newick_str = pathlib.Path(args.input_file).read_text()

    with log_context_duration(
        "phyloframe.legacy.alifestd_from_newick", logging.info
    ):
        logging.info("converting from Newick format...")
        phylogeny_df = alifestd_from_newick(
            newick_str,
            allow_forest=args.allow_forest,
            branch_length_dtype=_dtype_lookup[args.branch_length_dtype],
            create_ancestor_list=args.create_ancestor_list,
            replace_unquoted=ast.literal_eval(args.replace_unquoted),
        )

    output_ext = os.path.splitext(args.output_file)[1]
    output_kwargs = eval_kwargs(args.output_kwargs)

    logging.info(
        f"writing alife-standard {output_ext} phylogeny data to "
        f"{args.output_file}...",
    )
    if args.output_engine == "polars":
        phylogeny_df = pl.from_pandas(phylogeny_df)
        dispatch_writer = {
            ".csv": pl.DataFrame.write_csv,
            ".fea": pl.DataFrame.write_ipc,
            ".feather": pl.DataFrame.write_ipc,
            ".pqt": pl.DataFrame.write_parquet,
            ".parquet": pl.DataFrame.write_parquet,
        }
    elif args.output_engine == "pandas":
        if output_ext == ".csv":
            output_kwargs.setdefault("index", False)
        dispatch_writer = {
            ".csv": pd.DataFrame.to_csv,
            ".fea": pd.DataFrame.to_feather,
            ".feather": pd.DataFrame.to_feather,
            ".pqt": pd.DataFrame.to_parquet,
            ".parquet": pd.DataFrame.to_parquet,
        }
    else:
        raise ValueError(f"unsupported output engine: {args.output_engine!r}")

    dispatch_writer[output_ext](
        phylogeny_df,
        args.output_file,
        **output_kwargs,
    )

    logging.info("done!")

"""Benchmark: original vs optimized alifestd_prefix_roots_polars.

Measures peak memory and wall-clock time for varying DataFrame sizes.
"""
import gc
import numbers
import time
import tracemalloc
import typing
import warnings

import numpy as np
import opytional as opyt
import polars as pl


# ── Original implementation ──────────────────────────────────────────────
def alifestd_prefix_roots_original(
    phylogeny_df: pl.DataFrame,
    *,
    allow_id_reassign: bool = False,
    origin_time: typing.Optional[numbers.Real] = None,
) -> pl.DataFrame:
    if "origin_time_delta" in phylogeny_df:
        warnings.warn("alifestd_prefix_roots ignores origin_time_delta values")
    if origin_time is not None and "origin_time" not in phylogeny_df:
        raise ValueError(
            "origin_time specified but not present in phylogeny dataframe",
        )
    if "ancestor_list" in phylogeny_df:
        raise NotImplementedError
    if not allow_id_reassign:
        raise NotImplementedError
    if phylogeny_df.lazy().limit(1).collect().is_empty():
        raise NotImplementedError
    has_contiguous_ids = phylogeny_df.select(
        pl.col("id").diff() == 1
    ).to_series().all() and (phylogeny_df["id"].first() == 0)
    if not has_contiguous_ids:
        raise NotImplementedError

    phylogeny_df = phylogeny_df.drop("is_root", strict=False)

    eligible_roots = (
        phylogeny_df.lazy()
        .with_columns(
            is_eligible=(
                pl.col("origin_time") > origin_time
                if origin_time is not None
                else True
            ),
            is_root=pl.col("id") == pl.col("ancestor_id"),
        )
        .select(
            pl.col("is_eligible") & pl.col("is_root"),
        )
        .collect()
        .to_series()
    )

    prepended_roots = (
        phylogeny_df.lazy()
        .filter(
            eligible_roots,
        )
        .select(
            "id",
            "origin_time",
            "ancestor_id",
        )
        .collect()
    )

    if "origin_time" in prepended_roots:
        prepended_roots = prepended_roots.with_columns(
            origin_time=pl.lit(opyt.or_value(origin_time, 0))
        )

    phylogeny_df = phylogeny_df.with_columns(
        id=pl.col("id") + pl.lit(len(prepended_roots)),
        ancestor_id=pl.col("ancestor_id") + pl.lit(len(prepended_roots)),
    )
    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy().copy()
    ancestor_ids[prepended_roots["id"].to_numpy()] = np.arange(
        len(prepended_roots),
    )
    phylogeny_df = phylogeny_df.with_columns(
        ancestor_id=pl.Series(ancestor_ids),
    )

    prepended_roots = prepended_roots.with_columns(
        id=pl.int_range(len(prepended_roots)),
        ancestor_id=pl.int_range(len(prepended_roots)),
    ).cast(
        {
            k: v
            for k, v in phylogeny_df.collect_schema().items()
            if k in prepended_roots.collect_schema()
        },
    )

    gather_indices = np.empty(
        len(phylogeny_df) + len(prepended_roots), dtype=np.int64
    )
    gather_indices[: len(prepended_roots)] = np.arange(len(prepended_roots))
    gather_indices[len(prepended_roots) :] = np.arange(len(phylogeny_df))

    return (
        phylogeny_df.lazy()
        .select(pl.all().gather(gather_indices))
        .with_row_index()
        .with_columns(
            pl.when(pl.col("index") < len(prepended_roots))
            .then(None)
            .otherwise(pl.all())
            .name.keep()
        )
        .drop("index")
        .update(prepended_roots.lazy())
        .collect()
    )


# ── Optimized implementation ────────────────────────────────────────────
def alifestd_prefix_roots_optimized(
    phylogeny_df: pl.DataFrame,
    *,
    allow_id_reassign: bool = False,
    origin_time: typing.Optional[numbers.Real] = None,
) -> pl.DataFrame:
    if "origin_time_delta" in phylogeny_df:
        warnings.warn("alifestd_prefix_roots ignores origin_time_delta values")
    if origin_time is not None and "origin_time" not in phylogeny_df:
        raise ValueError(
            "origin_time specified but not present in phylogeny dataframe",
        )
    if "ancestor_list" in phylogeny_df:
        raise NotImplementedError
    if not allow_id_reassign:
        raise NotImplementedError
    if phylogeny_df.lazy().limit(1).collect().is_empty():
        raise NotImplementedError
    has_contiguous_ids = phylogeny_df.select(
        pl.col("id").diff() == 1
    ).to_series().all() and (phylogeny_df["id"].first() == 0)
    if not has_contiguous_ids:
        raise NotImplementedError

    phylogeny_df = phylogeny_df.drop("is_root", strict=False)

    # Identify eligible roots purely in polars (no materialized boolean Series)
    is_root_expr = pl.col("id") == pl.col("ancestor_id")
    is_eligible_expr = (
        (pl.col("origin_time") > origin_time) & is_root_expr
        if origin_time is not None
        else is_root_expr
    )

    # Build prepended roots from eligible roots — only 3 lightweight columns
    prepended_roots = (
        phylogeny_df.lazy()
        .filter(is_eligible_expr)
        .select("id", "ancestor_id", "origin_time")
        .collect()
    )
    num_prepended = len(prepended_roots)
    if num_prepended == 0:
        return phylogeny_df

    # Map original root ids → new prepended root ids, all in polars
    # Build a small lookup: original_root_row_index -> new_ancestor_id
    root_original_ids = prepended_roots["id"]  # zero-copy reference
    new_ancestor_map = pl.DataFrame(
        {
            "row_idx": root_original_ids,
            "new_ancestor_id": pl.int_range(num_prepended, eager=True).cast(
                root_original_ids.dtype
            ),
        },
    )

    # Shift ids and remap ancestor_ids for eligible roots, all in one lazy pass
    phylogeny_df = (
        phylogeny_df.lazy()
        .with_columns(
            id=pl.col("id") + num_prepended,
            ancestor_id=pl.col("ancestor_id") + num_prepended,
        )
        .with_row_index("row_idx")
        .join(new_ancestor_map.lazy(), on="row_idx", how="left")
        .with_columns(
            ancestor_id=pl.coalesce("new_ancestor_id", "ancestor_id"),
        )
        .drop("row_idx", "new_ancestor_id")
        .collect()
    )

    # Build full prepended-root rows with correct schema
    prepended_root_rows = prepended_roots.with_columns(
        id=pl.int_range(num_prepended),
        ancestor_id=pl.int_range(num_prepended),
        origin_time=pl.lit(opyt.or_value(origin_time, 0)),
    ).cast(
        {
            k: v
            for k, v in phylogeny_df.collect_schema().items()
            if k in prepended_roots.collect_schema()
        },
    )

    # Align columns: add missing columns as null to prepended rows, then concat
    for col_name in phylogeny_df.columns:
        if col_name not in prepended_root_rows.columns:
            prepended_root_rows = prepended_root_rows.with_columns(
                pl.lit(None).cast(phylogeny_df[col_name].dtype).alias(col_name)
            )
    prepended_root_rows = prepended_root_rows.select(phylogeny_df.columns)

    return pl.concat([prepended_root_rows, phylogeny_df])


# ── Benchmark harness ────────────────────────────────────────────────────
def make_test_df(n_rows: int, root_fraction: float = 0.01) -> pl.DataFrame:
    """Create a synthetic phylogeny DataFrame with n_rows rows."""
    rng = np.random.default_rng(42)
    n_roots = max(1, int(n_rows * root_fraction))
    root_indices = rng.choice(n_rows, size=n_roots, replace=False)

    ids = np.arange(n_rows, dtype=np.int64)
    ancestor_ids = rng.integers(0, n_rows, size=n_rows).astype(np.int64)
    ancestor_ids[root_indices] = ids[root_indices]  # roots: ancestor == self
    origin_times = rng.uniform(0, 100, size=n_rows)

    # Add some extra payload columns to make memory pressure realistic
    return pl.DataFrame(
        {
            "id": ids,
            "ancestor_id": ancestor_ids,
            "origin_time": origin_times,
            "taxon_label": [f"taxon_{i}" for i in range(n_rows)],
            "payload_a": rng.standard_normal(n_rows),
            "payload_b": rng.standard_normal(n_rows),
        }
    )


def bench(fn, df, **kwargs):
    """Run fn(df, **kwargs), return (result, peak_mem_bytes, elapsed_s)."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(df, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak, elapsed


def verify_equivalence(res_orig, res_opt):
    """Check that the two results are schema- and value-equivalent."""
    assert res_orig.columns == res_opt.columns, (
        f"Column mismatch: {res_orig.columns} vs {res_opt.columns}"
    )
    assert res_orig.shape == res_opt.shape, (
        f"Shape mismatch: {res_orig.shape} vs {res_opt.shape}"
    )
    for col in res_orig.columns:
        orig_s = res_orig[col]
        opt_s = res_opt[col]
        if orig_s.dtype.is_float():
            mismatches = (
                (orig_s.is_null() != opt_s.is_null())
                | (
                    orig_s.is_not_null()
                    & opt_s.is_not_null()
                    & ((orig_s - opt_s).abs() > 1e-10)
                )
            )
        else:
            mismatches = (
                (orig_s.is_null() != opt_s.is_null())
                | (
                    orig_s.is_not_null()
                    & opt_s.is_not_null()
                    & (orig_s != opt_s)
                )
            )
        n_bad = mismatches.sum()
        assert n_bad == 0, (
            f"Column '{col}': {n_bad} mismatched values"
        )


if __name__ == "__main__":
    sizes = [10_000, 100_000, 500_000, 1_000_000]

    print(f"{'N':>12} | {'Impl':>12} | {'Peak Mem (MB)':>14} | {'Time (s)':>10}")
    print("-" * 60)

    for n in sizes:
        df = make_test_df(n)

        res_orig, mem_orig, t_orig = bench(
            alifestd_prefix_roots_original,
            df,
            allow_id_reassign=True,
            origin_time=50.0,
        )
        res_opt, mem_opt, t_opt = bench(
            alifestd_prefix_roots_optimized,
            df,
            allow_id_reassign=True,
            origin_time=50.0,
        )

        verify_equivalence(res_orig, res_opt)

        print(
            f"{n:>12,} | {'original':>12} | {mem_orig / 1e6:>14.2f} | {t_orig:>10.4f}"
        )
        print(
            f"{n:>12,} | {'optimized':>12} | {mem_opt / 1e6:>14.2f} | {t_opt:>10.4f}"
        )
        ratio_mem = mem_orig / max(mem_opt, 1)
        ratio_time = t_orig / max(t_opt, 1e-9)
        print(
            f"{'':>12} | {'speedup':>12} | {ratio_mem:>13.1f}x | {ratio_time:>9.1f}x"
        )
        print("-" * 60)

        del df, res_orig, res_opt
        gc.collect()

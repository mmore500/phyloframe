# Bug Brief: stale `schema_names` causes `is_root` column loss in `alifestd_as_newick_polars`

## Summary

`alifestd_as_newick_polars` uses a stale `schema_names` snapshot to decide whether to add an `is_root` column.
When a precomputed `num_children` column is present (as in the CSR benchmark variant), `is_root` is silently present in the input schema because `alifestd_mark_num_children_polars` adds it as a side effect.
A later `.select()` drops `is_root`, but the stale schema check thinks it still exists and skips re-adding it — causing a column-not-found error.

## Root Cause

In `_alifestd_as_newick_polars.py`:

1. **Line 73–74** — `schema_names` is captured once from the input dataframe.
2. **Line 163** — A `.select()` keeps only `["id", "ancestor_id", "node_depth", "origin_time_deltas"]`, dropping every other column including `is_root`.
3. **Line 197** — The guard `if "is_root" not in schema_names` checks the *original* schema (which includes `is_root` from the `num_children` marking side effect), finds it present, and **skips** re-adding it.
4. **Line 226** — `pl.col("is_root")` fails because the column was dropped and never restored.

## Trigger Condition

The bug triggers when:
- The input dataframe already has `is_root` (e.g., because `alifestd_mark_num_children_polars` added it as a side effect during `_mark_after_load`), **and**
- `save_newick` / `alifestd_as_newick_polars` is called on that dataframe.

Without a precomputed `num_children`, `is_root` is absent from the input schema, so the line-197 guard correctly adds it after the select, and everything works.

## Minimal Reproducible Example

```python
import polars as pl
from phyloframe.legacy import (
    alifestd_as_newick_polars,
    alifestd_from_newick_polars,
    alifestd_mark_num_children_polars,
)

newick = "((A,B),(C,D));"
df = alifestd_from_newick_polars(newick)

# This works — no precomputed num_children, so is_root is absent
# from the input schema and gets correctly added after the select.
print(alifestd_as_newick_polars(df))  # ok

# Now precompute num_children (adds is_root as side effect)
df_with_nc = alifestd_mark_num_children_polars(df)
assert "is_root" in df_with_nc.columns  # side-effect column

# This fails — is_root is in the stale schema_names but gets
# dropped by the .select(), then is never re-added.
print(alifestd_as_newick_polars(df_with_nc))  # raises ColumnNotFoundError
```

## Possible Fixes

1. **Refresh `schema_names` after the `.select()` at line 172** so subsequent column-existence checks reflect the actual state of the dataframe.

2. **Include `is_root` in `select_cols`** when it is already present, so it survives the select.

3. **Always re-derive `is_root`** after the select (unconditionally), since it's cheap (`ancestor_id == id`) and removes the dependency on schema state.

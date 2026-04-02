=====================
Performance (Legacy)
=====================

.. note::

   This documentation covers the **legacy** API (from phyloframe import legacy).
   The legacy API is stable and will continue to be maintained for backward
   compatibility.
   A redesigned API will accompany phyloframe v1.0.0.


This guide covers techniques for getting the best performance out of phyloframe: JIT compilation, Polars, working format, and writing custom high-performance operations.

Working Format
==============

Converting to working format is the single most impactful optimization.
Many operations have fast paths that activate only when the data is topologically sorted with contiguous IDs:

.. code-block:: python

   from phyloframe import legacy as pfl

   df = pfl.alifestd_from_newick("((A:1,B:2):3,(C:4,D:5):6);")
   df = pfl.alifestd_to_working_format(df)

Convert once, then chain operations.
In working format, ``ancestor_id`` values can be used directly as array indices.

Mutation for Pipeline Performance
=================================

Use ``mutate=True`` in Pandas-based pipelines to avoid unnecessary copies:

.. code-block:: python

   df = (
       df.pipe(pfl.alifestd_to_working_format, mutate=True)
       .pipe(pfl.alifestd_mark_leaves, mutate=True)
       .pipe(pfl.alifestd_mark_node_depth_asexual, mutate=True)
       .pipe(pfl.alifestd_mark_num_descendants_asexual, mutate=True)
   )

JIT Compilation with Numba
===========================

Install with JIT support:

.. code-block:: bash

   python3 -m pip install "phyloframe[jit]==0.6.1"

Many phyloframe operations automatically use JIT-compiled fast paths when Numba is available.
No code changes are needed --- just install the ``[jit]`` extra.

Writing Custom JIT Functions
-----------------------------

Use phyloframe's ``jit`` utility for custom native-speed operations:

.. code-block:: python

   import numpy as np
   from phyloframe._auxlib import jit
   from phyloframe import legacy as pfl

   @jit(nopython=True, cache=False)
   def count_deep_nodes(
       ancestor_ids: np.ndarray, threshold: int,
   ) -> int:
       """Count nodes deeper than `threshold`.

       Requires contiguous IDs and topological sorting.
       """
       n = len(ancestor_ids)
       depths = np.zeros(n, dtype=np.int64)
       count = 0
       for i in range(n):
           if ancestor_ids[i] != i:  # not a root
               depths[i] = depths[ancestor_ids[i]] + 1
           if depths[i] > threshold:
               count += 1
       return count

   # Use with phyloframe data in working format
   df = pfl.alifestd_make_balanced_bifurcating(depth=10)
   df = pfl.alifestd_to_working_format(df)
   n_deep = count_deep_nodes(df["ancestor_id"].values, threshold=5)

Note that in practice, this particular operation is more idiomatically done by marking node depths and then filtering with a DataFrame operation:

.. code-block:: python

   df = pfl.alifestd_mark_node_depth_asexual(df)
   n_deep = (df["node_depth"] > 5).sum()

The JIT approach is most useful for custom algorithms that cannot be expressed as combinations of existing phyloframe operations.

The ``jit`` decorator:

- Uses Numba when available for native-speed compilation.
- Falls back to pure Python if Numba is not installed.
- Automatically disables compilation during coverage runs.
- Enables Numba's function caching by default (pass ``cache=False`` to disable, e.g., for short-lived or dynamically defined functions).

Common Pattern: Array-based Algorithms
--------------------------------------

Working format enables a common pattern: extract NumPy arrays from the DataFrame, process with JIT-compiled functions, and store results back:

.. code-block:: python

   @jit(nopython=True, cache=False)
   def compute_subtree_sizes(ancestor_ids: np.ndarray) -> np.ndarray:
       """Count nodes in each subtree (including self)."""
       n = len(ancestor_ids)
       sizes = np.ones(n, dtype=np.int64)
       # Reverse iteration is postorder for topologically sorted data
       for i in range(n - 1, -1, -1):
           parent = ancestor_ids[i]
           if parent != i:
               sizes[parent] += sizes[i]
       return sizes

   df["subtree_size"] = compute_subtree_sizes(df["ancestor_id"].values)

Polars for Large-scale Data
============================

Polars can outperform Pandas for large phylogenies due to multithreading, lazy evaluation, and memory-efficient representation.

.. note::

   Set the ``POLARS_MAX_THREADS`` environment variable to control the number of threads Polars uses.
   Set ``POLARS_ENGINE_AFFINITY`` or use ``pl.Config.set_engine_affinity()`` to control the `query engine optimization strategy <https://docs.pola.rs/api/python/dev/reference/api/polars.Config.set_engine_affinity.html>`_.
   Polars streaming mode can also be enabled for larger-than-memory datasets via ``pl.Config.set_streaming_chunk_size()``.

.. code-block:: python

   import polars as pl
   from phyloframe import legacy as pfl

   # Direct Polars construction
   df_pl = pfl.alifestd_from_newick_polars("((A,B),(C,D));")

   # Use _polars suffixed functions for Polars DataFrames
   df_pl = pfl.alifestd_mark_leaves_polars(df_pl)

Polars restrictions:

- Asexual phylogenies only.
- Requires data in working format (topological sortedness and contiguous IDs).
- No ``mutate`` parameter (Polars DataFrames are immutable).

CLI Performance
===============

For CLI pipelines on Parquet data, prefer Polars entrypoints to eliminate conversion overhead:

.. code-block:: bash

   # Slower: converts Parquet -> Pandas -> process -> Pandas -> Parquet
   python3 -m phyloframe.legacy._alifestd_mark_leaves "output.pqt" < "input.pqt"

   # Faster: native Polars processing
   python3 -m phyloframe.legacy._alifestd_mark_leaves_polars \
       "output.pqt" < "input.pqt"

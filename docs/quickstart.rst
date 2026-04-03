==========
Quickstart
==========

This guide walks through the basics of phyloframe: creating phylogenies, inspecting tree structure, marking properties, transforming trees, and exporting results.

Installation
============

Install phyloframe with JIT acceleration (recommended):

.. code-block:: bash

   python3 -m pip install "phyloframe[jit]==0.6.1"

Omit ``[jit]`` if you do not need Numba-based just-in-time compilation:

.. code-block:: bash

   python3 -m pip install "phyloframe==0.6.1"

Import Convention
=================

.. code-block:: python

   from phyloframe import legacy as pfl

The ``legacy`` module contains all current phyloframe operations.
As phyloframe evolves, ``legacy`` will continue to be maintained for backward compatibility while new API designs are developed.

The Official Alife Standard Format
===================================

Phyloframe represents phylogenies as DataFrames in the **alife standard format**.
Each row represents an organism (or taxon).

Required Columns
----------------

``id`` : int
    Unique, non-negative identifier for this organism.

``ancestor_list`` : str
    JSON-encoded list of ancestor IDs.
    For asexual phylogenies, this is a single-element list like ``"[0]"``.
    Root nodes use ``"[None]"``, ``"[none]"``, or ``"[]"``.

.. note::

   The ambiguity of root representations (``"[None]"`` vs ``"[none]"`` vs ``"[]"``) is a known defect in the current alife data standard.
   The use of ``none`` also deviates from valid JSON.
   The string encoding additionally incurs parsing overhead on every access.
   Phyloframe's ``ancestor_id`` column avoids these issues.

Optional Columns (Official Standard)
-------------------------------------

``origin_time`` : numeric
    Time at which this organism originated.

``destruction_time`` : numeric
    Time at which this organism was destroyed or went extinct.

``taxon_label`` : str
    Human-readable label or species name for this organism.

See the `alife data standards specification <https://alife-data-standards.github.io/alife-data-standards/phylogeny.html>`_ for the full list of official optional columns.

Unofficial Extension: ``ancestor_id``
--------------------------------------

``ancestor_id`` : int
    Direct ancestor ID for asexual phylogenies.
    This is an optimized integer representation of ``ancestor_list`` that avoids repeated string parsing.
    Root nodes store their own ID as ``ancestor_id``.

All phyloframe operations on asexual trees support ``ancestor_id`` in place of ``ancestor_list``.
Using ``ancestor_id`` is recommended unless interoperability with other alife standard ecosystem tools is needed.
Use ``alifestd_try_add_ancestor_list_col`` to generate ``ancestor_list`` on demand when required:

.. code-block:: python

   df = pfl.alifestd_try_add_ancestor_list_col(df)

Additional user-defined columns (e.g., trait data, fitness values) can be freely added --- the DataFrame is yours to extend.

Representing Roots
------------------

Root nodes have no ancestor.
In ``ancestor_list``, this is represented as ``"[None]"``, ``"[none]"``, or ``"[]"``.
In ``ancestor_id``, roots store their own ID (i.e., ``ancestor_id == id``).

Example
-------

.. code-block:: python

   import pandas as pd

   # A simple three-node chain: root -> internal -> leaf
   phylogeny_df = pd.DataFrame({
       "id": [0, 1, 2],
       "ancestor_list": ["[None]", "[0]", "[1]"],
   })

This represents::

   0 (root)
   +-- 1 (internal)
       +-- 2 (leaf)

Asexual vs. Sexual Phylogenies
------------------------------

**Asexual** phylogenies have at most one ancestor per organism (i.e., single-element ``ancestor_list``).
Most phyloframe operations target asexual phylogenies, where the ``ancestor_id`` column enables fast integer-based lookups.

**Sexual** phylogenies allow multiple ancestors (e.g., ``"[3, 7]"``).
Sexual phylogeny support is limited to operations that work with ``ancestor_list`` directly (primarily in Pandas).

.. code-block:: python

   # Check phylogeny type
   pfl.alifestd_is_asexual(phylogeny_df)  # True
   pfl.alifestd_is_sexual(phylogeny_df)  # False

Creating Phylogenies
====================

From Scratch
------------

.. code-block:: python

   # Empty phylogeny
   empty_df = pfl.alifestd_make_empty()

   # Empty phylogeny with ancestor_id column
   empty_df = pfl.alifestd_make_empty(ancestor_id=True)

Synthetic Trees
---------------

.. code-block:: python

   # Balanced bifurcating tree
   #   depth=1 -> 1 node (root only)
   #   depth=3 -> 7 nodes (4 leaves)
   #   depth=n -> 2^n - 1 nodes, 2^(n-1) leaves
   balanced_df = pfl.alifestd_make_balanced_bifurcating(depth=3)

   # Comb (caterpillar) tree
   comb_df = pfl.alifestd_make_comb(n_leaves=10)

From Newick Format
------------------

.. code-block:: python

   # Parse a Newick string
   df = pfl.alifestd_from_newick("((A:1,B:2):3,(C:4,D:5):6);")

   # The result includes columns: id, ancestor_id, taxon_label,
   # origin_time_delta, and branch_length
   print(df.columns.tolist())

Working Format
==============

Many phyloframe operations run fastest when the DataFrame is in **working format**:

1. **Topologically sorted** --- ancestors appear before descendants.
2. **Contiguous IDs** --- each organism's ``id`` equals its row number.
3. **``ancestor_id`` column** --- integer ancestor reference (asexual only).

Convert to working format once, then chain operations:

.. code-block:: python

   df = pfl.alifestd_make_balanced_bifurcating(depth=3)
   df = pfl.alifestd_to_working_format(df)

   # Verify properties
   assert pfl.alifestd_is_topologically_sorted(df)
   assert pfl.alifestd_has_contiguous_ids(df)

Marking Properties
==================

"Mark" functions add computed columns to a phylogeny DataFrame.
The original data is preserved; a new column is appended.

.. code-block:: python

   df = pfl.alifestd_pipe_unary_ops(
       pfl.alifestd_from_newick("((A,B),(C,D));"),
       pfl.alifestd_mark_leaves,  # leaf detection
       pfl.alifestd_mark_node_depth_asexual,  # depth from root
       pfl.alifestd_mark_num_descendants_asexual,  # descendant count
       pfl.alifestd_mark_num_children_asexual,  # direct children count
       pfl.alifestd_mark_roots,  # root detection
   )

   print(df[["id", "ancestor_id", "is_leaf", "node_depth",
             "num_descendants", "num_children", "is_root"]])

Custom Column Names
-------------------

All mark functions accept a ``mark_as`` parameter to customize the output column name:

.. code-block:: python

   df = pfl.alifestd_mark_leaves(df, mark_as="is_tip")
   df = pfl.alifestd_mark_node_depth_asexual(df, mark_as="depth")

Counting and Querying
=====================

.. code-block:: python

   df = pfl.alifestd_from_newick("((A,B),(C,D));")

   pfl.alifestd_count_leaf_nodes(df)  # 4
   pfl.alifestd_count_inner_nodes(df)  # 3
   pfl.alifestd_count_root_nodes(df)  # 1

   pfl.alifestd_is_asexual(df)  # True
   pfl.alifestd_is_topologically_sorted(df)  # True/False
   pfl.alifestd_has_contiguous_ids(df)  # True/False

   # Validate format compliance
   pfl.alifestd_validate(df)  # True

Tree Transformations
====================

.. code-block:: python

   df = pfl.alifestd_pipe_unary_ops(
       pfl.alifestd_from_newick("((A,B),(C,D));"),
       pfl.alifestd_collapse_unifurcations,  # remove single-child nodes
       pfl.alifestd_splay_polytomies,  # expand polytomies into bifurcations
       pfl.alifestd_add_global_root,  # add synthetic root above all roots
       pfl.alifestd_join_roots,  # join multiple roots to oldest root
   )

Composed Example: Downsampling with Combined Masks
===================================================

A common workflow: select tips using multiple sampling criteria, combine them with boolean OR, and prune extinct lineages.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from phyloframe import legacy as pfl

   # Create a tree with origin times computed from branch length deltas
   df = pfl.alifestd_from_newick(
       "((A:1,B:2):3,(C:4,(D:5,E:6):7):8);",
   )
   ancestor_ids = df["ancestor_id"].values
   deltas = df["origin_time_delta"].fillna(0).values
   origin_time = np.zeros(len(df))
   for i in range(len(df)):
       parent = ancestor_ids[i]
       if parent != i:
           origin_time[i] = origin_time[parent] + deltas[i]
   df["origin_time"] = origin_time

   # Strategy 1: keep the most recent tips (canopy sampling)
   df = pfl.alifestd_mark_sample_tips_canopy_asexual(
       df, n_sample=2, mark_as="keep_canopy",
   )

   # Strategy 2: keep tips closest to a focal lineage
   df = pfl.alifestd_mark_sample_tips_lineage_asexual(
       df, n_sample=2, mark_as="keep_lineage",
   )

   # Combine masks with boolean OR --- keep tips matching either criterion
   df["extant"] = df["keep_canopy"] | df["keep_lineage"]

   # Prune lineages without any extant descendants
   pruned_df = pfl.alifestd_prune_extinct_lineages_asexual(df)
   print(pruned_df[["id", "ancestor_id"]])

The ``alifestd_mark_sample_tips_*`` functions add boolean columns indicating which tips to retain.
Combining masks with ``|`` (OR), ``&`` (AND), or ``~`` (NOT) gives full control over tip selection.
The ``alifestd_prune_extinct_lineages_asexual`` function then removes any lineages that have no descendants marked as extant via the ``"extant"`` column (configurable with the ``criterion`` parameter).

Newick I/O
==========

.. code-block:: python

   # Parse Newick
   df = pfl.alifestd_from_newick("((A:1,B:2):3,(C:4,D:5):6);")

   # Export to Newick
   newick_str = pfl.alifestd_as_newick_asexual(df)

   # Use taxon labels from a column
   newick_str = pfl.alifestd_as_newick_asexual(df, taxon_label="taxon_label")

CSV and Parquet I/O
-------------------

Because phyloframe uses standard DataFrames, loading and saving is trivial:

.. code-block:: python

   import pandas as pd

   # CSV round-trip
   df.to_csv("phylogeny.csv", index=False)
   df = pd.read_csv("phylogeny.csv")

   # Parquet round-trip (recommended for large trees)
   df.to_parquet("phylogeny.pqt")
   df = pd.read_parquet("phylogeny.pqt")

   # Polars Parquet
   import polars as pl
   df_polars = pl.read_parquet("phylogeny.pqt")

Mutation Semantics
==================

By default, operations return a new DataFrame without modifying the input:

.. code-block:: python

   original = df.copy()
   result = pfl.alifestd_mark_leaves(df)
   assert original.equals(df)  # input unchanged

Set ``mutate=True`` to allow in-place modification for better performance in pipelines.
Even with ``mutate=True``, always use the return value:

.. code-block:: python

   # Faster: allows reuse of input memory
   df = pfl.alifestd_mark_leaves(df, mutate=True)
   df = pfl.alifestd_mark_node_depth_asexual(df, mutate=True)

Piping Operations
=================

Pandas provides ``DataFrame.pipe()`` for chaining operations idiomatically:

.. code-block:: python

   result = (
       df.pipe(pfl.alifestd_collapse_unifurcations)
       .pipe(pfl.alifestd_mark_leaves)
       .pipe(pfl.alifestd_mark_node_depth_asexual)
   )

Polars DataFrames also support ``.pipe()``:

.. code-block:: python

   import polars as pl

   df_pl = pfl.alifestd_from_newick_polars("((A,B),(C,D));")
   result_pl = (
       df_pl.pipe(pfl.alifestd_mark_leaves_polars)
   )

Alternatively, ``alifestd_pipe_unary_ops`` accepts multiple operations:

.. code-block:: python

   result = pfl.alifestd_pipe_unary_ops(
       df,
       pfl.alifestd_collapse_unifurcations,
       pfl.alifestd_mark_leaves,
       lambda df: pfl.alifestd_mark_node_depth_asexual(df, mark_as="depth"),
   )

Use ``tqdm`` for progress feedback on long pipelines:

.. code-block:: python

   from tqdm import tqdm

   result = pfl.alifestd_pipe_unary_ops(
       df,
       pfl.alifestd_collapse_unifurcations,
       pfl.alifestd_mark_leaves,
       progress_wrap=tqdm,
   )

Next Steps
==========

- :doc:`legacy-guides/concepts` --- Data format, tree data structures, and design decisions
- :doc:`legacy-guides/tree_creation` --- Synthetic trees, parsing, and construction
- :doc:`legacy-guides/tree_properties` --- Marking, counting, and metrics
- :doc:`legacy-guides/tree_manipulation` --- Transformations, pruning, and downsampling
- :doc:`legacy-guides/traversals` --- Tree traversal and supplemental data structures
- :doc:`legacy-guides/io` --- Newick, CSV, and Parquet I/O
- :doc:`legacy-guides/cli` --- Command-line interface and pipe operations
- :doc:`legacy-guides/performance` --- JIT compilation, Polars, and optimization
- :doc:`api` --- Full API reference

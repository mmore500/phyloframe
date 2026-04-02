====================================
Concepts and Data Structures (Legacy)
====================================

.. note::

   This documentation covers the **legacy** API (from phyloframe import legacy).
   The legacy API is stable and will continue to be maintained for backward
   compatibility.
   A redesigned API will accompany phyloframe v1.0.0.


This guide covers the core concepts behind phyloframe's data model:
the alife standard format, supplemental tree data structures, and the
relationship between Pandas and Polars implementations.

The Alife Standard Format
=========================

Phyloframe stores phylogenies as DataFrames in the `alife standard format
<https://alife-data-standards.github.io/alife-data-standards/phylogeny.html>`_,
originally developed for the Artificial Life community.
Each row represents a single organism or taxon.

Core Columns
------------

``id`` : int
    A unique, non-negative integer identifying this organism.
    In **working format**, IDs are contiguous and equal to row indices
    (i.e., ``id == row_number``).

``ancestor_list`` : str
    A JSON-encoded list of ancestor IDs.
    For asexual phylogenies: ``"[42]"`` (single ancestor) or ``"[None]"``
    (root).
    For sexual phylogenies: ``"[3, 7]"`` (multiple ancestors).

``ancestor_id`` : int
    An optimized representation for asexual phylogenies.
    Stores the single ancestor's ID directly as an integer, avoiding
    repeated string parsing.
    Root nodes store their own ID: ``ancestor_id == id``.

.. note::

   The ``ancestor_list`` column is part of the original alife data standard.
   The ``ancestor_id`` column is an unofficial extension introduced by
   phyloframe for efficient asexual phylogeny operations.
   Use ``alifestd_try_add_ancestor_id_col`` to add it automatically.

.. note::

   The alife data standard specifies ``ancestor_list`` as a string-encoded
   JSON list.
   A known defect in the standard is the ambiguity of empty-list
   representations: ``"[None]"``, ``"[none]"``, and ``"[]"`` are all used to
   denote roots.
   The string encoding also incurs parsing overhead on every access.
   Phyloframe's ``ancestor_id`` column avoids both issues and is the
   recommended representation for asexual phylogenies.

Root Representation
-------------------

Roots are organisms with no ancestor.

- ``ancestor_list``: ``"[None]"``, ``"[none]"``, or ``"[]"``
- ``ancestor_id``: equal to the organism's own ``id``

.. code-block:: python

   import pandas as pd

   # Root at id=0, child at id=1
   df = pd.DataFrame({
       "id": [0, 1],
       "ancestor_list": ["[None]", "[0]"],
   })

Working Format
==============

Many operations run fastest when the DataFrame is in **working format**,
satisfying three properties:

1. **Topologically sorted** --- every ancestor appears in a row before
   its descendants.
   This means that when iterating through rows in order, you will always
   encounter a node's parent before the node itself.
   This property enables single-pass algorithms that process the tree
   from root to leaves (or vice versa) by simply iterating through the
   array.
2. **Contiguous IDs** --- ``id`` values are ``0, 1, 2, ...`` matching
   row indices.
3. **``ancestor_id`` column present** --- enables direct integer indexing
   instead of string parsing.

.. note::

   Polars implementations require data in working format.

Convert to working format with a single call:

.. code-block:: python

   from phyloframe import legacy as pfl

   df = pfl.alifestd_to_working_format(df)

This applies ``alifestd_try_add_ancestor_id_col``,
``alifestd_topological_sort``, and ``alifestd_assign_contiguous_ids``
as needed.

In working format, ``ancestor_id`` values can be used directly as array
indices.
This enables efficient NumPy and JIT-compiled operations:

.. code-block:: python

   import numpy as np

   ancestor_ids = df["ancestor_id"].values
   # ancestor_ids[i] gives the row index of node i's parent
   # For root nodes, ancestor_ids[i] == i

Types of Phylogeny Data
========================

Phyloframe focuses on asexual phylogenies.
Partial support for sexual phylogenies is provided for compatibility with
the alife data standard.

**Asexual** phylogenies have at most one ancestor per organism.
The ``ancestor_id`` column can represent the tree structure efficiently.
Most phyloframe operations (especially those with ``_asexual`` suffix) target
this mode.

**Sexual** phylogenies (pedigrees) allow multiple ancestors per organism.
In Pandas, these have fewer optimized operations available.

.. code-block:: python

   # Check mode
   pfl.alifestd_is_asexual(df)   # True if all entries have <= 1 ancestor
   pfl.alifestd_is_sexual(df)    # True if any entry has > 1 ancestor

.. note::

   Polars implementations exclusively support asexual phylogenies.

Function Naming Conventions
===========================

Phyloframe function names follow consistent suffix patterns:

- **No suffix** (e.g., ``alifestd_mark_leaves``) --- works with both asexual
  and sexual phylogenies in Pandas.
- **``_asexual``** (e.g., ``alifestd_mark_node_depth_asexual``) --- optimized
  for asexual phylogenies using ``ancestor_id``.
  Raises an error or produces incorrect results if called on sexual
  phylogenies.
- **``_polars``** (e.g., ``alifestd_mark_leaves_polars``) --- Polars-based
  implementation.
  Requires asexual phylogeny with topological sorting and contiguous IDs.

When both a non-suffixed and ``_asexual`` version exist, prefer the
``_asexual`` version for asexual phylogenies --- it will often be faster.

Validation
==========

Use ``alifestd_validate`` to check that a DataFrame conforms to the alife
standard format.
It returns ``True`` if valid or ``False`` if problems are detected, issuing
warnings describing each issue found:

.. code-block:: python

   from phyloframe import legacy as pfl

   df = pfl.alifestd_from_newick("((A,B),(C,D));")
   pfl.alifestd_validate(df)  # True

Topological Sensitivity
========================

When topology-altering operations (e.g., pruning, collapsing, rerooting)
modify the tree structure, previously computed columns like ``node_depth``,
``branch_length``, or ``num_descendants`` may become stale.
Phyloframe's **topological sensitivity** system detects this and warns you.

Operations that alter topology are decorated to automatically emit a warning
listing any topology-dependent columns found in the DataFrame:

.. code-block:: text

   UserWarning: alifestd_collapse_unifurcations performs delete/update
   operations that do not update topology-dependent columns, which may
   be invalidated: ['node_depth', 'branch_length']. ...

To handle this:

1. **Drop sensitive columns before the operation** using
   ``alifestd_drop_topological_sensitivity`` or
   ``alifestd_drop_topological_sensitivity_polars``, then recompute them
   afterward.
2. **Pass ``drop_topological_sensitivity=True``** to the operation itself,
   which automatically drops topology-dependent columns as part of the call.
3. **Suppress the warning** by passing
   ``ignore_topological_sensitivity=True`` to the operation, or by setting
   the ``HSTRAT_ALIFESTD_WARN_TOPOLOGICAL_SENSITIVITY_SUPPRESS``
   environment variable.

.. code-block:: python

   from phyloframe import legacy as pfl

   df = pfl.alifestd_from_newick("((A,B),(C,D));")
   df = pfl.alifestd_to_working_format(df)
   df = pfl.alifestd_mark_node_depth_asexual(df)

   # Option 1: drop topology-dependent columns explicitly
   df = pfl.alifestd_drop_topological_sensitivity(df)
   df = pfl.alifestd_collapse_unifurcations(df)

   # Recompute as needed
   df = pfl.alifestd_mark_node_depth_asexual(df)

Option 2 lets the operation drop them automatically:

.. code-block:: python

   df = pfl.alifestd_from_newick("((A,B),(C,D));")
   df = pfl.alifestd_to_working_format(df)
   df = pfl.alifestd_mark_node_depth_asexual(df)

   df = pfl.alifestd_collapse_unifurcations(
       df, drop_topological_sensitivity=True,
   )

Supplemental Data Structures
=============================

For algorithms that need to navigate the tree beyond the parent pointer
(``ancestor_id``), phyloframe provides two supplemental representations
that can be added as columns.
These structures optimize certain tree operations, and they will typically
be automatically generated as needed.

CSR (Compressed Sparse Row)
---------------------------

The CSR format represents the parent-to-children mapping as two flat arrays,
enabling O(1) lookup of any node's children:

``csr_offsets`` : array of int
    ``csr_offsets[i]`` is the index in ``csr_children`` where node ``i``'s
    children begin.

``csr_children`` : array of int
    A flat array of all child IDs, grouped by parent.
    Node ``i``'s children are
    ``csr_children[csr_offsets[i] : csr_offsets[i] + num_children[i]]``.

.. code-block:: python

   df = pfl.alifestd_from_newick("((A,B),(C,D));")
   df = pfl.alifestd_to_working_format(df)

   # Add CSR columns
   df = pfl.alifestd_mark_num_children_asexual(df)
   df = pfl.alifestd_mark_csr_offsets_asexual(df)
   df = pfl.alifestd_mark_csr_children_asexual(df)

   # Access node 0's children
   offsets = df["csr_offsets"].values
   children = df["csr_children"].values
   num_children = df["num_children"].values

   node = 0
   node_children = children[offsets[node]:offsets[node] + num_children[node]]

The CSR format is used internally by traversal algorithms (preorder,
postorder) and distance matrix computation.

First-Child/Next-Sibling Linked List
-------------------------------------

An alternative child-navigation structure uses two integer columns to form
a linked list:

``first_child_id`` : int
    The smallest-ID child of this node, or the node's own ID if it is a leaf.

``next_sibling_id`` : int
    The next sibling (by ID order) sharing the same parent, or the node's own
    ID if there is no next sibling.

.. code-block:: python

   df = pfl.alifestd_from_newick("((A,B),(C,D));")
   df = pfl.alifestd_to_working_format(df)

   # Add linked list columns
   df = pfl.alifestd_mark_first_child_id_asexual(df)
   df = pfl.alifestd_mark_next_sibling_id_asexual(df)

   # Walk children of a node
   first_child = df["first_child_id"].values
   next_sibling = df["next_sibling_id"].values

   def iter_children(node_id):
       """Iterate over children of node_id."""
       child = first_child[node_id]
       if child == node_id:  # leaf node, no children
           return
       while True:
           yield child
           nxt = next_sibling[child]
           if nxt == child:  # no more siblings
               break
           child = nxt

This representation uses less memory than CSR for sparse trees and is used
by some traversal algorithms internally.

Pandas vs. Polars
=================

Phyloframe provides dual implementations for many operations:

- **Pandas** --- the default, supporting both asexual and sexual phylogenies.
- **Polars** --- available via operations with a ``_polars`` suffix (e.g.,
  ``alifestd_mark_leaves_polars``).

Polars Usage
------------

Use the ``_polars`` suffixed functions for Polars DataFrames:

.. code-block:: python

   import polars as pl
   from phyloframe import legacy as pfl

   df_pl = pfl.alifestd_from_newick_polars("((A,B),(C,D));")
   df_pl = pfl.alifestd_mark_leaves_polars(df_pl)

Polars Restrictions
-------------------

Polars implementations are more restrictive than Pandas:

- **Asexual only** --- sexual phylogenies are not supported.
- **Topological sortedness required** --- data must be sorted before use.
- **Contiguous IDs required** --- IDs must equal row indices.
- **No ``mutate`` parameter** --- Polars DataFrames are immutable by design.

When to Prefer Polars
---------------------

- Working with large trees (millions of nodes).
- `Query optimization <https://docs.pola.rs/user-guide/lazy/optimizations/>`_
  via Polars' lazy evaluation engine (predicate pushdown, projection pushdown,
  and other automatic rewrites).
- Multithreaded operations.
- CLI pipelines (the CLI interface is Polars-based, so using ``_polars`` entrypoints avoids conversion overhead).

User-extensible Columns
========================

Because the underlying representation is a standard DataFrame, you can freely
add custom columns for your analysis:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from phyloframe import legacy as pfl

   df = pfl.alifestd_from_newick("((A:1,B:2):3,(C:4,D:5):6);")
   df = pfl.alifestd_to_working_format(df)
   df = pfl.alifestd_mark_node_depth_asexual(df)

   # Add trait data
   df["fitness"] = np.random.random(len(df))

   # Filter using standard DataFrame operations
   deep = df[df["node_depth"] > 1]

   # Group, aggregate, join --- all standard operations work
   clade_stats = df.groupby("node_depth").agg(
       count=("id", "count"),
       mean_fitness=("fitness", "mean"),
   )

This extensibility is a key advantage of the DataFrame-based approach.

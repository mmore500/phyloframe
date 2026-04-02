===========================
Concepts and Data Structures
===========================

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
   The ``ancestor_id`` column is an extension introduced by phyloframe for
   efficient asexual phylogeny operations.
   Use ``alifestd_try_add_ancestor_id_col`` to add it automatically.

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

Many operations run fastest when the DataFrame satisfies three properties:

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

Convert to working format with a single call:

.. code-block:: python

   from phyloframe import legacy as pfl

   df = pfl.alifestd_to_working_format(df)

This applies ``alifestd_try_add_ancestor_id_col``,
``alifestd_topological_sort``, and ``alifestd_assign_contiguous_ids``
as needed.

In working format, ``ancestor_id`` values can be used directly as array
indices, enabling efficient NumPy and JIT-compiled operations:

.. code-block:: python

   import numpy as np

   ancestor_ids = df["ancestor_id"].values
   # ancestor_ids[i] gives the row index of node i's parent
   # For root nodes, ancestor_ids[i] == i

Asexual vs. Sexual Phylogenies
==============================

Phyloframe supports two reproduction modes:

**Asexual** phylogenies have at most one ancestor per organism.
The ``ancestor_id`` column can represent the tree structure efficiently.
Most phyloframe operations (especially those with ``_asexual`` suffix) target
this mode.

**Sexual** phylogenies allow multiple ancestors per organism.
These rely on the ``ancestor_list`` column (string-based) and have fewer
optimized operations available.

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
  and sexual phylogenies, using ``ancestor_list`` or auto-detecting mode.
- **``_asexual``** (e.g., ``alifestd_mark_node_depth_asexual``) --- optimized
  for asexual phylogenies using ``ancestor_id``.
  Raises an error or produces incorrect results if called on sexual
  phylogenies.
- **``_polars``** (e.g., ``alifestd_mark_leaves_polars``) --- Polars
  implementation.
  Requires asexual phylogeny with topological sorting and contiguous IDs.

When both a non-suffixed and ``_asexual`` version exist, prefer the
``_asexual`` version for asexual phylogenies --- it will typically be faster.
When a ``_polars`` version exists and your data is already in Polars format,
prefer it to avoid conversion overhead.

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
       mean_fitness=("fitness", "mean"),
       count=("id", "count"),
   )

This extensibility is a key advantage of the DataFrame-based approach.

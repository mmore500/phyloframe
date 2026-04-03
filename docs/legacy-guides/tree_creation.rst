=============================
Creating Phylogenies (Legacy)
=============================

.. note::

   This documentation covers the **legacy** API (from phyloframe import legacy).
   The legacy API is stable and will continue to be maintained for backward
   compatibility.
   A redesigned API will accompany phyloframe v1.0.0.


This guide covers the different ways to create phylogeny DataFrames in phyloframe.

Empty Phylogenies
=================

``alifestd_make_empty`` creates a zero-row DataFrame with the correct column names and dtypes (``id`` as int, ``ancestor_list`` as str, and optionally ``ancestor_id`` as int).
This ensures downstream operations receive properly typed input:

.. code-block:: python

   from phyloframe import legacy as pfl

   # Minimal empty DataFrame (id and ancestor_list columns)
   df = pfl.alifestd_make_empty()

   # With ancestor_id column pre-created
   df = pfl.alifestd_make_empty(ancestor_id=True)

   # Polars version (ancestor_id=True by default)
   df_polars = pfl.alifestd_make_empty_polars()

From Scratch with Pandas
========================

Build a phylogeny by constructing a DataFrame directly:

.. code-block:: python

   import pandas as pd

   # A simple tree:
   #       0 (root)
   #      / \
   #     1   2
   #    / \
   #   3   4
   phylogeny_df = pd.DataFrame({
       "id": [0, 1, 2, 3, 4],
       "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
   })

   # Validate the format
   assert pfl.alifestd_validate(phylogeny_df)

You can include additional columns at creation time:

.. code-block:: python

   phylogeny_df = pd.DataFrame({
       "id": [0, 1, 2, 3, 4],
       "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
       "origin_time": [0, 10, 10, 20, 20],
       "taxon_label": ["root", "A", "B", "C", "D"],
   })

Synthetic Trees
===============

Balanced Bifurcating Trees
--------------------------

Create perfectly balanced binary trees by specifying depth:

.. code-block:: python

   # depth=0 -> empty tree
   # depth=1 -> 1 node (root only)
   # depth=2 -> 3 nodes (1 root, 2 leaves)
   # depth=3 -> 7 nodes (3 internal, 4 leaves)
   # depth=n -> 2^n - 1 total nodes, 2^(n-1) leaves

   df = pfl.alifestd_make_balanced_bifurcating(depth=4)  # 15 nodes
   print(f"Nodes: {len(df)}")
   print(f"Leaves: {pfl.alifestd_count_leaf_nodes(df)}")

Comb (Caterpillar) Trees
-------------------------

Create maximally unbalanced trees by specifying the number of leaves:

.. code-block:: python

   df = pfl.alifestd_make_comb(n_leaves=10)
   print(f"Nodes: {len(df)}")
   print(f"Leaves: {pfl.alifestd_count_leaf_nodes(df)}")

Parsing Newick Format
=====================

Parse Newick-format strings into phyloframe DataFrames:

.. code-block:: python

   # Simple topology
   df = pfl.alifestd_from_newick("((A,B),(C,D));")

   # With branch lengths
   df = pfl.alifestd_from_newick("((A:1.0,B:2.0):3.0,(C:4.0,D:5.0):6.0);")

   # Parsed columns include:
   #   id, ancestor_id, taxon_label, origin_time_delta, branch_length
   print(df.columns.tolist())

Polars Newick Parsing
---------------------

.. code-block:: python

   df_polars = pfl.alifestd_from_newick_polars("((A,B),(C,D));")

Options
-------

.. code-block:: python

   # Integer branch lengths
   df = pfl.alifestd_from_newick(
       "((A:1,B:2):3,(C:4,D:5):6);",
       branch_length_dtype=int,
   )

   # Include ancestor_list column (slower, for compatibility)
   df = pfl.alifestd_from_newick(
       "((A,B),(C,D));",
       create_ancestor_list=True,
   )

Loading from Files
==================

Since phyloframe uses standard DataFrames, loading from files uses standard library calls:

.. code-block:: python

   import pandas as pd

   # From CSV
   df = pd.read_csv("phylogeny.csv")

   # From Parquet (recommended for large trees)
   df = pd.read_parquet("phylogeny.pqt")

   # From a URL
   df = pd.read_csv("https://example.com/data/phylogeny.csv")

   # From cloud storage
   df = pd.read_parquet("s3://bucket/phylogeny.pqt")

.. code-block:: python

   import polars as pl

   # Polars Parquet (efficient columnar reads)
   df = pl.read_parquet("phylogeny.pqt")

   # Selective column loading with Parquet
   df = pl.read_parquet("phylogeny.pqt", columns=["id", "ancestor_id"])

After loading, convert to working format for analysis:

.. code-block:: python

   df = pfl.alifestd_to_working_format(df)

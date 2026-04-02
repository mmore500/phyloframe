=========================
Input and Output (Legacy)
=========================

.. note::

   This documentation covers the **legacy** API (from phyloframe import legacy).
   The legacy API is stable and will continue to be maintained for backward
   compatibility.
   A redesigned API will accompany phyloframe v1.0.0.


This guide covers reading, writing, and converting phylogenetic data in
phyloframe.

Newick Format
=============

Parsing Newick Strings
----------------------

.. code-block:: python

   from phyloframe import legacy as pfl

   # Basic topology
   df = pfl.alifestd_from_newick("((A,B),(C,D));")

   # With branch lengths
   df = pfl.alifestd_from_newick("((A:1.0,B:2.5):3.0,(C:4.0,D:5.0):6.0);")

   # Columns created: id, ancestor_id, taxon_label,
   #                  origin_time_delta, branch_length

Newick Parsing Options
----------------------

.. code-block:: python

   # Integer branch lengths (uses nullable integer dtype)
   df = pfl.alifestd_from_newick(
       "((A:1,B:2):3,(C:4,D:5):6);",
       branch_length_dtype=int,
   )

   # Include ancestor_list column for compatibility
   df = pfl.alifestd_from_newick(
       "((A,B),(C,D));",
       create_ancestor_list=True,
   )

   # Polars version
   df_polars = pfl.alifestd_from_newick_polars("((A,B),(C,D));")

Exporting to Newick
--------------------

.. code-block:: python

   # Export to Newick string
   newick_str = pfl.alifestd_as_newick_asexual(df)

   # Include taxon labels from a column
   newick_str = pfl.alifestd_as_newick_asexual(
       df, taxon_label="taxon_label",
   )

Tabular File Formats (CSV, Parquet)
====================================

Use standard Pandas and Polars I/O utilities for reading and writing
phylogeny DataFrames.
Parquet is recommended for large phylogenies due to columnar compression,
selective column loading, explicit typing, and efficient enum-based
categorical string storage.

.. code-block:: python

   import pandas as pd
   import polars as pl

   # CSV --- Pandas
   df.to_csv("phylogeny.csv", index=False)
   df = pd.read_csv("phylogeny.csv")

   # Parquet --- Pandas
   df.to_parquet("phylogeny.pqt")
   df = pd.read_parquet("phylogeny.pqt")

   # Parquet --- Polars (selective column loading)
   df_polars.write_parquet("phylogeny.pqt")
   df_polars = pl.read_parquet(
       "phylogeny.pqt", columns=["id", "ancestor_id", "origin_time"],
   )

Selective column deserialization is particularly advantageous with Polars
streaming operations.
See the `Pandas I/O docs <https://pandas.pydata.org/docs/user_guide/io.html>`_
and `Polars I/O docs <https://docs.pola.rs/user-guide/io/>`_ for full details.

Remote and Cloud Sources
========================

DataFrame libraries transparently handle URLs and cloud storage:

.. code-block:: python

   import pandas as pd

   # From URL
   df = pd.read_csv("https://example.com/data/phylogeny.csv")

   # From S3
   df = pd.read_parquet("s3://bucket/phylogeny.pqt")

   # From Google Cloud Storage
   df = pd.read_parquet("gs://bucket/phylogeny.pqt")

.. code-block:: python

   import polars as pl

   # Polars also supports remote sources
   df = pl.read_parquet("s3://bucket/phylogeny.pqt")

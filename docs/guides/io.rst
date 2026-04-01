================
Input and Output
================

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

CSV Files
=========

Standard Pandas/Polars CSV I/O works directly:

.. code-block:: python

   import pandas as pd

   # Write
   df.to_csv("phylogeny.csv", index=False)

   # Read
   df = pd.read_csv("phylogeny.csv")
   df = pfl.alifestd_to_working_format(df)

.. code-block:: python

   import polars as pl

   # Write
   df_polars.write_csv("phylogeny.csv")

   # Read
   df_polars = pl.read_csv("phylogeny.csv")

Parquet Files
=============

Parquet is recommended for large phylogenies.
It offers columnar compression, explicit typing, and selective column
loading.

.. code-block:: python

   import pandas as pd

   # Write
   df.to_parquet("phylogeny.pqt")

   # Read
   df = pd.read_parquet("phylogeny.pqt")

.. code-block:: python

   import polars as pl

   # Write
   df_polars.write_parquet("phylogeny.pqt")

   # Read --- only load the columns you need
   df_polars = pl.read_parquet(
       "phylogeny.pqt", columns=["id", "ancestor_id", "origin_time"],
   )

Advantages of Parquet:

- Columnar compression reduces file size.
- Selective column deserialization speeds up loading.
- Explicit column typing avoids dtype inference issues.
- Binary format is faster to read/write than CSV.
- Categorical strings are stored efficiently.

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

Cross-language Interoperation
=============================

The DataFrame-based format enables zero-copy or near-zero-copy
interoperation across languages:

R via reticulate and Arrow
--------------------------

.. code-block:: r

   # In R, read a phyloframe-produced file
   library(arrow)
   df <- read_parquet("phylogeny.pqt")

   # Or use reticulate for direct Python interop
   library(reticulate)
   pfl <- import("phyloframe.legacy")
   df <- pfl$alifestd_from_newick("((A,B),(C,D));")

R's ``read.table`` and ``read.csv`` also work for CSV files:

.. code-block:: r

   df <- read.csv("phylogeny.csv")

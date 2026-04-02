======================
Command-line Interface
======================

All dataframe-to-dataframe transforms are available as CLI commands, as well
as some other operations.
This enables use from shell scripts and pipelines without writing Python code.

Listing Available Commands
==========================

.. code-block:: bash

   python3 -m phyloframe --help

This prints all available CLI commands, each corresponding to a module in
``phyloframe.legacy``.

Basic Usage
===========

Each command reads a DataFrame (e.g., CSV, Parquet, etc.) from stdin and writes the
result to an output file.
Note that stdin takes filenames, one item per line; use the ``--stdin`` flag to read file content directly (requires ``--input-filetype`` and ``--output-filetype``).

.. code-block:: bash

   # Read from stdin, write to file
   python3 -m phyloframe.legacy._alifestd_mark_leaves output.csv < input.csv

   # With custom arguments
   python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --mark-as is_tip output.csv < input.csv

Get help for any command:

.. code-block:: bash

   python3 -m phyloframe.legacy._alifestd_mark_leaves --help

Input and Output Formats
=========================

The data format is determined by file extension:

- ``.csv`` --- CSV format
- ``.pqt`` or ``.parquet`` --- Parquet format

.. code-block:: bash

   # CSV to CSV
   python3 -m phyloframe.legacy._alifestd_mark_leaves output.csv < input.csv

   # Parquet to Parquet
   python3 -m phyloframe.legacy._alifestd_mark_leaves output.pqt < input.pqt

In-place Modification
---------------------

Use ``--eager-read`` when reading and writing the same file:

.. code-block:: bash

   python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --eager-read data.csv < data.csv

Piping Commands
===============

Chain operations using Unix pipes.
Write intermediate output to ``/dev/stdout``:

.. code-block:: bash

   python3 -m phyloframe.legacy._alifestd_collapse_unifurcations /dev/stdout \
       < input.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves /dev/stdout \
     | python3 -m phyloframe.legacy._alifestd_mark_node_depth_asexual \
       output.csv

Multi-operation Pipe Utility
----------------------------

For multi-step pipelines, ``_alifestd_pipe_unary_ops`` applies several
operations in sequence within a single process:

.. code-block:: bash

   python3 -m phyloframe.legacy._alifestd_pipe_unary_ops \
       --op "pfl.alifestd_collapse_unifurcations" \
       --op "pfl.alifestd_mark_leaves" \
       --op "pfl.alifestd_mark_node_depth_asexual" \
       output.csv < input.csv

Available names in ``--op`` expressions: ``pfl`` (phyloframe.legacy),
``pf`` (phyloframe), ``pd`` (pandas), ``pl`` (polars), ``np`` (numpy),
``opyt`` (opytional).

Lambda expressions also work:

.. code-block:: bash

   python3 -m phyloframe.legacy._alifestd_pipe_unary_ops \
       --op "pfl.alifestd_mark_leaves" \
       --op "lambda df: df[df['is_leaf']]" \
       output.csv < input.csv

Polars CLI Entrypoints
======================

For best performance, prefer the Polars CLI entrypoints (modules ending in
``_polars``) when working with Parquet data.
This avoids Pandas-to-Polars conversion overhead:

.. code-block:: bash

   # Pandas entrypoint (converts internally)
   python3 -m phyloframe.legacy._alifestd_mark_leaves output.pqt < input.pqt

   # Polars entrypoint (no conversion, faster)
   python3 -m phyloframe.legacy._alifestd_mark_leaves_polars \
       output.pqt < input.pqt

The Polars pipe utility:

.. code-block:: bash

   python3 -m phyloframe.legacy._alifestd_pipe_unary_ops_polars \
       --op "pfl.alifestd_mark_leaves_polars" \
       output.pqt < input.pqt

joinem CLI Engine
=================

Phyloframe's CLI is built on the `joinem <https://github.com/mmore500/joinem>`_ CLI engine.
All joinem features are available in phyloframe CLI commands.

Column Selection
----------------

Use ``--select`` and ``--drop`` to control which columns appear in the output:

.. code-block:: bash

   # Keep only specific columns
   python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --select id --select ancestor_list --select is_leaf \
       output.csv < input.csv

   # Drop unwanted columns
   python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --drop ancestor_list output.csv < input.csv

Row Selection
-------------

Use ``--head``, ``--tail``, ``--sample``, and ``--shuffle`` to control which rows appear in the output:

.. code-block:: bash

   # Keep only the first 100 rows
   python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --head 100 output.csv < input.csv

   # Random sample of 50 rows
   python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --sample 50 output.csv < input.csv

Filtering and Computed Columns
------------------------------

Use ``--filter`` to filter rows and ``--with-column`` to add computed columns using Polars expressions:

.. code-block:: bash

   # Filter to leaf nodes only
   python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --filter "pl.col('is_leaf')" output.csv < input.csv

   # Add a computed column
   python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --with-column "pl.col('id').cast(pl.Utf8).alias('id_str')" \
       output.csv < input.csv

Other joinem Features
---------------------

``--shrink-dtypes``
    Minimize numeric column sizes for smaller output files.

``--read-kwarg KEY=VALUE``
    Pass additional keyword arguments to the reader (e.g., CSV delimiter).

``--write-kwarg KEY=VALUE``
    Pass additional keyword arguments to the writer.

See the `joinem documentation <https://github.com/mmore500/joinem>`_ for full details.

Common CLI Arguments
====================

Most commands share these arguments:

``--eager-read``
    Read the input file eagerly (required for in-place modification).

``--mark-as COLUMN``
    Output column name (for mark operations).

``--help``
    Show help text and available arguments.

``--version``
    Show version information.

Container Usage
===============

A containerized release of phyloframe is available:

.. code-block:: bash

   # Via Singularity
   singularity exec docker://ghcr.io/mmore500/phyloframe:v0.6.1 \
       python3 -m phyloframe.legacy._alifestd_mark_leaves \
       output.csv < input.csv

   # Via Docker
   docker run --rm -i ghcr.io/mmore500/phyloframe:v0.6.1 \
       python3 -m phyloframe.legacy._alifestd_mark_leaves \
       output.csv < input.csv

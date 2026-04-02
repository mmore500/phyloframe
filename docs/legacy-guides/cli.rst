===============================
Command-line Interface (Legacy)
===============================

.. note::

   This documentation covers the **legacy** API (from phyloframe import legacy).
   The legacy API is stable and will continue to be maintained for backward
   compatibility.
   A redesigned API will accompany phyloframe v1.0.0.


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

Each command takes an output file as a positional argument.
Input files are provided via stdin, typically using ``ls -1``.
The data format is inferred from the file extension; use ``--input-filetype``
and ``--output-filetype`` flags when the type cannot be inferred (e.g., when
piping through ``/dev/stdin`` or ``/dev/stdout``).

.. code-block:: bash

   # Read from file, write to file
   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves output.csv

   # With custom arguments
   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --mark-as is_tip output.csv

   # With explicit filetype flags (when extension is unavailable)
   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --input-filetype .csv --output-filetype .csv /dev/stdout > output.csv

Get help for any command:

.. code-block:: bash

   python3 -m phyloframe.legacy._alifestd_mark_leaves --help

Input and Output Formats
=========================

The data format is determined by file extension:

- ``.csv`` --- CSV format
- ``.pqt`` or ``.parquet`` --- Parquet format
- ``.json`` --- JSON format
- ``.feather`` or ``.ipc`` --- Feather/IPC format

.. code-block:: bash

   # CSV to CSV
   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves output.csv

   # Parquet to Parquet
   ls -1 input.pqt \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves output.pqt

   # CSV to JSON
   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves output.json

   # CSV to Feather
   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves output.feather

In-place Modification
---------------------

Use ``--eager-read`` when reading and writing the same file:

.. code-block:: bash

   ls -1 data.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --eager-read data.csv

Piping Commands
===============

Chain operations using Unix pipes.
Use ``/dev/stdout`` and ``/dev/stdin`` with ``--input-filetype`` and
``--output-filetype`` flags:

.. code-block:: bash

   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_collapse_unifurcations \
       --output-filetype .csv /dev/stdout \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --input-filetype .csv --output-filetype .csv /dev/stdout \
     | python3 -m phyloframe.legacy._alifestd_mark_node_depth_asexual \
       --input-filetype .csv output.csv

Multi-operation Pipe Utility
----------------------------

For multi-step pipelines, ``_alifestd_pipe_unary_ops`` applies several
operations in sequence within a single process:

.. code-block:: bash

   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_pipe_unary_ops \
       --op "pfl.alifestd_collapse_unifurcations" \
       --op "pfl.alifestd_mark_leaves" \
       --op "pfl.alifestd_mark_node_depth_asexual" \
       output.csv

Available names in ``--op`` expressions: ``pfl`` (phyloframe.legacy),
``pf`` (phyloframe), ``pd`` (pandas), ``pl`` (polars), ``np`` (numpy),
``opyt`` (opytional).

Use lambda expressions for multi-step workflows like combining sample
marks and pruning:

.. code-block:: bash

   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_pipe_unary_ops \
       --op "pfl.alifestd_to_working_format" \
       --op "lambda df: pfl.alifestd_mark_sample_tips_canopy_asexual(df, n_sample=5, mark_as='keep_canopy')" \
       --op "lambda df: pfl.alifestd_mark_sample_tips_lineage_asexual(df, n_sample=5, mark_as='keep_lineage')" \
       --op "lambda df: df.assign(extant=df['keep_canopy'] | df['keep_lineage'])" \
       --op "pfl.alifestd_prune_extinct_lineages_asexual" \
       output.csv

Polars CLI Entrypoints
======================

For best performance, prefer the Polars CLI entrypoints (modules ending in
``_polars``) when working with Parquet data.
This avoids Pandas-to-Polars conversion overhead:

.. code-block:: bash

   # Pandas entrypoint (converts internally)
   ls -1 input.pqt \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves output.pqt

   # Polars entrypoint (no conversion, faster)
   ls -1 input.pqt \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves_polars output.pqt

The Polars pipe utility:

.. code-block:: bash

   ls -1 input.pqt \
     | python3 -m phyloframe.legacy._alifestd_pipe_unary_ops_polars \
       --op "pfl.alifestd_mark_leaves_polars" \
       output.pqt

joinem CLI Engine
=================

Phyloframe's CLI is built on the `joinem <https://github.com/mmore500/joinem>`_ CLI engine.
All joinem features are available in phyloframe CLI commands.

Column Selection
----------------

Use ``--select`` and ``--drop`` to control which columns appear in the output:

.. code-block:: bash

   # Keep only specific columns
   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --select id --select ancestor_list --select is_leaf \
       output.csv

   # Drop unwanted columns
   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --drop ancestor_list output.csv

Row Selection
-------------

Use ``--head``, ``--tail``, ``--sample``, and ``--shuffle`` to control which rows appear in the output:

.. code-block:: bash

   # Keep only the first 100 rows
   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --head 100 output.csv

   # Random sample of 50 rows
   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --sample 50 output.csv

Filtering and Computed Columns
------------------------------

Use ``--filter`` to filter rows and ``--with-column`` to add computed columns using Polars expressions:

.. code-block:: bash

   # Filter to leaf nodes only
   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --filter "pl.col('is_leaf')" output.csv

   # Add a computed column
   ls -1 input.csv \
     | python3 -m phyloframe.legacy._alifestd_mark_leaves \
       --with-column "pl.col('id').cast(pl.Utf8).alias('id_str')" \
       output.csv

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
   ls -1 input.csv \
     | singularity exec docker://ghcr.io/mmore500/phyloframe:v0.6.1 \
       python3 -m phyloframe.legacy._alifestd_mark_leaves output.csv

   # Via Docker
   ls -1 input.csv \
     | docker run --rm -i ghcr.io/mmore500/phyloframe:v0.6.1 \
       python3 -m phyloframe.legacy._alifestd_mark_leaves output.csv

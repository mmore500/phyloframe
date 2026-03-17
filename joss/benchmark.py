#!/usr/bin/env python3
"""Benchmark phyloframe against other phylogenetics libraries.

Reproduces the TreeSwift paper benchmark: preorder, postorder, inorder,
levelorder traversals, all-pairs MRCA, and all-pairs pairwise distances
on binary trees with 100 to 1,000,000 leaves.  Also benchmarks newick
load/save.  Any single operation exceeding TIMEOUT seconds is skipped.

Each operation runs in a forkserver subprocess that does its own setup
(imports, JIT warmup) and reports self-timed results back via a pipe,
so import/warmup overhead never pollutes the measured time.
"""

import csv
import gc
import io
import multiprocessing
import sys
import time

# Use forkserver so each subprocess starts clean — avoids deadlocks from
# forking a process with polars/numba background threads.
multiprocessing.set_start_method("forkserver", force=True)

TIMEOUT_PROBE = 10  # seconds for 100x-smaller preamble load
TIMEOUT_LOAD = 100  # seconds for load_newick
TIMEOUT_OP = 210  # seconds for subsequent operations
SIZES = [
    100,
    300,
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
    3_000_000,
    10_000_000,
    30_000_000,
]
OPERATIONS = [
    "load_newick",
    "save_newick",
    "preorder",
    "postorder",
    "inorder",
    "levelorder",
    "topological_order",
    "mrca_allpairs",
    "pairwise_dist",
    "memory_bytes",
]


def _set_memory_limit():
    """Cap this process's memory at 120% of available RAM.

    Sets limits via resource.setrlimit (RLIMIT_AS and RLIMIT_DATA) so the
    OS kills the subprocess instead of letting it OOM the whole CI runner.
    """
    import resource

    import psutil

    available = psutil.virtual_memory().available
    limit = int(available * 1.2)
    print(
        f"  |   mem limit: {limit / 1e9:.1f} GB"
        f" (120% of {available / 1e9:.1f} GB avail)",
        file=sys.stderr,
    )

    # RLIMIT_AS — virtual address space; most reliable on Linux.
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except (ValueError, OSError):
        pass

    # RLIMIT_DATA — heap size; redundant safety net.
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_DATA)
        resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))
    except (ValueError, OSError):
        pass


def _run_in_child(conn, bench_cls, newick, op, return_value):
    """Target for subprocess: construct bench, warmup, run op, send timing."""
    _set_memory_limit()
    try:
        bench = bench_cls(newick)
        bench.warmup()
        # Pre-load the tree so parsing isn't included in operation timing.
        if op not in ("load_newick", "memory_bytes"):
            bench.load_newick()
        fn = getattr(bench, op)
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        if return_value:
            conn.send(("ok", elapsed, result))
        else:
            conn.send(("ok", elapsed, None))
    except NotImplementedError as exc:
        conn.send(("unavailable", str(exc)))
    except Exception as exc:
        conn.send(("error", str(exc)))
    finally:
        conn.close()


def _run_memory_child(conn, bench_cls, newick):
    """Measure memory in a subprocess with full warmup.

    Full warmup triggers lazy runtime init (thread pools, JIT, etc.).
    _measure_memory() handles gc + malloc_trim before taking baseline RSS.
    """
    _set_memory_limit()
    try:
        bench = bench_cls(newick)
        bench.warmup()
        result = bench.memory_bytes()
        conn.send(("ok", result))
    except NotImplementedError as exc:
        conn.send(("unavailable", str(exc)))
    except Exception as exc:
        conn.send(("error", str(exc)))
    finally:
        conn.close()


def timed(bench_cls, newick, op, timeout=TIMEOUT_OP):
    """Run an operation in a subprocess and return (seconds, status).

    Status is one of "SUCCESS", "TIMEOUT", "FAIL", or "UNAVAILABLE".
    seconds is None for non-SUCCESS outcomes.
    """
    parent_conn, child_conn = multiprocessing.Pipe()
    proc = multiprocessing.Process(
        target=_run_in_child,
        args=(child_conn, bench_cls, newick, op, False),
    )
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        parent_conn.close()
        return None, "TIMEOUT"
    if proc.exitcode != 0:
        print(
            f"  |   error: child exited with code {proc.exitcode}",
            file=sys.stderr,
        )
        parent_conn.close()
        return None, "FAIL"
    if parent_conn.poll():
        result = parent_conn.recv()
        parent_conn.close()
        if result[0] == "unavailable":
            return None, "UNAVAILABLE"
        if result[0] == "error":
            print(f"  |   error: {result[1]}", file=sys.stderr)
            return None, "FAIL"
        return result[1], "SUCCESS"
    parent_conn.close()
    return None, "FAIL"


def timed_call(bench_cls, newick, op, timeout=TIMEOUT_OP):
    """Run an operation in a subprocess and return (result, status)."""
    parent_conn, child_conn = multiprocessing.Pipe()
    proc = multiprocessing.Process(
        target=_run_in_child,
        args=(child_conn, bench_cls, newick, op, True),
    )
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        parent_conn.close()
        return None, "TIMEOUT"
    if proc.exitcode != 0:
        print(
            f"  |   error: child exited with code {proc.exitcode}",
            file=sys.stderr,
        )
        parent_conn.close()
        return None, "FAIL"
    if parent_conn.poll():
        result = parent_conn.recv()
        parent_conn.close()
        if result[0] == "unavailable":
            return None, "UNAVAILABLE"
        if result[0] == "error":
            print(f"  |   error: {result[1]}", file=sys.stderr)
            return None, "FAIL"
        return result[2], "SUCCESS"
    parent_conn.close()
    return None, "FAIL"


def measure_memory(bench_cls, newick, timeout=TIMEOUT_OP):
    """Measure memory in a clean subprocess. Returns (bytes, status)."""
    parent_conn, child_conn = multiprocessing.Pipe()
    proc = multiprocessing.Process(
        target=_run_memory_child,
        args=(child_conn, bench_cls, newick),
    )
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        parent_conn.close()
        return None, "TIMEOUT"
    if proc.exitcode != 0:
        print(
            f"  |   error: child exited with code {proc.exitcode}",
            file=sys.stderr,
        )
        parent_conn.close()
        return None, "FAIL"
    if parent_conn.poll():
        result = parent_conn.recv()
        parent_conn.close()
        if result[0] == "unavailable":
            return None, "UNAVAILABLE"
        if result[0] == "error":
            print(f"  |   error: {result[1]}", file=sys.stderr)
            return None, "FAIL"
        return result[1], "SUCCESS"
    parent_conn.close()
    return None, "FAIL"


def _get_rss_bytes():
    """Return current RSS in bytes via /proc/self/statm."""
    import os

    page_size = os.sysconf("SC_PAGE_SIZE")
    with open("/proc/self/statm") as f:
        # statm fields: size resident shared text lib data dt
        resident_pages = int(f.read().split()[1])
    return resident_pages * page_size


def _measure_memory(load_fn):
    """Measure memory consumed by the data structure returned by load_fn().

    Uses RSS (resident set size) delta, which captures both Python-level
    and native/C++ allocations.  This is important for libraries like
    CompactTree that allocate primarily through the system allocator.
    Calls malloc_trim to compact the heap before measuring so that
    freed pages from prior allocations don't mask the delta.
    """
    import ctypes

    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (OSError, AttributeError):
        pass
    before = _get_rss_bytes()
    result = load_fn()  # keep reference alive during measurement
    gc.collect()
    after = _get_rss_bytes()
    _keep_alive = result  # prevent optimizing away  # noqa: F841
    return max(0, after - before)


# ── tree generation ──────────────────────────────────────────────────
def _random_binary_newick(n_leaves):
    """Build a random binary tree with n_leaves as a Newick string.

    Uses treeswift's coalescent simulator when available, otherwise
    builds a balanced binary tree newick string directly.
    """
    try:
        import dendropy

        taxa = dendropy.TaxonNamespace([f"t{i}" for i in range(n_leaves)])
        tree = dendropy.simulate.treesim.birth_death_tree(
            birth_rate=1.0,
            death_rate=0.0,
            num_extant_tips=n_leaves,
            taxon_namespace=taxa,
        )
        return tree.as_string(schema="newick")
    except Exception:
        pass
    # fallback: balanced binary tree
    return _balanced_newick(n_leaves)


def _balanced_newick(n):
    """Build a balanced binary newick string with n leaves."""
    if n <= 0:
        return "();"
    labels = [f"t{i}" for i in range(n)]

    def _build(lo, hi):
        if lo == hi:
            return f"{labels[lo]}:1"
        mid = (lo + hi) // 2
        return f"({_build(lo, mid)},{_build(mid + 1, hi)}):1"

    return _build(0, n - 1) + ";"


# ── library adapters ────────────────────────────────────────────────
class PhyloframeBench:
    name = "phyloframe"
    engine_affinity = None
    _from_newick_dtype_id = None  # None = default (pl.Int64)
    _mark_after_load = ()  # extra columns to mark after load_newick

    _env_overrides = {
        "PHYLOFRAME_DANGEROUSLY_ASSUME_LEGACY_ALIFESTD_HAS_CONTIGUOUS_IDS_POLARS": "1",
        "PHYLOFRAME_DANGEROUSLY_ASSUME_LEGACY_ALIFESTD_IS_TOPOLOGICALLY_SORTED_POLARS": "1",
    }

    def __init__(self, newick):
        self._newick = newick
        self._df = None

    def warmup(self):
        import os

        for key, val in self._env_overrides.items():
            os.environ[key] = val

        import polars as pl

        pl.Config.set_engine_affinity(self.engine_affinity)

        from phyloframe.legacy import (
            alifestd_calc_mrca_id_matrix_asexual_polars,
            alifestd_from_newick_polars,
            alifestd_mark_csr_children_polars,
            alifestd_mark_csr_offsets_polars,
            alifestd_mark_first_child_id_polars,
            alifestd_mark_next_sibling_id_polars,
            alifestd_mark_num_children_polars,
            alifestd_mark_ot_mrca_polars,
            alifestd_unfurl_traversal_inorder_polars,
            alifestd_unfurl_traversal_levelorder_polars,
            alifestd_unfurl_traversal_postorder_contiguous_polars,
            alifestd_unfurl_traversal_preorder_polars,
            alifestd_unfurl_traversal_topological_polars,
        )

        tiny = _balanced_newick(8)
        pldf = alifestd_from_newick_polars(tiny)
        alifestd_mark_first_child_id_polars(pldf)
        alifestd_mark_next_sibling_id_polars(pldf)
        alifestd_mark_num_children_polars(pldf)
        pldf_csr = alifestd_mark_csr_offsets_polars(pldf)
        alifestd_mark_csr_children_polars(pldf_csr)
        alifestd_unfurl_traversal_postorder_contiguous_polars(pldf)
        alifestd_unfurl_traversal_preorder_polars(pldf)
        alifestd_unfurl_traversal_levelorder_polars(pldf)
        alifestd_unfurl_traversal_inorder_polars(pldf)
        alifestd_unfurl_traversal_topological_polars(pldf)
        pldf_with_ot = pldf.with_columns(
            pl.col("origin_time_delta").cum_sum().alias("origin_time"),
        )
        alifestd_mark_ot_mrca_polars(pldf_with_ot)
        alifestd_calc_mrca_id_matrix_asexual_polars(pldf)
        from phyloframe.legacy import (
            alifestd_calc_distance_matrix_polars,
        )

        alifestd_calc_distance_matrix_polars(pldf_with_ot)

    def _do_from_newick(self, newick):
        """Load newick and apply any post-load mark operations."""
        from phyloframe.legacy import alifestd_from_newick_polars

        kwargs = {}
        if self._from_newick_dtype_id is not None:
            kwargs["dtype_id"] = self._from_newick_dtype_id
        df = alifestd_from_newick_polars(newick, **kwargs)
        for mark_fn_name in self._mark_after_load:
            from phyloframe import legacy

            mark_fn = getattr(legacy, mark_fn_name)
            df = mark_fn(df)
        return df

    def load_newick(self):
        self._df = self._do_from_newick(self._newick)

    def save_newick(self):
        from phyloframe.legacy import alifestd_as_newick_polars

        df = self._ensure_df()
        alifestd_as_newick_polars(df)

    def preorder(self):
        from phyloframe.legacy import (
            alifestd_unfurl_traversal_preorder_polars,
        )

        df = self._ensure_df()
        alifestd_unfurl_traversal_preorder_polars(df)

    def postorder(self):
        from phyloframe.legacy import (
            alifestd_unfurl_traversal_postorder_contiguous_polars,
        )

        df = self._ensure_df()
        alifestd_unfurl_traversal_postorder_contiguous_polars(df)

    def inorder(self):
        from phyloframe.legacy import (
            alifestd_unfurl_traversal_inorder_polars,
        )

        df = self._ensure_df()
        alifestd_unfurl_traversal_inorder_polars(df)

    def levelorder(self):
        from phyloframe.legacy import (
            alifestd_unfurl_traversal_levelorder_polars,
        )

        df = self._ensure_df()
        alifestd_unfurl_traversal_levelorder_polars(df)

    def topological_order(self):
        from phyloframe.legacy import (
            alifestd_unfurl_traversal_topological_polars,
        )

        df = self._ensure_df()
        alifestd_unfurl_traversal_topological_polars(df)

    def mrca_allpairs(self):
        from phyloframe.legacy import (
            alifestd_calc_mrca_id_matrix_asexual_polars,
        )

        df = self._ensure_df()
        alifestd_calc_mrca_id_matrix_asexual_polars(df)

    def pairwise_dist(self):
        from phyloframe.legacy import (
            alifestd_calc_distance_matrix_polars,
        )

        df = self._ensure_df()
        if "origin_time" not in df.columns:
            import polars as pl

            df = df.with_columns(
                pl.col("origin_time_delta").cum_sum().alias("origin_time"),
            )
        alifestd_calc_distance_matrix_polars(df)

    def memory_bytes(self):
        newick = self._newick
        return _measure_memory(
            lambda: self._do_from_newick(newick),
        )

    def _ensure_df(self):
        if self._df is None:
            self._df = self._do_from_newick(self._newick)
        return self._df


class PhyloframeInMemoryBench(PhyloframeBench):
    name = "phyloframe (in-memory)"
    engine_affinity = "in-memory"


class PhyloframeStreamingInt32Bench(PhyloframeBench):
    name = "phyloframe (streaming+i32)"
    engine_affinity = "streaming"

    @property
    def _from_newick_dtype_id(self):
        import polars as pl

        return pl.Int32


class PhyloframeStreamingInt32ChildSibBench(PhyloframeStreamingInt32Bench):
    name = "phyloframe (streaming+i32+child/sib)"
    _mark_after_load = (
        "alifestd_mark_first_child_id_polars",
        "alifestd_mark_next_sibling_id_polars",
    )


class PhyloframeStreamingInt32CsrBench(PhyloframeStreamingInt32Bench):
    name = "phyloframe (streaming+i32+csr)"
    _mark_after_load = (
        "alifestd_mark_num_children_polars",
        "alifestd_mark_csr_offsets_polars",
        "alifestd_mark_csr_children_polars",
    )


class TreeswiftBench:
    name = "treeswift"

    def __init__(self, newick):
        self._newick = newick
        self._tree = None

    def warmup(self):
        import treeswift  # noqa: F401

    def load_newick(self):
        import treeswift

        self._tree = treeswift.read_tree_newick(self._newick)

    def save_newick(self):
        t = self._ensure_tree()
        t.newick()

    def preorder(self):
        t = self._ensure_tree()
        for _ in t.traverse_preorder():
            pass

    def postorder(self):
        t = self._ensure_tree()
        for _ in t.traverse_postorder():
            pass

    def inorder(self):
        t = self._ensure_tree()
        for _ in t.traverse_inorder():
            pass

    def levelorder(self):
        t = self._ensure_tree()
        for _ in t.traverse_levelorder():
            pass

    def topological_order(self):
        raise NotImplementedError(
            "topological_order not available in treeswift"
        )

    def mrca_allpairs(self):
        t = self._ensure_tree()
        labels = [n.label for n in t.traverse_leaves()]
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                t.mrca({a, b})

    def pairwise_dist(self):
        t = self._ensure_tree()
        t.distance_matrix(leaf_labels=True)

    def memory_bytes(self):
        newick = self._newick

        def _load():
            import treeswift

            return treeswift.read_tree_newick(newick)

        return _measure_memory(_load)

    def _ensure_tree(self):
        if self._tree is None:
            import treeswift

            self._tree = treeswift.read_tree_newick(self._newick)
        return self._tree


class BiopythonBench:
    name = "biopython"

    def __init__(self, newick):
        self._newick = newick
        self._tree = None

    def warmup(self):
        from Bio import Phylo  # noqa: F401

    def load_newick(self):
        from Bio import Phylo

        self._tree = Phylo.read(io.StringIO(self._newick), "newick")

    def save_newick(self):
        from Bio import Phylo

        t = self._ensure_tree()
        buf = io.StringIO()
        Phylo.write(t, buf, "newick")

    def preorder(self):
        t = self._ensure_tree()
        for _ in t.find_clades(order="preorder"):
            pass

    def postorder(self):
        t = self._ensure_tree()
        for _ in t.find_clades(order="postorder"):
            pass

    def inorder(self):
        raise NotImplementedError("inorder not available in Bio.Phylo")

    def levelorder(self):
        t = self._ensure_tree()
        for _ in t.find_clades(order="level"):
            pass

    def topological_order(self):
        raise NotImplementedError(
            "topological_order not available in biopython"
        )

    def mrca_allpairs(self):
        t = self._ensure_tree()
        terminals = t.get_terminals()
        for i, a in enumerate(terminals):
            for b in terminals[i + 1 :]:
                t.common_ancestor(a, b)

    def pairwise_dist(self):
        t = self._ensure_tree()
        terminals = t.get_terminals()
        for i, a in enumerate(terminals):
            for b in terminals[i + 1 :]:
                t.distance(a, b)

    def memory_bytes(self):
        newick = self._newick

        def _load():
            from Bio import Phylo

            return Phylo.read(io.StringIO(newick), "newick")

        return _measure_memory(_load)

    def _ensure_tree(self):
        if self._tree is None:
            from Bio import Phylo

            self._tree = Phylo.read(io.StringIO(self._newick), "newick")
        return self._tree


class DendropyBench:
    name = "dendropy"

    def __init__(self, newick):
        self._newick = newick
        self._tree = None

    def warmup(self):
        import dendropy  # noqa: F401

    def load_newick(self):
        import dendropy

        self._tree = dendropy.Tree.get(data=self._newick, schema="newick")

    def save_newick(self):
        t = self._ensure_tree()
        t.as_string(schema="newick")

    def preorder(self):
        t = self._ensure_tree()
        for _ in t.preorder_node_iter():
            pass

    def postorder(self):
        t = self._ensure_tree()
        for _ in t.postorder_node_iter():
            pass

    def inorder(self):
        t = self._ensure_tree()
        for _ in t.inorder_node_iter():
            pass

    def levelorder(self):
        t = self._ensure_tree()
        for _ in t.levelorder_node_iter():
            pass

    def topological_order(self):
        raise NotImplementedError(
            "topological_order not available in dendropy"
        )

    def mrca_allpairs(self):

        t = self._ensure_tree()
        pdm = t.phylogenetic_distance_matrix()
        leaf_list = list(t.leaf_node_iter())
        for i, a in enumerate(leaf_list):
            for b in leaf_list[i + 1 :]:
                pdm.mrca(a.taxon, b.taxon)

    def pairwise_dist(self):
        t = self._ensure_tree()
        pdm = t.phylogenetic_distance_matrix()
        leaf_list = list(t.leaf_node_iter())
        for i, a in enumerate(leaf_list):
            for b in leaf_list[i + 1 :]:
                pdm(a.taxon, b.taxon)

    def memory_bytes(self):
        newick = self._newick

        def _load():
            import dendropy

            return dendropy.Tree.get(data=newick, schema="newick")

        return _measure_memory(_load)

    def _ensure_tree(self):
        if self._tree is None:
            import dendropy

            self._tree = dendropy.Tree.get(data=self._newick, schema="newick")
        return self._tree


class EteBench:
    name = "ete"

    def __init__(self, newick):
        self._newick = newick
        self._tree = None

    def warmup(self):
        from ete3 import Tree  # noqa: F401

    def load_newick(self):
        from ete3 import Tree

        self._tree = Tree(self._newick)

    def save_newick(self):
        t = self._ensure_tree()
        t.write()

    def preorder(self):
        t = self._ensure_tree()
        for _ in t.traverse("preorder"):
            pass

    def postorder(self):
        t = self._ensure_tree()
        for _ in t.traverse("postorder"):
            pass

    def inorder(self):
        raise NotImplementedError("inorder not available in ete")

    def levelorder(self):
        t = self._ensure_tree()
        for _ in t.traverse("levelorder"):
            pass

    def topological_order(self):
        raise NotImplementedError("topological_order not available in ete")

    def mrca_allpairs(self):
        t = self._ensure_tree()
        leaves = list(t.get_leaves())
        for i, a in enumerate(leaves):
            for b in leaves[i + 1 :]:
                t.get_common_ancestor(a, b)

    def pairwise_dist(self):
        t = self._ensure_tree()
        leaves = list(t.get_leaves())
        for i, a in enumerate(leaves):
            for b in leaves[i + 1 :]:
                a.get_distance(b)

    def memory_bytes(self):
        newick = self._newick

        def _load():
            from ete3 import Tree

            return Tree(newick)

        return _measure_memory(_load)

    def _ensure_tree(self):
        if self._tree is None:
            from ete3 import Tree

            self._tree = Tree(self._newick)
        return self._tree


class CompactTreeBench:
    name = "compacttree"

    def __init__(self, newick):
        import tempfile

        tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".nwk", delete=False
        )
        tmpfile.write(newick)
        tmpfile.close()
        self._tmppath = tmpfile.name
        self._tree = None

    def warmup(self):
        from CompactTree import compact_tree  # noqa: F401

    def load_newick(self):
        from CompactTree import compact_tree

        self._tree = compact_tree(self._tmppath)

    def save_newick(self):
        t = self._ensure_tree()
        t.get_newick()

    def preorder(self):
        from CompactTree import traverse_preorder

        t = self._ensure_tree()
        for _ in traverse_preorder(t):
            pass

    def postorder(self):
        from CompactTree import traverse_postorder

        t = self._ensure_tree()
        for _ in traverse_postorder(t):
            pass

    def inorder(self):
        raise NotImplementedError("inorder not available in CompactTree")

    def levelorder(self):
        from CompactTree import traverse_levelorder

        t = self._ensure_tree()
        for _ in traverse_levelorder(t):
            pass

    def topological_order(self):
        raise NotImplementedError(
            "topological_order not available in CompactTree"
        )

    def mrca_allpairs(self):
        from CompactTree import traverse_leaves

        t = self._ensure_tree()
        leaves = list(traverse_leaves(t))

        def _mrca(a, b):
            ancestors_a = set()
            while not t.is_root(a):
                ancestors_a.add(a)
                a = t.get_parent(a)
            ancestors_a.add(a)
            while b not in ancestors_a:
                b = t.get_parent(b)
            return b

        for i, a in enumerate(leaves):
            for b in leaves[i + 1 :]:
                _mrca(a, b)

    def pairwise_dist(self):
        t = self._ensure_tree()
        t.calc_distance_matrix()

    def memory_bytes(self):
        tmpname = self._tmppath

        def _load():
            from CompactTree import compact_tree

            return compact_tree(tmpname)

        return _measure_memory(_load)

    def _ensure_tree(self):
        if self._tree is None:
            from CompactTree import compact_tree

            self._tree = compact_tree(self._tmppath)
        return self._tree


class ScikitBioBench:
    name = "scikit-bio"

    def __init__(self, newick):
        self._newick = newick
        self._tree = None

    def warmup(self):
        from skbio import TreeNode  # noqa: F401

        tiny = _balanced_newick(8)
        TreeNode.read(io.StringIO(tiny), format="newick")

    def load_newick(self):
        from skbio import TreeNode

        self._tree = TreeNode.read(io.StringIO(self._newick), format="newick")

    def save_newick(self):
        t = self._ensure_tree()
        buf = io.StringIO()
        t.write(buf, format="newick")

    def preorder(self):
        t = self._ensure_tree()
        for _ in t.preorder():
            pass

    def postorder(self):
        t = self._ensure_tree()
        for _ in t.postorder():
            pass

    def inorder(self):
        raise NotImplementedError("inorder not available in scikit-bio")

    def levelorder(self):
        t = self._ensure_tree()
        for _ in t.levelorder():
            pass

    def topological_order(self):
        raise NotImplementedError(
            "topological_order not available in scikit-bio"
        )

    def mrca_allpairs(self):
        t = self._ensure_tree()
        tips = list(t.tips())
        for i, a in enumerate(tips):
            for b in tips[i + 1 :]:
                t.lca([a, b])

    def pairwise_dist(self):
        t = self._ensure_tree()
        t.cophenet()

    def memory_bytes(self):
        newick = self._newick

        def _load():
            from skbio import TreeNode

            return TreeNode.read(io.StringIO(newick), format="newick")

        return _measure_memory(_load)

    def _ensure_tree(self):
        if self._tree is None:
            from skbio import TreeNode

            self._tree = TreeNode.read(
                io.StringIO(self._newick), format="newick"
            )
        return self._tree


LIBRARIES = [
    PhyloframeInMemoryBench,
    PhyloframeStreamingInt32Bench,
    PhyloframeStreamingInt32ChildSibBench,
    PhyloframeStreamingInt32CsrBench,
    CompactTreeBench,
    ScikitBioBench,
    TreeswiftBench,
    BiopythonBench,
    DendropyBench,
    EteBench,
]


def run_benchmarks(sizes=None):
    if sizes is None:
        sizes = SIZES
    results = []
    # Track ops that failed/timed out per library — skip on larger sizes.
    skip_ops = {}  # (LibClass, op) -> True
    # Track libraries that failed the probe — skip entirely.
    skip_libs = set()
    op_col_w = max(len(op) for op in OPERATIONS)
    for n_leaves in sizes:
        print(
            f"\n{'=' * 60}\n"
            f"  TREE: {n_leaves:>12,} leaves\n"
            f"{'=' * 60}",
            file=sys.stderr,
        )
        newick = _balanced_newick(n_leaves)
        print(
            f"  newick length: {len(newick):,} chars",
            file=sys.stderr,
        )

        # Generate a 100x smaller probe tree for the preamble test.
        probe_n = max(4, n_leaves // 100)
        probe_newick = _balanced_newick(probe_n)

        for LibClass in LIBRARIES:
            print(
                f"\n  {'-' * 56}\n"
                f"  | {LibClass.name}\n"
                f"  {'-' * 56}",
                file=sys.stderr,
            )

            if LibClass in skip_libs:
                print(
                    f"  | {'SKIP':<{op_col_w}}   (probe failed)",
                    file=sys.stderr,
                )
                for op in OPERATIONS:
                    results.append(
                        {
                            "library": LibClass.name,
                            "n_leaves": n_leaves,
                            "operation": op,
                            "seconds": None,
                            "status": "SKIP",
                        }
                    )
                continue

            # ── Preamble: probe with 100x smaller tree ──────────────
            _, probe_status = timed(
                LibClass,
                probe_newick,
                "load_newick",
                timeout=TIMEOUT_PROBE,
            )
            if probe_status != "SUCCESS":
                print(
                    f"  | probe ({probe_n:,} leaves) {probe_status}"
                    " -- skipping library",
                    file=sys.stderr,
                )
                skip_libs.add(LibClass)
                for op in OPERATIONS:
                    results.append(
                        {
                            "library": LibClass.name,
                            "n_leaves": n_leaves,
                            "operation": op,
                            "seconds": None,
                            "status": "SKIP",
                        }
                    )
                continue
            print(
                f"  | {'probe':<{op_col_w}}   ok ({probe_n:,} leaves)",
                file=sys.stderr,
            )

            # ── Load newick (100s timeout) ──────────────────────────
            load_val, load_status = timed(
                LibClass,
                newick,
                "load_newick",
                timeout=TIMEOUT_LOAD,
            )
            skip_remaining = load_status != "SUCCESS"
            if skip_remaining:
                skip_ops[(LibClass, "load_newick")] = True
            if load_status == "SUCCESS":
                print(
                    f"  | {'load_newick':<{op_col_w}}   {load_val:.4f}s",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  | {'load_newick':<{op_col_w}}   {load_status}",
                    file=sys.stderr,
                )
            results.append(
                {
                    "library": LibClass.name,
                    "n_leaves": n_leaves,
                    "operation": "load_newick",
                    "seconds": load_val,
                    "status": load_status,
                }
            )

            # ── Remaining operations (210s timeout) ─────────────────
            for op in OPERATIONS:
                if op == "load_newick":
                    continue  # already handled above

                fn = getattr(LibClass, op, None)
                if fn is None:
                    value, status = None, "UNAVAILABLE"
                elif skip_remaining or (LibClass, op) in skip_ops:
                    value, status = None, "SKIP"
                elif op == "memory_bytes":
                    value, status = measure_memory(
                        LibClass,
                        newick,
                        timeout=TIMEOUT_OP,
                    )
                else:
                    value, status = timed(
                        LibClass,
                        newick,
                        op,
                        timeout=TIMEOUT_OP,
                    )

                # Remember non-success so larger sizes skip this op.
                if status in ("TIMEOUT", "FAIL", "UNAVAILABLE"):
                    skip_ops[(LibClass, op)] = True

                if status == "SUCCESS" and op == "memory_bytes":
                    label = f"{value:,} B"
                elif status == "SUCCESS":
                    label = f"{value:.4f}s"
                else:
                    label = status
                print(
                    f"  | {op:<{op_col_w}}   {label}",
                    file=sys.stderr,
                )
                results.append(
                    {
                        "library": LibClass.name,
                        "n_leaves": n_leaves,
                        "operation": op,
                        "seconds": value,
                        "status": status,
                    }
                )

    return results


def print_summary_table(results):
    """Print a pivoted summary table to stderr."""
    import pandas as pd

    df = pd.DataFrame(results)
    df["seconds"] = pd.to_numeric(df["seconds"])
    for n_leaves, grp in df.groupby("n_leaves"):
        pivot = grp.pivot(
            index="operation", columns="library", values="seconds"
        )
        # reorder columns to match LIBRARIES order
        lib_order = [L.name for L in LIBRARIES]
        pivot = pivot[[c for c in lib_order if c in pivot.columns]]
        header = f" SUMMARY: {n_leaves:,} leaves "
        pad = max(0, 60 - len(header))
        banner = "=" * (pad // 2) + header + "=" * (pad - pad // 2)
        print(f"\n{banner}", file=sys.stderr)
        with pd.option_context(
            "display.float_format",
            "{:.6f}".format,
            "display.max_columns",
            20,
            "display.width",
            200,
        ):
            print(pivot.to_string(na_rep="--"), file=sys.stderr)
        print("=" * 60, file=sys.stderr)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run phylo benchmarks")
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Run benchmark for a single tree size (n_leaves).",
    )
    args = parser.parse_args()

    sizes = [args.size] if args.size is not None else None
    results = run_benchmarks(sizes=sizes)
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=["library", "n_leaves", "operation", "seconds", "status"],
    )
    writer.writeheader()
    writer.writerows(results)
    print_summary_table(results)


if __name__ == "__main__":
    main()

import json
import os

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_as_taxonium_jsonl,
    alifestd_from_newick,
    alifestd_from_taxonium_jsonl,
    alifestd_try_add_ancestor_id_col,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def test_empty():
    result = alifestd_from_taxonium_jsonl("")
    assert len(result) == 0
    assert "id" in result.columns
    assert "ancestor_id" in result.columns
    assert "taxon_label" in result.columns
    assert "origin_time_delta" in result.columns
    assert "branch_length" in result.columns
    assert "ancestor_list" not in result.columns


def test_empty_with_ancestor_list():
    result = alifestd_from_taxonium_jsonl("", create_ancestor_list=True)
    assert "ancestor_list" in result.columns


def test_header_only():
    header = json.dumps(
        {
            "version": "0.0.0",
            "mutations": [],
            "total_nodes": 0,
            "config": {"num_tips": 0},
        }
    )
    result = alifestd_from_taxonium_jsonl(header)
    assert len(result) == 0


def test_single_node():
    header = json.dumps(
        {
            "version": "0.0.0",
            "mutations": [],
            "total_nodes": 1,
            "config": {"num_tips": 1},
        }
    )
    node = json.dumps(
        {
            "name": "root",
            "x_dist": 0.0,
            "y": 0,
            "mutations": [],
            "is_tip": True,
            "parent_id": 0,
            "node_id": 0,
            "num_tips": 1,
        }
    )
    result = alifestd_from_taxonium_jsonl(f"{header}\n{node}")
    assert len(result) == 1
    assert result["id"].iloc[0] == 0
    assert result["ancestor_id"].iloc[0] == 0  # root is own ancestor
    assert result["taxon_label"].iloc[0] == "root"


def test_simple_tree():
    header = json.dumps(
        {
            "version": "0.0.0",
            "mutations": [],
            "total_nodes": 3,
            "config": {"num_tips": 2},
        }
    )
    nodes = [
        {
            "name": "A",
            "x_dist": 10.0,
            "y": 0,
            "mutations": [],
            "is_tip": True,
            "parent_id": 2,
            "node_id": 0,
            "num_tips": 1,
        },
        {
            "name": "B",
            "x_dist": 20.0,
            "y": 1,
            "mutations": [],
            "is_tip": True,
            "parent_id": 2,
            "node_id": 1,
            "num_tips": 1,
        },
        {
            "name": "root",
            "x_dist": 0.0,
            "y": 0.5,
            "mutations": [],
            "is_tip": False,
            "parent_id": 2,
            "node_id": 2,
            "num_tips": 2,
        },
    ]
    jsonl = header + "\n" + "\n".join(json.dumps(n) for n in nodes)
    result = alifestd_from_taxonium_jsonl(jsonl)

    assert len(result) == 3

    root = result[result["taxon_label"] == "root"]
    a = result[result["taxon_label"] == "A"]
    b = result[result["taxon_label"] == "B"]

    assert len(root) == 1
    assert len(a) == 1
    assert len(b) == 1

    # root is its own ancestor
    assert root["ancestor_id"].iloc[0] == root["id"].iloc[0]

    # A and B have root as ancestor
    assert a["ancestor_id"].iloc[0] == root["id"].iloc[0]
    assert b["ancestor_id"].iloc[0] == root["id"].iloc[0]


def test_branch_lengths_from_x_dist():
    """Branch lengths should be computed from x_dist differences."""
    header = json.dumps(
        {
            "version": "0.0.0",
            "mutations": [],
            "total_nodes": 3,
            "config": {"num_tips": 2},
        }
    )
    nodes = [
        {
            "name": "A",
            "x_dist": 30.0,
            "y": 0,
            "mutations": [],
            "is_tip": True,
            "parent_id": 2,
            "node_id": 0,
            "num_tips": 1,
        },
        {
            "name": "B",
            "x_dist": 50.0,
            "y": 1,
            "mutations": [],
            "is_tip": True,
            "parent_id": 2,
            "node_id": 1,
            "num_tips": 1,
        },
        {
            "name": "root",
            "x_dist": 10.0,
            "y": 0.5,
            "mutations": [],
            "is_tip": False,
            "parent_id": 2,
            "node_id": 2,
            "num_tips": 2,
        },
    ]
    jsonl = header + "\n" + "\n".join(json.dumps(n) for n in nodes)
    result = alifestd_from_taxonium_jsonl(jsonl)

    a = result[result["taxon_label"] == "A"]
    b = result[result["taxon_label"] == "B"]
    root = result[result["taxon_label"] == "root"]

    assert a["branch_length"].iloc[0] == pytest.approx(20.0)  # 30 - 10
    assert b["branch_length"].iloc[0] == pytest.approx(40.0)  # 50 - 10
    assert np.isnan(root["branch_length"].iloc[0])  # root has no branch


def test_origin_time_delta_matches_branch_length():
    header = json.dumps(
        {
            "version": "0.0.0",
            "mutations": [],
            "total_nodes": 2,
            "config": {"num_tips": 1},
        }
    )
    nodes = [
        {
            "name": "child",
            "x_dist": 5.0,
            "y": 0,
            "mutations": [],
            "is_tip": True,
            "parent_id": 1,
            "node_id": 0,
            "num_tips": 1,
        },
        {
            "name": "root",
            "x_dist": 0.0,
            "y": 0.5,
            "mutations": [],
            "is_tip": False,
            "parent_id": 1,
            "node_id": 1,
            "num_tips": 1,
        },
    ]
    jsonl = header + "\n" + "\n".join(json.dumps(n) for n in nodes)
    result = alifestd_from_taxonium_jsonl(jsonl)

    child = result[result["taxon_label"] == "child"]
    assert child["branch_length"].iloc[0] == child["origin_time_delta"].iloc[0]


def test_ancestor_list():
    header = json.dumps(
        {
            "version": "0.0.0",
            "mutations": [],
            "total_nodes": 2,
            "config": {"num_tips": 1},
        }
    )
    nodes = [
        {
            "name": "child",
            "x_dist": 5.0,
            "y": 0,
            "mutations": [],
            "is_tip": True,
            "parent_id": 1,
            "node_id": 0,
            "num_tips": 1,
        },
        {
            "name": "root",
            "x_dist": 0.0,
            "y": 0.5,
            "mutations": [],
            "is_tip": False,
            "parent_id": 1,
            "node_id": 1,
            "num_tips": 1,
        },
    ]
    jsonl = header + "\n" + "\n".join(json.dumps(n) for n in nodes)

    result = alifestd_from_taxonium_jsonl(jsonl, create_ancestor_list=True)
    assert "ancestor_list" in result.columns
    root = result[result["taxon_label"] == "root"]
    assert root["ancestor_list"].iloc[0] == "[none]"

    result_no_al = alifestd_from_taxonium_jsonl(jsonl)
    assert "ancestor_list" not in result_no_al.columns


def test_dtype_id_default():
    header = json.dumps(
        {
            "version": "0.0.0",
            "mutations": [],
            "total_nodes": 2,
            "config": {},
        }
    )
    nodes = [
        {
            "name": "A",
            "x_dist": 1.0,
            "y": 0,
            "mutations": [],
            "is_tip": True,
            "parent_id": 1,
            "node_id": 0,
            "num_tips": 1,
        },
        {
            "name": "",
            "x_dist": 0.0,
            "y": 0.5,
            "mutations": [],
            "is_tip": False,
            "parent_id": 1,
            "node_id": 1,
            "num_tips": 1,
        },
    ]
    jsonl = header + "\n" + "\n".join(json.dumps(n) for n in nodes)
    result = alifestd_from_taxonium_jsonl(jsonl)
    assert result["id"].dtype == np.int64
    assert result["ancestor_id"].dtype == np.int64


def test_dtype_id_int32():
    header = json.dumps(
        {
            "version": "0.0.0",
            "mutations": [],
            "total_nodes": 1,
            "config": {},
        }
    )
    node = json.dumps(
        {
            "name": "A",
            "x_dist": 0.0,
            "y": 0,
            "mutations": [],
            "is_tip": True,
            "parent_id": 0,
            "node_id": 0,
            "num_tips": 1,
        }
    )
    result = alifestd_from_taxonium_jsonl(
        f"{header}\n{node}", dtype_id=np.int32
    )
    assert result["id"].dtype == np.int32
    assert result["ancestor_id"].dtype == np.int32


def test_dtype_id_none():
    header = json.dumps(
        {
            "version": "0.0.0",
            "mutations": [],
            "total_nodes": 2,
            "config": {},
        }
    )
    nodes = [
        {
            "name": "A",
            "x_dist": 1.0,
            "y": 0,
            "mutations": [],
            "is_tip": True,
            "parent_id": 1,
            "node_id": 0,
            "num_tips": 1,
        },
        {
            "name": "",
            "x_dist": 0.0,
            "y": 0.5,
            "mutations": [],
            "is_tip": False,
            "parent_id": 1,
            "node_id": 1,
            "num_tips": 1,
        },
    ]
    jsonl = header + "\n" + "\n".join(json.dumps(n) for n in nodes)
    result = alifestd_from_taxonium_jsonl(jsonl, dtype_id=None)
    assert result["id"].dtype == np.int8


def test_contiguous_ids():
    """IDs in the output should be contiguous 0..n-1."""
    phylogeny_df = alifestd_from_newick("(ant:17,(bat:31,cow:22):7,dog:22);")
    jsonl = alifestd_as_taxonium_jsonl(phylogeny_df)
    result = alifestd_from_taxonium_jsonl(jsonl)
    assert list(result["id"]) == list(range(len(result)))


def test_all_ancestor_ids_valid():
    """All ancestor_ids should reference existing ids."""
    phylogeny_df = alifestd_from_newick("((A,B),(C,D));")
    jsonl = alifestd_as_taxonium_jsonl(phylogeny_df)
    result = alifestd_from_taxonium_jsonl(jsonl)

    for _, row in result.iterrows():
        assert row["ancestor_id"] in result["id"].values


def test_single_root():
    """Roundtripped tree should have exactly one root."""
    phylogeny_df = alifestd_from_newick("((A,B),(C,D));")
    jsonl = alifestd_as_taxonium_jsonl(phylogeny_df)
    result = alifestd_from_taxonium_jsonl(jsonl)

    roots = result[result["ancestor_id"] == result["id"]]
    assert len(roots) == 1


@pytest.mark.parametrize(
    "newick",
    [
        "(A,B)root;",
        "(A:1,B:2)root:3;",
        "((A,B),(C,D));",
        "(ant:17,(bat:31,cow:22):7,dog:22);",
        "((((leaf)a)b)c)root;",
        "(A,B,C,D,E,F,G,H);",
    ],
)
def test_roundtrip_topology(newick):
    """Test roundtrip: newick -> alifestd -> taxonium -> alifestd preserves
    topology and node count."""
    original = alifestd_from_newick(newick)
    jsonl = alifestd_as_taxonium_jsonl(original, taxon_label="taxon_label")
    reconstructed = alifestd_from_taxonium_jsonl(jsonl)

    assert len(reconstructed) == len(original)

    # verify roots
    roots = reconstructed[reconstructed["ancestor_id"] == reconstructed["id"]]
    assert len(roots) == 1


@pytest.mark.parametrize(
    "newick",
    [
        "(A:5,B:10);",
        "(A:1,(B:2,C:3):4);",
    ],
)
def test_roundtrip_branch_length_ratios(newick):
    """Branch length ratios should be preserved through roundtrip."""
    original = alifestd_from_newick(newick)
    jsonl = alifestd_as_taxonium_jsonl(original, taxon_label="taxon_label")
    reconstructed = alifestd_from_taxonium_jsonl(jsonl)

    # get branch lengths for leaves by name
    orig_bls = {}
    for _, row in original.iterrows():
        if row["taxon_label"]:
            orig_bls[row["taxon_label"]] = row["branch_length"]

    recon_bls = {}
    for _, row in reconstructed.iterrows():
        if row["taxon_label"]:
            recon_bls[row["taxon_label"]] = row["branch_length"]

    # verify ratios are preserved
    leaf_names = list(orig_bls.keys())
    if len(leaf_names) >= 2:
        ref = leaf_names[0]
        for name in leaf_names[1:]:
            orig_ratio = orig_bls[name] / orig_bls[ref]
            recon_ratio = recon_bls[name] / recon_bls[ref]
            assert orig_ratio == pytest.approx(recon_ratio, rel=1e-3)


@pytest.mark.parametrize(
    "phylogeny_csv",
    [
        "example-standard-toy-asexual-phylogeny.csv",
        "nk_ecoeaselection.csv",
        "nk_lexicaseselection.csv",
        "nk_tournamentselection.csv",
    ],
)
def test_asset_roundtrip_topology(phylogeny_csv):
    """Test roundtrip with real phylogeny data."""
    phylogeny_df = pd.read_csv(f"{assets_path}/{phylogeny_csv}")
    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df)

    jsonl = alifestd_as_taxonium_jsonl(phylogeny_df, taxon_label="id")
    reconstructed = alifestd_from_taxonium_jsonl(jsonl)

    assert len(reconstructed) == len(phylogeny_df)

    # verify topology via edge matching
    taxon_labels = dict(zip(reconstructed["id"], reconstructed["taxon_label"]))
    reconstructed_edges = set()
    for _, row in reconstructed.iterrows():
        child_label = row["taxon_label"]
        if row["ancestor_id"] != row["id"]:
            parent_label = taxon_labels[row["ancestor_id"]]
            reconstructed_edges.add((int(child_label), int(parent_label)))
        else:
            reconstructed_edges.add((int(child_label), int(child_label)))

    original_edges = set()
    for _, row in phylogeny_df.iterrows():
        original_edges.add((int(row["id"]), int(row["ancestor_id"])))

    assert reconstructed_edges == original_edges


def test_taxoniumtools_compatibility():
    """Parse a JSONL string in the format produced by taxoniumtools."""
    # This mimics the output of `newick_to_taxonium` for "(ant:17,(bat:31,cow:22):7,dog:22);"
    jsonl = (
        '{"version":"2.1.24","mutations":[],"total_nodes":6,"config":{"num_tips":4,"date_created":"2026-04-11"}}\n'
        '{"name":"ant","x_dist":268.42105,"y":0,"mutations":[],"is_tip":true,"parent_id":2,"node_id":0,"num_tips":1}\n'
        '{"name":"dog","x_dist":347.36842,"y":1,"mutations":[],"is_tip":true,"parent_id":2,"node_id":1,"num_tips":1}\n'
        '{"name":"","x_dist":0.0,"y":1.25,"mutations":[],"is_tip":false,"parent_id":2,"node_id":2,"num_tips":4}\n'
        '{"name":"cow","x_dist":457.89474,"y":2,"mutations":[],"is_tip":true,"parent_id":4,"node_id":3,"num_tips":1}\n'
        '{"name":"","x_dist":110.52632,"y":2.5,"mutations":[],"is_tip":false,"parent_id":2,"node_id":4,"num_tips":2}\n'
        '{"name":"bat","x_dist":600.0,"y":3,"mutations":[],"is_tip":true,"parent_id":4,"node_id":5,"num_tips":1}\n'
    )
    result = alifestd_from_taxonium_jsonl(jsonl)

    assert len(result) == 6

    # verify topology
    roots = result[result["ancestor_id"] == result["id"]]
    assert len(roots) == 1

    # all non-root nodes should reference valid ids
    non_root = result[result["ancestor_id"] != result["id"]]
    assert non_root["ancestor_id"].isin(result["id"]).all()

    # verify leaf names are present
    labels = set(result["taxon_label"])
    assert "ant" in labels
    assert "bat" in labels
    assert "cow" in labels
    assert "dog" in labels

    # verify branch lengths are positive for non-root nodes
    for _, row in non_root.iterrows():
        assert row["branch_length"] > 0


def test_meta_fields_ignored():
    """Metadata fields (meta_*) should be ignored gracefully."""
    jsonl = (
        '{"version":"0.0.0","mutations":[],"total_nodes":1,"config":{}}\n'
        '{"name":"A","x_dist":0.0,"y":0,"mutations":[],"is_tip":true,'
        '"meta_country":"USA","meta_date":"2024-01-01",'
        '"parent_id":0,"node_id":0,"num_tips":1}\n'
    )
    result = alifestd_from_taxonium_jsonl(jsonl)
    assert len(result) == 1
    assert result["taxon_label"].iloc[0] == "A"

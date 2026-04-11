import bz2
import gzip
import lzma
import zipfile

import pytest

from phyloframe._auxlib import write_text_with_compression

_SAMPLE_TEXT = "((A:1,B:2)C:3);\n"


# --- uncompressed writing ---


def test_write_uncompressed(tmp_path):
    out = tmp_path / "out.newick"
    write_text_with_compression(str(out), _SAMPLE_TEXT)
    assert out.read_text() == _SAMPLE_TEXT


def test_write_uncompressed_none_explicit(tmp_path):
    out = tmp_path / "out.newick"
    write_text_with_compression(str(out), _SAMPLE_TEXT, compression=None)
    assert out.read_text() == _SAMPLE_TEXT


# --- gzip ---


def test_write_gzip(tmp_path):
    out = tmp_path / "out.newick.gz"
    write_text_with_compression(str(out), _SAMPLE_TEXT, compression="gzip")
    assert out.exists()
    assert out.stat().st_size > 0
    with gzip.open(out, "rt") as f:
        assert f.read() == _SAMPLE_TEXT


def test_write_gzip_no_extension(tmp_path):
    out = tmp_path / "out.newick"
    write_text_with_compression(str(out), _SAMPLE_TEXT, compression="gzip")
    with gzip.open(out, "rt") as f:
        assert f.read() == _SAMPLE_TEXT


# --- bz2 ---


def test_write_bz2(tmp_path):
    out = tmp_path / "out.newick.bz2"
    write_text_with_compression(str(out), _SAMPLE_TEXT, compression="bz2")
    assert out.exists()
    assert out.stat().st_size > 0
    with bz2.open(out, "rt") as f:
        assert f.read() == _SAMPLE_TEXT


def test_write_bz2_no_extension(tmp_path):
    out = tmp_path / "out.newick"
    write_text_with_compression(str(out), _SAMPLE_TEXT, compression="bz2")
    with bz2.open(out, "rt") as f:
        assert f.read() == _SAMPLE_TEXT


# --- xz ---


def test_write_xz(tmp_path):
    out = tmp_path / "out.newick.xz"
    write_text_with_compression(str(out), _SAMPLE_TEXT, compression="xz")
    assert out.exists()
    assert out.stat().st_size > 0
    with lzma.open(out, "rt") as f:
        assert f.read() == _SAMPLE_TEXT


def test_write_xz_no_extension(tmp_path):
    out = tmp_path / "out.newick"
    write_text_with_compression(str(out), _SAMPLE_TEXT, compression="xz")
    with lzma.open(out, "rt") as f:
        assert f.read() == _SAMPLE_TEXT


# --- zip ---


def test_write_zip(tmp_path):
    out = tmp_path / "out.newick.zip"
    write_text_with_compression(str(out), _SAMPLE_TEXT, compression="zip")
    assert out.exists()
    assert out.stat().st_size > 0
    with zipfile.ZipFile(out, "r") as zf:
        names = zf.namelist()
        assert len(names) == 1
        assert names[0] == "out.newick"
        assert zf.read(names[0]).decode() == _SAMPLE_TEXT


def test_write_zip_no_extension(tmp_path):
    out = tmp_path / "out.newick"
    write_text_with_compression(str(out), _SAMPLE_TEXT, compression="zip")
    with zipfile.ZipFile(out, "r") as zf:
        assert zf.read(zf.namelist()[0]).decode() == _SAMPLE_TEXT


# --- extension mismatch errors ---


def test_gz_extension_no_compression(tmp_path):
    out = tmp_path / "out.newick.gz"
    with pytest.raises(ValueError, match="implies 'gzip' compression"):
        write_text_with_compression(str(out), _SAMPLE_TEXT)


def test_bz2_extension_no_compression(tmp_path):
    out = tmp_path / "out.newick.bz2"
    with pytest.raises(ValueError, match="implies 'bz2' compression"):
        write_text_with_compression(str(out), _SAMPLE_TEXT)


def test_xz_extension_no_compression(tmp_path):
    out = tmp_path / "out.newick.xz"
    with pytest.raises(ValueError, match="implies 'xz' compression"):
        write_text_with_compression(str(out), _SAMPLE_TEXT)


def test_zip_extension_no_compression(tmp_path):
    out = tmp_path / "out.newick.zip"
    with pytest.raises(ValueError, match="implies 'zip' compression"):
        write_text_with_compression(str(out), _SAMPLE_TEXT)


def test_zst_extension_no_compression(tmp_path):
    out = tmp_path / "out.newick.zst"
    with pytest.raises(ValueError, match="implies 'zstd' compression"):
        write_text_with_compression(str(out), _SAMPLE_TEXT)


def test_gz_extension_wrong_compression(tmp_path):
    out = tmp_path / "out.newick.gz"
    with pytest.raises(ValueError, match="implies 'gzip' compression"):
        write_text_with_compression(str(out), _SAMPLE_TEXT, compression="bz2")


def test_bz2_extension_wrong_compression(tmp_path):
    out = tmp_path / "out.newick.bz2"
    with pytest.raises(ValueError, match="implies 'bz2' compression"):
        write_text_with_compression(str(out), _SAMPLE_TEXT, compression="gzip")


def test_xz_extension_wrong_compression(tmp_path):
    out = tmp_path / "out.newick.xz"
    with pytest.raises(ValueError, match="implies 'xz' compression"):
        write_text_with_compression(str(out), _SAMPLE_TEXT, compression="gzip")


def test_zip_extension_wrong_compression(tmp_path):
    out = tmp_path / "out.newick.zip"
    with pytest.raises(ValueError, match="implies 'zip' compression"):
        write_text_with_compression(str(out), _SAMPLE_TEXT, compression="gzip")


def test_zst_extension_wrong_compression(tmp_path):
    out = tmp_path / "out.newick.zst"
    with pytest.raises(ValueError, match="implies 'zstd' compression"):
        write_text_with_compression(str(out), _SAMPLE_TEXT, compression="gzip")


# --- unsupported compression format ---


def test_unsupported_compression(tmp_path):
    out = tmp_path / "out.newick"
    with pytest.raises(ValueError, match="Unsupported compression"):
        write_text_with_compression(str(out), _SAMPLE_TEXT, compression="lz4")


# --- empty text ---


def test_write_gzip_empty(tmp_path):
    out = tmp_path / "out.newick.gz"
    write_text_with_compression(str(out), "", compression="gzip")
    with gzip.open(out, "rt") as f:
        assert f.read() == ""


def test_write_uncompressed_empty(tmp_path):
    out = tmp_path / "out.newick"
    write_text_with_compression(str(out), "")
    assert out.read_text() == ""

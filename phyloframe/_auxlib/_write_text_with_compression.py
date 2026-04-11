import bz2
import gzip
import logging
import lzma
import pathlib
import typing
import zipfile

_COMPRESSED_EXTENSIONS = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".xz": "xz",
    ".zip": "zip",
    ".zst": "zstd",
}


def write_text_with_compression(
    path: str,
    text: str,
    compression: typing.Optional[str] = None,
) -> None:
    """Write text to a file, optionally with compression.

    Parameters
    ----------
    path : str
        Output file path.
    text : str
        Text content to write.
    compression : str or None, optional
        Compression format: 'gzip', 'bz2', 'xz', 'zip', or None.

    Raises
    ------
    ValueError
        If the file extension implies compression but doesn't match the
        ``--compression`` flag, or if an unsupported compression format
        is requested.
    """
    path = pathlib.Path(path)
    ext = path.suffix.lower()

    if ext in _COMPRESSED_EXTENSIONS:
        expected = _COMPRESSED_EXTENSIONS[ext]
        if compression is None:
            raise ValueError(
                f"Output filename ends in '{ext}', which implies "
                f"'{expected}' compression, but --compression was not "
                f"set. Pass --compression {expected} to confirm, or use "
                f"a different filename.",
            )
        if compression != expected:
            raise ValueError(
                f"Output filename ends in '{ext}', which implies "
                f"'{expected}' compression, but --compression was set to "
                f"'{compression}'. Use a matching extension or change "
                f"--compression.",
            )

    if compression is None:
        logging.info(f"writing uncompressed output to {path}...")
        path.write_text(text)
    elif compression == "gzip":
        logging.info(f"writing gzip-compressed output to {path}...")
        with gzip.open(path, "wt") as f:
            f.write(text)
    elif compression == "bz2":
        logging.info(f"writing bz2-compressed output to {path}...")
        with bz2.open(path, "wt") as f:
            f.write(text)
    elif compression == "xz":
        logging.info(f"writing xz-compressed output to {path}...")
        with lzma.open(path, "wt") as f:
            f.write(text)
    elif compression == "zip":
        logging.info(f"writing zip-compressed output to {path}...")
        member_name = path.stem
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(member_name, text)
    else:
        raise ValueError(f"Unsupported compression format: {compression!r}")

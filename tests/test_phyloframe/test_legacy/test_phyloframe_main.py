import pathlib
import re
import subprocess

import phyloframe

legacy_path = pathlib.Path(phyloframe.__file__).parent / "legacy"


def test_phyloframe_main():
    subprocess.run(  # nosec B603
        ["python3", "-m", "phyloframe"],
        check=True,
    )


def test_phyloframe_main_lists_all_entrypoints():
    main_text = (legacy_path.parent / "__main__.py").read_text()
    listed = re.findall(r"^\$ python3 -m (\S+)$", main_text, re.MULTILINE)
    entrypoints = sorted(
        f"phyloframe.legacy.{path.stem}"
        for path in legacy_path.glob("*.py")
        if '\nif __name__ == "__main__":' in path.read_text()
    )
    assert listed == entrypoints

import logging

from ._configure_prod_logging import configure_prod_logging
from ._get_phyloframe_version import get_phyloframe_version


def begin_prod_logging() -> None:
    configure_prod_logging()
    logging.info(f"phyloframe v{get_phyloframe_version()}")

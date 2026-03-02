import logging

from phyloframe._auxlib import begin_prod_logging, get_phyloframe_version


def test_smoke():
    begin_prod_logging()


def test_logs_version(caplog):
    with caplog.at_level(logging.INFO):
        begin_prod_logging()

    assert any(
        get_phyloframe_version() in record.message for record in caplog.records
    )

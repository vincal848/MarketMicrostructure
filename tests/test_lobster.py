"""LOBSTER message/orderbook CSV parsing against a tiny hand-made fixture."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIXTURES = os.path.join(ROOT, "tests", "fixtures")

import pytest

import lobster


def test_read_messages_parses_columns_and_event_types():
    df = lobster.read_messages(os.path.join(FIXTURES, "tiny_message.csv"))
    assert list(df.columns) == lobster.MESSAGE_COLUMNS
    assert len(df) == 5
    assert df.loc[0, "type"] == 1
    assert df.loc[0, "price"] == 585000
    assert df.loc[3, "type"] == 4  # the visible execution of order 2


def test_read_orderbook_parses_n_levels():
    df = lobster.read_orderbook(
        os.path.join(FIXTURES, "tiny_orderbook.csv"), n_levels=1)
    assert list(df.columns) == ["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"]
    assert len(df) == 5
    assert df.loc[1, "ask_price_1"] == 585100
    assert df.loc[1, "bid_size_1"] == 100


def test_read_paired_checks_row_alignment(tmp_path):
    messages, book = lobster.read_paired(
        os.path.join(FIXTURES, "tiny_message.csv"),
        os.path.join(FIXTURES, "tiny_orderbook.csv"),
        n_levels=1,
    )
    assert len(messages) == len(book) == 5

    short_book = tmp_path / "short_orderbook.csv"
    short_book.write_text("9999999999,0,585000,100\n")
    with pytest.raises(ValueError):
        lobster.read_paired(
            os.path.join(FIXTURES, "tiny_message.csv"), str(short_book), n_levels=1)


def test_read_messages_rejects_an_unknown_event_type(tmp_path):
    bad = tmp_path / "bad_message.csv"
    bad.write_text("34200.1,99,1,100,585000,1\n")
    with pytest.raises(ValueError):
        lobster.read_messages(str(bad))

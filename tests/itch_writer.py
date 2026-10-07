"""Builds synthetic Nasdaq TotalView-ITCH 5.0 byte streams for tests.

Layouts follow the ITCH 5.0 specification: every message starts with a
type byte, stock locate (u16), tracking number (u16) and a 6-byte
nanoseconds-since-midnight timestamp, all big-endian. In a file, each message
is preceded by its length as a u16.
"""

import struct


def _header(kind: bytes, locate: int, ts: int) -> bytes:
    return kind + struct.pack(">HH", locate, 0) + ts.to_bytes(6, "big")


def _stock(symbol: str) -> bytes:
    return symbol.ljust(8).encode("ascii")


def frame(*messages: bytes) -> bytes:
    return b"".join(struct.pack(">H", len(m)) + m for m in messages)


def system_event(ts: int, code: str) -> bytes:
    return _header(b"S", 0, ts) + code.encode("ascii")


def stock_directory(locate: int, symbol: str, ts: int = 0) -> bytes:
    body = _stock(symbol) + b"Q" + b"N" + struct.pack(">I", 100) + b"N" + b"Q" + b"  " + b"P"
    body += b"N" + b"N" + b"1" + b"N" + struct.pack(">I", 0) + b"N"
    return _header(b"R", locate, ts) + body


def add_order(locate: int, ts: int, ref: int, side: str, shares: int, symbol: str, price: int) -> bytes:
    body = struct.pack(">QcI", ref, side.encode(), shares) + _stock(symbol) + struct.pack(">I", price)
    return _header(b"A", locate, ts) + body


def add_order_mpid(locate: int, ts: int, ref: int, side: str, shares: int, symbol: str, price: int) -> bytes:
    return add_order(locate, ts, ref, side, shares, symbol, price).replace(b"A", b"F", 1) + b"MPID"


def order_executed(locate: int, ts: int, ref: int, shares: int, match: int = 1) -> bytes:
    return _header(b"E", locate, ts) + struct.pack(">QIQ", ref, shares, match)


def order_executed_with_price(
    locate: int, ts: int, ref: int, shares: int, price: int, printable: bool, match: int = 1
) -> bytes:
    flag = b"Y" if printable else b"N"
    return _header(b"C", locate, ts) + struct.pack(">QIQ", ref, shares, match) + flag + struct.pack(">I", price)


def order_cancel(locate: int, ts: int, ref: int, shares: int) -> bytes:
    return _header(b"X", locate, ts) + struct.pack(">QI", ref, shares)


def order_delete(locate: int, ts: int, ref: int) -> bytes:
    return _header(b"D", locate, ts) + struct.pack(">Q", ref)


def order_replace(locate: int, ts: int, ref: int, new_ref: int, shares: int, price: int) -> bytes:
    return _header(b"U", locate, ts) + struct.pack(">QQII", ref, new_ref, shares, price)


def trade(locate: int, ts: int, side: str, shares: int, symbol: str, price: int) -> bytes:
    body = struct.pack(">QcI", 0, side.encode(), shares) + _stock(symbol) + struct.pack(">IQ", price, 1)
    return _header(b"P", locate, ts) + body


def stock_trading_action(locate: int, ts: int, symbol: str) -> bytes:
    """An 'H' message: present in real files, ignored by the decoder."""
    return _header(b"H", locate, ts) + _stock(symbol) + b"T" + b" " + b"    "

"""Streaming decoder for Nasdaq TotalView-ITCH 5.0, one symbol at a time.

A full-day ITCH file is a gzip of length-prefixed binary messages: a u16
length, then the message, whose first 11 bytes are always

    type (1 byte) | stock locate (u16) | tracking number (u16) | timestamp (u48)

big-endian, with the timestamp in nanoseconds since midnight. Every symbol
is assigned a stock-locate code by an 'R' (stock directory) message at the
start of the day; after that, a message's locate code alone says which
symbol it belongs to. The decoder reads in large chunks and skips other
symbols' messages after looking only at those two bytes, which is what
makes a full-day pass (hundreds of millions of messages) practical.

Decoded message types (ITCH 5.0 specification, section 1.3 onwards):

    S  system event              -> SystemEvent
    R  stock directory           -> learns the symbol's locate code
    A  add order                 -> AddOrder
    F  add order with MPID       -> AddOrder
    E  order executed            -> ExecuteOrder
    C  order executed with price -> ExecuteOrderWithPrice
    X  order cancel              -> CancelOrder
    D  order delete              -> DeleteOrder
    U  order replace             -> ReplaceOrder
    P  trade (non-cross)         -> HiddenTrade

Everything else (trading actions, crosses, NOII, ...) is skipped by length.
Prices are u32 with four implied decimals, i.e. 1/10000 of a dollar --
the same unit LOBSTER uses.
"""

from __future__ import annotations

import gzip
import os
import struct
from collections.abc import Iterator
from io import BufferedIOBase
from typing import BinaryIO

from microstructure.book import Side
from microstructure.events import (
    AddOrder,
    CancelOrder,
    DeleteOrder,
    ExecuteOrder,
    ExecuteOrderWithPrice,
    HiddenTrade,
    OrderEvent,
    ReplaceOrder,
    SystemEvent,
)

FilePath = str | os.PathLike[str]

_HEADER = 11
_REF_SIDE_SHARES = struct.Struct(">QcI")  # A, F, P at offset 11
_U32 = struct.Struct(">I")
_REF_SHARES = struct.Struct(">QI")  # E, C, X at offset 11
_REF = struct.Struct(">Q")  # D at offset 11
_REPLACE = struct.Struct(">QQII")  # U at offset 11
_SIDES = {b"B": Side.BID, b"S": Side.ASK}
_GZIP_MAGIC = b"\x1f\x8b"


def _open(source: FilePath | BinaryIO) -> BinaryIO | BufferedIOBase:
    if not isinstance(source, str | os.PathLike):
        return source
    with open(source, "rb") as probe:
        is_gzip = probe.read(2) == _GZIP_MAGIC
    if is_gzip:
        return gzip.open(source, "rb")
    return open(source, "rb")


def _decode(kind: int, msg: bytes, ts: int) -> OrderEvent | None:
    if kind in (0x41, 0x46):  # 'A', 'F'
        ref, side, shares = _REF_SIDE_SHARES.unpack_from(msg, _HEADER)
        (price,) = _U32.unpack_from(msg, 32)
        return AddOrder(ts=ts, order_id=ref, side=_SIDES[side], price=price, qty=shares)
    if kind == 0x45:  # 'E'
        ref, shares = _REF_SHARES.unpack_from(msg, _HEADER)
        return ExecuteOrder(ts=ts, order_id=ref, qty=shares)
    if kind == 0x43:  # 'C'
        ref, shares = _REF_SHARES.unpack_from(msg, _HEADER)
        (price,) = _U32.unpack_from(msg, 32)
        return ExecuteOrderWithPrice(ts=ts, order_id=ref, qty=shares, price=price, printable=msg[31] == 0x59)
    if kind == 0x58:  # 'X'
        ref, shares = _REF_SHARES.unpack_from(msg, _HEADER)
        return CancelOrder(ts=ts, order_id=ref, qty=shares)
    if kind == 0x44:  # 'D'
        (ref,) = _REF.unpack_from(msg, _HEADER)
        return DeleteOrder(ts=ts, order_id=ref)
    if kind == 0x55:  # 'U'
        ref, new_ref, shares, price = _REPLACE.unpack_from(msg, _HEADER)
        return ReplaceOrder(ts=ts, order_id=ref, new_order_id=new_ref, price=price, qty=shares)
    if kind == 0x50:  # 'P'
        _, side, shares = _REF_SIDE_SHARES.unpack_from(msg, _HEADER)
        (price,) = _U32.unpack_from(msg, 32)
        return HiddenTrade(ts=ts, resting_side=_SIDES[side], price=price, qty=shares)
    return None


def read_itch(source: FilePath | BinaryIO, symbol: str, chunk_size: int = 1 << 22) -> Iterator[OrderEvent]:
    """Yield `symbol`'s order events, plus all system events, in feed order.

    `source` is a path (gzip detected from its magic bytes) or an open
    binary stream of uncompressed ITCH. Raises ValueError if the stream ends
    mid-message or never lists `symbol` in a stock directory message.
    """
    target = symbol.ljust(8).encode("ascii")
    target_locate = -1
    stream = _open(source)
    buffer = bytearray()
    try:
        while chunk := stream.read(chunk_size):
            buffer += chunk
            pos, end = 0, len(buffer)
            while pos + 2 <= end:
                length = (buffer[pos] << 8) | buffer[pos + 1]
                start, stop = pos + 2, pos + 2 + length
                if stop > end:
                    break
                kind = buffer[start]
                locate = (buffer[start + 1] << 8) | buffer[start + 2]
                pos = stop
                if locate == target_locate or kind in (0x53, 0x52):  # matching symbol, 'S', 'R'
                    msg = bytes(buffer[start:stop])
                    ts = int.from_bytes(msg[5:11], "big")
                    if kind == 0x53:
                        yield SystemEvent(ts=ts, code=chr(msg[11]))
                    elif kind == 0x52:
                        if msg[11:19] == target:
                            target_locate = locate
                    elif (event := _decode(kind, msg, ts)) is not None:
                        yield event
            del buffer[:pos]
    finally:
        if stream is not source:
            stream.close()
    if buffer:
        raise ValueError(f"ITCH stream truncated: {len(buffer)} bytes after the last complete message")
    if target_locate < 0:
        raise ValueError(f"no stock directory ('R') message for symbol {symbol!r}")

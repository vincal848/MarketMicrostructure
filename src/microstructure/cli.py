"""`microstructure` command-line interface.

Each subcommand is a thin wrapper: parse arguments, call library code, write
a JSON result. All filesystem and console I/O for the package lives here or
in the source adapters, never in the core modules.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from microstructure.book import Depth
from microstructure.itch import read_itch
from microstructure.replay import Replayer


def _depth_json(depth: Depth) -> dict[str, list[list[int]]]:
    bids, asks = depth
    return {"bids": [list(level) for level in bids], "asks": [list(level) for level in asks]}


def _replay_itch(args: argparse.Namespace) -> int:
    replayer = Replayer(audit_continuous_only=True)
    started = time.perf_counter()
    for event in read_itch(args.path, args.symbol):
        replayer.apply(event)
    elapsed = time.perf_counter() - started

    report = replayer.report
    result: dict[str, Any] = {
        "source": str(args.path),
        "symbol": args.symbol,
        "seconds": round(elapsed, 3),
        "events_per_second": round(report.events / elapsed) if elapsed > 0 else 0,
        "clean": report.is_clean(),
        "report": {k: v for k, v in vars(report).items() if k != "max_issues_kept"},
        "final_depth": _depth_json(replayer.book.depth_snapshot(args.depth)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("symbol", "clean", "seconds", "events_per_second")}))
    return 0 if report.is_clean() else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="microstructure", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    replay_itch = commands.add_parser(
        "replay-itch", help="replay one symbol from a TotalView-ITCH 5.0 file and audit price-time priority"
    )
    replay_itch.add_argument("path", type=Path, help="ITCH 5.0 file, plain or gzip")
    replay_itch.add_argument("--symbol", required=True)
    replay_itch.add_argument("--out", type=Path, required=True, help="where to write the JSON report")
    replay_itch.add_argument("--depth", type=int, default=10, help="levels in the final depth snapshot")
    replay_itch.set_defaults(handler=_replay_itch)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.handler(args)
    return code


if __name__ == "__main__":
    sys.exit(main())

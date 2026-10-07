"""`microstructure` command-line interface.

Each subcommand is a thin wrapper: parse arguments, call library code, write
a JSON result. All filesystem and console I/O for the package lives here or
in the source adapters, never in the core modules.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from microstructure.bench import run_benchmarks
from microstructure.book import Depth
from microstructure.calibration import DecayGrid, KernelSpec, ProfiledDecay, calibrate_window, session_windows
from microstructure.databento import read_mbp10
from microstructure.experiment import build_scenario, load_config, run_experiment
from microstructure.flow import ClassifiedFlow, classify, marks
from microstructure.itch import read_itch
from microstructure.replay import Replayer, ReplayReport, depth_at
from microstructure.simulator import MarketSimulator
from microstructure.stylized import record_tape, summarize

log = logging.getLogger("microstructure")
NS_PER_MINUTE = 60 * 1_000_000_000


def _write_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def _report_json(report: ReplayReport) -> dict[str, Any]:
    return {k: v for k, v in vars(report).items() if k != "max_issues_kept"}


def _depth_json(depth: Depth) -> dict[str, list[list[int]]]:
    bids, asks = depth
    return {"bids": [list(level) for level in bids], "asks": [list(level) for level in asks]}


def _clock(text: str) -> int:
    """'HH:MM' -> nanoseconds since midnight."""
    hours, minutes = (int(part) for part in text.split(":"))
    return (hours * 60 + minutes) * NS_PER_MINUTE


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
        "report": _report_json(report),
        "final_depth": _depth_json(replayer.book.depth_snapshot(args.depth)),
    }
    _write_json(args.out, result)
    print(json.dumps({k: result[k] for k in ("symbol", "clean", "seconds", "events_per_second")}))
    return 0 if report.is_clean() else 1


def _kernel(args: argparse.Namespace) -> KernelSpec:
    if args.single_decay:
        return ProfiledDecay((args.decay_min, args.decay_max))
    return DecayGrid(tuple(float(d) for d in args.decays.split(",")))


def _calibrate(flow: ClassifiedFlow, args: argparse.Namespace, header: dict[str, Any]) -> int:
    start, end = _clock(args.start), _clock(args.end)
    session = flow.between(start, end)
    kernel = _kernel(args)
    fits = []
    for window_start, window_end in session_windows(start, end, args.minutes):
        fit = calibrate_window(flow, window_start, window_end, kernel)
        log.info(
            "window %s: %d events, decays %s/s, branching %.3f, AIC gain %.0f",
            window_start // NS_PER_MINUTE,
            fit.n_events,
            fit.hawkes.params.beta[:, 0, 0].tolist(),
            fit.branching_ratio,
            fit.aic_improvement,
        )
        fits.append(fit.summary())

    result = header | {
        "kernel": repr(kernel),
        "session": [args.start, args.end],
        "window_minutes": args.minutes,
        "classified_events_in_session": len(session),
        "unsigned_trades": flow.unsigned_trades,
        "windows": fits,
    }
    _write_json(args.out, result)
    if args.marks_out is not None:
        session_marks = marks(session)
        arrays = {f"size_{kind.name}": sizes for kind, sizes in session_marks.sizes.items()}
        arrays |= {f"distance_{kind.name}": d for kind, d in session_marks.distances.items()}
        args.marks_out.parent.mkdir(parents=True, exist_ok=True)
        # numpy types **kwds as ArrayLike but also declares a bool allow_pickle kwarg, so a
        # str -> ndarray mapping needs the ignore.
        np.savez_compressed(args.marks_out, **arrays)  # type: ignore[arg-type]
    return 0


def _calibrate_itch(args: argparse.Namespace) -> int:
    flow = classify(read_itch(args.path, args.symbol), tick=args.tick)
    log.info("classified %d events (%d unsigned trades)", len(flow), flow.unsigned_trades)
    header = {"source": str(args.path), "symbol": args.symbol, "replay": _report_json(flow.replay)}
    return _calibrate(flow, args, header)


def _calibrate_mbp10(args: argparse.Namespace) -> int:
    flow = read_mbp10(args.path, tick=args.tick)
    log.info("classified %d events (%d unsigned trades)", len(flow), flow.unsigned_trades)
    return _calibrate(flow, args, {"source": str(args.path)})


def _depth_itch(args: argparse.Namespace) -> int:
    depth = depth_at(read_itch(args.path, args.symbol), _clock(args.at), args.levels)
    _write_json(args.out, _depth_json(depth))
    log.info("book at %s: %d bid and %d ask levels", args.at, len(depth[0]), len(depth[1]))
    return 0


def _stylized_itch(args: argparse.Namespace) -> int:
    tape = record_tape(read_itch(args.path, args.symbol), _clock(args.start), _clock(args.end))
    _write_json(args.out, summarize(tape, args.tick))
    return 0


def _seed_range(text: str) -> range:
    start, stop = (int(part) for part in text.split(":"))
    return range(start, stop)


def _simulate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    scenario = build_scenario(config, horizon=args.horizon)
    generated = np.zeros(scenario.params.n_types)
    seeds = []
    for seed in _seed_range(args.seeds):
        simulator = MarketSimulator(
            scenario.params, scenario.marks, scenario.initial_depth, scenario.config(seed)
        )
        result = simulator.run()
        generated += result.generated
        seeds.append(
            {
                "seed": seed,
                "generated": result.generated.tolist(),
                "skipped": result.skipped.tolist(),
                "stylized": summarize(result.tape, scenario.tick),
            }
        )
        log.info("seed %d: %d events", seed, int(result.generated.sum()))
    observed_rate = generated / (len(seeds) * scenario.horizon)
    stationary = scenario.params.stationary_intensity()
    _write_json(
        args.out,
        {
            "config": str(args.config),
            "horizon": scenario.horizon,
            "stationary_intensity": stationary.tolist(),
            "observed_rate": observed_rate.tolist(),
            "rate_ratio": (observed_rate / stationary).tolist(),
            "seeds": seeds,
        },
    )
    return 0


def _experiment(args: argparse.Namespace) -> int:
    run = run_experiment(load_config(args.config), args.out)
    log.info("results in %s", run)
    print(run)
    return 0


def _bench(args: argparse.Namespace) -> int:
    result = run_benchmarks(scale=args.scale)
    _write_json(args.out, result)
    print(json.dumps({k: round(v) for k, v in result.items()}))
    return 0


def _add_calibration_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--start", default="10:00", help="session start, HH:MM local exchange time")
    command.add_argument("--end", default="15:30", help="session end, HH:MM local exchange time")
    command.add_argument("--minutes", type=int, default=30, help="window length")
    command.add_argument("--tick", type=int, default=100, help="tick size in 1/10000 dollars")
    command.add_argument(
        "--decays",
        default="100000,10000,1000,100,10,1",
        help="comma-separated decay grid (1/s): one exponential component per decay",
    )
    command.add_argument(
        "--single-decay", action="store_true", help="instead fit one exponential, profiling its decay"
    )
    command.add_argument("--decay-min", type=float, default=0.1, help="--single-decay lower bound (1/s)")
    command.add_argument("--decay-max", type=float, default=5000.0, help="--single-decay upper bound (1/s)")
    command.add_argument("--out", type=Path, required=True, help="where to write the JSON results")
    command.add_argument("--marks-out", type=Path, default=None, help="optional .npz of session mark samples")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="microstructure", description=__doc__)
    parser.add_argument("-q", "--quiet", action="store_true", help="only log warnings")
    commands = parser.add_subparsers(dest="command", required=True)

    replay_itch = commands.add_parser(
        "replay-itch", help="replay one symbol from a TotalView-ITCH 5.0 file and audit price-time priority"
    )
    replay_itch.add_argument("path", type=Path, help="ITCH 5.0 file, plain or gzip")
    replay_itch.add_argument("--symbol", required=True)
    replay_itch.add_argument("--out", type=Path, required=True, help="where to write the JSON report")
    replay_itch.add_argument("--depth", type=int, default=10, help="levels in the final depth snapshot")
    replay_itch.set_defaults(handler=_replay_itch)

    calibrate_itch = commands.add_parser(
        "calibrate-itch", help="classify one symbol's ITCH flow and fit Hawkes vs Poisson per window"
    )
    calibrate_itch.add_argument("path", type=Path, help="ITCH 5.0 file, plain or gzip")
    calibrate_itch.add_argument("--symbol", required=True)
    _add_calibration_arguments(calibrate_itch)
    calibrate_itch.set_defaults(handler=_calibrate_itch)

    calibrate_mbp = commands.add_parser(
        "calibrate-mbp10", help="classify a Databento MBP-10 CSV by snapshot diffs and fit per window"
    )
    calibrate_mbp.add_argument("path", type=Path, help="Databento MBP-10 CSV for one symbol and day")
    _add_calibration_arguments(calibrate_mbp)
    calibrate_mbp.set_defaults(handler=_calibrate_mbp10)

    depth_itch = commands.add_parser("depth-itch", help="write one symbol's ITCH book just before a time")
    depth_itch.add_argument("path", type=Path)
    depth_itch.add_argument("--symbol", required=True)
    depth_itch.add_argument("--at", required=True, help="HH:MM local exchange time")
    depth_itch.add_argument("--levels", type=int, default=10)
    depth_itch.add_argument("--out", type=Path, required=True)
    depth_itch.set_defaults(handler=_depth_itch)

    stylized_itch = commands.add_parser(
        "stylized-itch", help="stylized facts of one symbol's real tape in a window"
    )
    stylized_itch.add_argument("path", type=Path)
    stylized_itch.add_argument("--symbol", required=True)
    stylized_itch.add_argument("--start", required=True, help="HH:MM")
    stylized_itch.add_argument("--end", required=True, help="HH:MM")
    stylized_itch.add_argument("--tick", type=int, default=100)
    stylized_itch.add_argument("--out", type=Path, required=True)
    stylized_itch.set_defaults(handler=_stylized_itch)

    simulate = commands.add_parser(
        "simulate", help="simulate an experiment's market without agents: rates vs theory, stylized facts"
    )
    simulate.add_argument("config", type=Path, help="experiment TOML")
    simulate.add_argument("--seeds", default="0:5", help="half-open seed range start:stop")
    simulate.add_argument("--horizon", type=float, default=None, help="seconds (default: the config's)")
    simulate.add_argument("--out", type=Path, required=True)
    simulate.set_defaults(handler=_simulate)

    experiment = commands.add_parser("experiment", help="run an experiment TOML into a new run directory")
    experiment.add_argument("config", type=Path)
    experiment.add_argument("--out", type=Path, default=Path("runs"), help="root for run directories")
    experiment.set_defaults(handler=_experiment)

    bench = commands.add_parser("bench", help="throughput of the book, replay and simulator")
    bench.add_argument("--scale", type=float, default=1.0)
    bench.add_argument("--out", type=Path, required=True)
    bench.set_defaults(handler=_bench)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(message)s")
    code: int = args.handler(args)
    return code


if __name__ == "__main__":
    sys.exit(main())

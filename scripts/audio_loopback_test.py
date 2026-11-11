#!/usr/bin/env python3
"""Anchi DAC Bench — 26-Point AES17 & ITU-R Verification Suite CLI.

Examples:
  # Run all 26 automated AES17 / ITU-R benchmarks and write plots + summary_report.json:
  python3 scripts/audio_loopback_test.py --all --no-audition-wavs

  # Benchmark an external resampler / DSP engine implementing NaitDSPBench.h:
  python3 scripts/audio_loopback_test.py --all --no-audition-wavs --dylib ./my_engine.dylib

  # Interactive Day 3 blind ABX shootout on local Mac CoreAudio output:
  python3 scripts/audio_loopback_test.py --day 3 --interactive
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

# Persist Matplotlib font cache in /tmp so headless/sandboxed runs emit zero warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/anchi-mpl-cache")

# Auto-reexec into workspace .venv if invoked with system python lacking numpy/scipy/matplotlib
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

# Ensure `scripts/` is importable
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from loopback_lab.bridge import NaitDSPBenchBridge, list_coreaudio_devices  # noqa: E402
from loopback_lab.days import (  # noqa: E402
    MetricCheck,
    run_day1,
    run_day2,
    run_day3,
    run_day4,
    run_preflight_checks,
    serialize_summary_report,
)
from loopback_lab.interactive import (  # noqa: E402
    DAY_METADATA,
    ensure_taste_journal_initialized,
    print_checks_table,
    run_interactive_day,
)
from loopback_lab.synthesizer import (  # noqa: E402
    generate_day1_audition_clips,
    generate_day2_audition_clips,
    generate_day3_audition_clips,
    generate_day4_audition_clips,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anchi DAC Bench — 26-Point AES17 & ITU-R Audio Verification Suite"
    )
    parser.add_argument(
        "--day",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run a specific verification domain day (1..4).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 26 AES17 / ITU-R checks, regenerate FIG. 01-04 plots, and write summary_report.json.",
    )
    parser.add_argument(
        "--dylib",
        type=str,
        default=None,
        help="Path to libNaitDSPBench.dylib or a custom .dylib implementing NaitDSPBench.h.",
    )
    parser.add_argument(
        "--no-audition-wavs",
        action="store_true",
        help="Skip synthesizing .wav ear-training clips during automated runs.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Walk interactively through Briefing -> Measure & Plot -> Local Audition/ABX -> Notes.",
    )
    parser.add_argument(
        "--step3",
        "--listen-only",
        dest="step3",
        action="store_true",
        help="Jump straight to Step 3 (Interactive Audition & ABX) without re-running Step 2 FFT sweeps.",
    )
    parser.add_argument(
        "--audition-target",
        choices=["local"],
        default="local",
        help="Playback output for interactive WAV auditions (default: 'local' Mac CoreAudio output).",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Optional path to a local reference audio track (.wav/.flac/.aiff/.m4a) for interactive listening.",
    )
    parser.add_argument(
        "--no-open-plots",
        action="store_true",
        help="Do not auto-open PNG plots in Preview during --interactive.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List detected CoreAudio devices and exit.",
    )
    return parser.parse_args()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    if args.step3:
        args.interactive = True

    if args.list_devices:
        devs = list_coreaudio_devices()
        print("Detected macOS CoreAudio Devices:")
        for d in devs:
            print(
                f"  [{d['index']:2d}] {d['name']} "
                f"(in={d['max_input_channels']}, out={d['max_output_channels']}, sr={d['default_samplerate']})"
            )
        return 0

    days_to_run = [1, 2, 3, 4] if (args.all or args.day is None) else [args.day]

    reports_dir = REPO_ROOT / "reports" / "loopback"
    plots_dir = reports_dir / "plots"
    clips_dir = reports_dir / "audition_clips"
    summary_json_path = reports_dir / "summary_report.json"
    journal_path = reports_dir / "listening-notes.md"

    if args.interactive:
        ensure_taste_journal_initialized(journal_path)

    bridge = NaitDSPBenchBridge(dylib_path=args.dylib)

    print("================================================================================")
    print("  ANCHI DAC BENCH — 26-POINT AES17 & ITU-R VERIFICATION SUITE")
    print(f"  Engine Bridge : {_display_path(bridge.dylib_path)}")
    print(f"  Mode          : {'INTERACTIVE LAB' if args.interactive else 'AUTOMATED BENCHMARK'}")
    if args.interactive:
        print(f"  Audition Out  : {args.audition_target.upper()}")
    print("================================================================================")

    all_checks: list[MetricCheck] = []
    day_details: dict[str, object] = {}

    # Always run Pre-Flight Real-Time RingBuffer Verification (512 & 64 frames)
    preflight_checks = run_preflight_checks(bridge)
    all_checks.extend(preflight_checks)
    if not args.interactive:
        print_checks_table("Day 0 — System Real-Time Pre-Flight (512 & 64-Frame SPSC RingBuffer)", preflight_checks)

    runners = {
        1: run_day1,
        2: run_day2,
        3: run_day3,
        4: run_day4,
    }

    for day in days_to_run:
        if args.step3 and summary_json_path.exists():
            import json

            cached = json.loads(summary_json_path.read_text(encoding="utf-8"))
            checks = [MetricCheck(**c) for c in cached.get("checks", []) if c.get("day") == day]
            details = cached.get("details", {}).get(f"day{day}", {})
            if not checks:
                checks, details = runners[day](bridge, plots_dir)
        else:
            checks, details = runners[day](bridge, plots_dir)
        all_checks.extend(checks)
        day_details[f"day{day}"] = details

        if args.interactive:
            run_interactive_day(
                day=day,
                bridge=bridge,
                checks=checks,
                details=details,
                clips_dir=clips_dir,
                journal_path=journal_path,
                audition_target=args.audition_target,
                custom_track=args.track,
                open_plots=not args.no_open_plots,
                skip_to_step3=args.step3,
            )
        else:
            print_checks_table(DAY_METADATA[day]["title"], checks)
            if not args.no_audition_wavs:
                if day == 1:
                    generate_day1_audition_clips(bridge, clips_dir)
                elif day == 2:
                    generate_day2_audition_clips(clips_dir)
                elif day == 3:
                    generate_day3_audition_clips(bridge, clips_dir, args.track, None)
                elif day == 4:
                    generate_day4_audition_clips(bridge, clips_dir, args.track)

    # Merge or write summary_report.json
    if summary_json_path.exists() and len(days_to_run) < 4:
        import json

        try:
            prev = json.loads(summary_json_path.read_text(encoding="utf-8"))
            prev_checks = [
                MetricCheck(**c)
                for c in prev.get("checks", [])
                if c.get("day") not in ([0] + days_to_run)
            ]
            merged_checks = sorted(prev_checks + all_checks, key=lambda x: (x.day, x.metric_id))
            merged_details = dict(prev.get("details", {}))
            merged_details.update(day_details)
            payload = serialize_summary_report(merged_checks, merged_details, summary_json_path)
        except Exception:
            payload = serialize_summary_report(all_checks, day_details, summary_json_path)
    else:
        payload = serialize_summary_report(all_checks, day_details, summary_json_path)

    passed_n = payload["passed_checks"]
    total_n = payload["total_checks"]
    verdict = "ALL AES17 THRESHOLDS PASSED" if payload["overall_passed"] else "ONE OR MORE CHECKS FAILED"
    print("================================================================================")
    print(f"  SUMMARY REPORT : {_display_path(summary_json_path)}")
    print(f"  PLOTS DIR      : {_display_path(plots_dir)}/")
    if not args.no_audition_wavs or args.interactive:
        print(f"  AUDITION CLIPS : {_display_path(clips_dir)}/")
    if args.interactive:
        print(f"  LISTENING LOG  : {_display_path(journal_path)}")
    print(f"  FINAL VERDICT  : {passed_n}/{total_n} - {verdict}")
    print("================================================================================")

    return 0 if payload["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the whole POC end to end.

    python run_all.py              # tests, then every pipeline, from a clean slate
    python run_all.py --keep       # keep outputs/ (the model registry accumulates)
    python run_all.py --no-tests   # skip the test suite
    python run_all.py 02 03        # just these scripts

By default outputs/ is cleared first, and that is deliberate rather than tidy.
The MLflow registry and the inference log are both append-only, so a second run
against a dirty directory sees a champion from the previous run and refuses to
promote, and the inference log doubles. Clearing makes the run reproducible;
--keep is there for when you want to watch the promotion guard actually refuse.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

SCRIPTS = [
    ("01", "01_churn_batch_pipeline.py", "churn, the batch path"),
    ("02", "02_risk_realtime_pipeline.py", "risk, the real-time path"),
    ("03", "03_live_scoring_demo.py", "serving, with an outage"),
    ("04", "04_feedback_loops.py", "what acting on a prediction costs"),
    ("05", "05_monitoring_two_clocks.py", "monitoring, two clocks"),
]


def run(command: list[str], label: str) -> tuple[bool, float]:
    print(f"\n{'=' * 78}\n>>> {label}\n{'=' * 78}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT)
    return completed.returncode == 0, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("only", nargs="*", help="script numbers to run, e.g. 02 03")
    parser.add_argument("--keep", action="store_true", help="do not clear outputs/ first")
    parser.add_argument("--no-tests", action="store_true", help="skip pytest")
    args = parser.parse_args()

    selected = [s for s in SCRIPTS if not args.only or s[0] in args.only]
    if not selected:
        print(f"nothing matched {args.only}; known: {[s[0] for s in SCRIPTS]}")
        return 2

    if not args.keep and not args.only and OUTPUTS.exists():
        shutil.rmtree(OUTPUTS)
        print(f"cleared {OUTPUTS.relative_to(ROOT)}/ for a reproducible run")

    results = []
    if not args.no_tests:
        ok, seconds = run([sys.executable, "-m", "pytest", "tests/", "-q"], "tests")
        results.append(("tests", ok, seconds))
        if not ok:
            print("\ntests failed; stopping before the pipelines.")
            _report(results)
            return 1

    for number, script, description in selected:
        ok, seconds = run([sys.executable, script], f"{number}  {description}")
        results.append((script, ok, seconds))
        if not ok:
            print(f"\n{script} failed; later scripts depend on it, stopping.")
            break

    # The claims ledger and the dashboard are both derived from the metrics the
    # pipelines just wrote, so they only make sense once everything has run.
    if all(ok for _, ok, _ in results) and not args.only:
        for command, label in (
                ([sys.executable, "scripts/verify_claims.py", "--render"], "claims ledger"),
                ([sys.executable, "scripts/build_dashboard.py"], "dashboard")):
            ok, seconds = run(command, label)
            results.append((label, ok, seconds))

    return _report(results)


def _report(results) -> int:
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for name, ok, seconds in results:
        print(f"  {'ok  ' if ok else 'FAIL'}  {seconds:6.1f}s  {name}")
    total = sum(seconds for _, _, seconds in results)
    print(f"  {'':4}  {total:6.1f}s  total")

    if all(ok for _, ok, _ in results):
        print("\noutputs/\n"
              "  metrics/    one json per pipeline\n"
              "  plots/      every figure\n"
              "  tables/     scores, call list, drift history\n"
              "  inference/  the request log (the inference-tables stand-in)\n"
              "  mlruns/     MLflow tracking and the @champion aliases\n"
              "  dashboard.html  a self-contained summary of this run\n\n"
              "docs/claims.md re-rendered from the ledger.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

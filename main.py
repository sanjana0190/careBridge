"""careBridge CLI entry point.

Usage:
    python main.py <patient_uuid_or_name>     # full pipeline for one patient
    python main.py --name "lindsay brekke"    # fuzzy name lookup
    python main.py --scan                     # top-N highest-barrier patients
    python main.py --scan 5

Examples:
    python main.py 2838f1db-106d-dbfd-cf95-4af92dbb0a24
    python main.py --name "lindsay brekke"
    python main.py --name "soledad white" --no-hitl
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from carebridge.db import PatientNotFound
from carebridge.pipeline import run
from carebridge.tools import scan_high_barrier_patients


def cmd_scan(top_n: int) -> int:
    rows = scan_high_barrier_patients(top_n)
    print(f"Top {len(rows)} highest-barrier patients (composite score):")
    print("─" * 90)
    print(
        f"  {'rank':>4} {'score':>5}  {'name':<28} {'ED':>4} {'chronic':>7} "
        f"{'careplan':>8}  {'debt':>10}"
    )
    print("─" * 90)
    for i, r in enumerate(rows, start=1):
        name = f"{r.get('first', '')} {r.get('last', '')}"[:28]
        print(
            f"  {i:>4} {r['barrier_score']:>5}  {name:<28} "
            f"{int(r.get('ed_visits') or 0):>4} "
            f"{int(r.get('chronic_condition_count') or 0):>7} "
            f"{('yes' if r.get('has_active_careplan') else 'no'):>8}  "
            f"${float(r.get('total_outstanding') or 0):>9,.0f}"
        )
    print("─" * 90)
    print("\nRun the full pipeline for one of them with:")
    if rows:
        print(f"  python main.py {rows[0]['id']}")
    return 0


def cmd_run(patient: str, *, interactive: bool) -> int:
    try:
        run(patient, interactive_hitl=interactive)
    except PatientNotFound as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="carebridge",
        description="Barrier-informed care plan agent",
    )
    p.add_argument(
        "patient",
        nargs="?",
        help="Patient UUID, or fuzzy name (use --name for clarity)",
    )
    p.add_argument("--name", help="Fuzzy name lookup, e.g. 'lindsay brekke'")
    p.add_argument(
        "--scan",
        nargs="?",
        const=10,
        type=int,
        metavar="N",
        help="Rank the top-N highest-barrier patients (default 10)",
    )
    p.add_argument(
        "--no-hitl",
        action="store_true",
        help="Skip the interactive coordinator-questions step",
    )
    args = p.parse_args(argv)

    if args.scan is not None:
        return cmd_scan(args.scan)

    target = args.name or args.patient
    if not target:
        p.print_help()
        return 1

    return cmd_run(target, interactive=not args.no_hitl)


if __name__ == "__main__":
    sys.exit(main())

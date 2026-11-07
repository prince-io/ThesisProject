"""Prepare and verify the unlabeled 2k ACDC test set for demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from litevit.datasets.acdc_test import prepare_acdc_test, verify_acdc_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the unlabeled ACDC test set")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--out", type=Path, default=Path("data/unlabeled/acdc_test")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("runs/dataset_checks/acdc_test")
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out if args.out.is_absolute() else root / args.out
    report_dir = args.report if args.report.is_absolute() else root / args.report
    stats = prepare_acdc_test(root / "ref" / "acdc", out_dir, force=args.force)
    report = verify_acdc_test(out_dir, report_dir)
    print(json.dumps({"stats": stats, "report": report}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

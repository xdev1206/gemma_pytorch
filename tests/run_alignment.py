"""Run all alignment tests and write a machine-readable test report."""

import argparse
import sys
import time
import unittest
from pathlib import Path

from tests.alignment_report import AlignmentTestResult, build_report, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("tests/reports"),
        help="directory for alignment_report.json and alignment_report.md",
    )
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.discover("tests", pattern="test*.py")
    started = time.perf_counter()
    runner = unittest.TextTestRunner(
        stream=sys.stdout, verbosity=2, resultclass=AlignmentTestResult
    )
    result = runner.run(suite)
    report = build_report(result, time.perf_counter() - started)
    json_path, markdown_path = write_report(report, args.report_dir)
    print(f"Alignment report: {markdown_path}")
    print(f"Alignment report JSON: {json_path}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

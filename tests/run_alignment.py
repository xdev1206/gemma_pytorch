"""Run all alignment tests and write a machine-readable test report."""

import argparse
import os
import platform
import secrets
import sys
import time
import unittest
from pathlib import Path

from tests.alignment_report import AlignmentTestResult, build_report, write_report


PROFILES = {
    "contract": (
        "tests.test_manifest",
        "tests.test_alignment",
        "tests.test_checkpoint_loading",
    ),
    "runtime-smoke": ("tests.test_runtime_alignment",),
    "gemma4-reference": ("tests.test_gemma4_reference_alignment",),
    "all": (
        "tests.test_manifest",
        "tests.test_alignment",
        "tests.test_checkpoint_loading",
        "tests.test_runtime_alignment",
        "tests.test_gemma4_reference_alignment",
    ),
}


def _load_profile(profile: str) -> unittest.TestSuite:
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for module in PROFILES[profile]:
        suite.addTests(loader.loadTestsFromName(module))
    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("tests/reports"),
        help="directory for alignment_report.json and alignment_report.md",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="all",
        help="test scope; runtime and reference profiles are strict",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when any test is skipped (implied by runtime/reference profiles)",
    )
    args = parser.parse_args()

    strict = args.strict or args.profile in {"runtime-smoke", "gemma4-reference"}
    suite = _load_profile(args.profile)
    started = time.perf_counter()
    runner = unittest.TextTestRunner(
        stream=sys.stdout, verbosity=2, resultclass=AlignmentTestResult
    )
    result = runner.run(suite)
    runtime_value = os.environ.get("GEMMA_RUN_RUNTIME_ALIGNMENT", "")
    if "key" in runtime_value.lower():
        runtime_value = secrets.token_hex(16)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime_alignment_enabled": runtime_value == "1",
    }
    report = build_report(
        result,
        time.perf_counter() - started,
        profile=args.profile,
        strict=strict,
        metadata=metadata,
    )
    json_path, markdown_path = write_report(report, args.report_dir)
    print(f"Alignment report: {markdown_path}")
    print(f"Alignment report JSON: {json_path}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

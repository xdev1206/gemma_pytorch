"""Result collection and report writers for alignment test runs."""

import json
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path


class AlignmentTestResult(unittest.TextTestResult):
    """A unittest result that keeps one serializable record per test."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self._started_at = {}

    def startTest(self, test):
        self._started_at[test.id()] = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):
        record = next(item for item in self.records if item["id"] == test.id())
        record["duration_seconds"] = round(
            time.perf_counter() - self._started_at.pop(test.id()), 3
        )
        super().stopTest(test)

    def _record(self, test, status, detail=None):
        record = {"id": test.id(), "status": status}
        if detail:
            record["detail"] = detail
        self.records.append(record)

    def addSuccess(self, test):
        self._record(test, "passed")
        super().addSuccess(test)

    def addFailure(self, test, err):
        self._record(test, "failed", self._exc_info_to_string(err, test))
        super().addFailure(test, err)

    def addError(self, test, err):
        self._record(test, "error", self._exc_info_to_string(err, test))
        super().addError(test, err)

    def addSkip(self, test, reason):
        self._record(test, "skipped", reason)
        super().addSkip(test, reason)


def build_report(result: AlignmentTestResult, duration_seconds: float) -> dict:
    """Convert a unittest result into a stable, machine-readable report."""
    counts = {status: 0 for status in ("passed", "failed", "error", "skipped")}
    for record in result.records:
        counts[record["status"]] += 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "success": result.wasSuccessful(),
        "duration_seconds": round(duration_seconds, 3),
        "summary": {
            "total": result.testsRun,
            **counts,
        },
        "tests": result.records,
    }


def _markdown_report(report: dict) -> str:
    summary = report["summary"]
    status = "PASS" if report["success"] else "FAIL"
    lines = [
        "# Alignment Test Report",
        "",
        f"- Status: **{status}**",
        f"- Generated: `{report['generated_at']}`",
        f"- Duration: `{report['duration_seconds']:.3f}s`",
        (
            "- Summary: "
            f"{summary['passed']} passed, {summary['failed']} failed, "
            f"{summary['error']} errors, {summary['skipped']} skipped "
            f"({summary['total']} total)"
        ),
        "",
        "## Test Results",
        "",
        "| Test | Status | Duration | Detail |",
        "| --- | --- | ---: | --- |",
    ]
    for record in report["tests"]:
        detail = record.get("detail", "").replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| `{record['id']}` | {record['status']} | "
            f"{record.get('duration_seconds', 0):.3f}s | {detail} |"
        )
    return "\n".join(lines) + "\n"


def write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown reports and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "alignment_report.json"
    markdown_path = output_dir / "alignment_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path

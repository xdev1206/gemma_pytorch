import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.alignment_report import build_report, write_report


class AlignmentReportTest(unittest.TestCase):
    def test_report_writes_summary_and_test_records(self):
        result = unittest.TestResult()
        result.testsRun = 2
        result.records = [
            {"id": "case.pass", "status": "passed", "duration_seconds": 0.1},
            {"id": "case.skip", "status": "skipped", "detail": "model missing"},
        ]
        report = build_report(result, 0.2)

        self.assertTrue(report["success"])
        self.assertEqual(report["summary"]["total"], 2)
        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(report["summary"]["skipped"], 1)
        with TemporaryDirectory() as temporary_dir:
            json_path, markdown_path = write_report(report, Path(temporary_dir))
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            self.assertIn("case.pass", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from yuyi_eval.eval import summarize_results as summary


class SummarizeResultsTests(unittest.TestCase):
    def test_summarize_file_computes_dimension_averages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.jsonl"
            records = [
                {
                    "eval_method": "direct_llm",
                    "source_model": "source",
                    "evaluator_model": "judge",
                    "with_target": True,
                    "scores": {
                        "issue": 1.0,
                        "method": 1.0,
                        "stance": 1.0,
                        "length": 1.0,
                        "overall": 1.0,
                    },
                },
                {
                    "eval_method": "direct_llm",
                    "source_model": "source",
                    "evaluator_model": "judge",
                    "with_target": True,
                    "scores": {
                        "issue": 0.0,
                        "method": 0.5,
                        "stance": 0.0,
                        "length": 1.0,
                        "overall": 0.375,
                    },
                },
            ]

            with path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            row = summary.summarize_file(path, threshold=0.8)

        self.assertEqual(row["n"], 2)
        self.assertEqual(row["issue"], 0.5)
        self.assertEqual(row["method"], 0.75)
        self.assertEqual(row["stance"], 0.5)
        self.assertEqual(row["length"], 1.0)
        self.assertEqual(row["overall"], 0.6875)
        self.assertEqual(row["pass_rate"], 0.5)
        self.assertEqual(row["worst_dim"], "issue")
        self.assertEqual(row["gap"], 0.5)

    def test_summarize_results_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.jsonl"
            output_path = root / "summary.csv"
            input_path.write_text(
                json.dumps(
                    {
                        "eval_method": "multi_agent",
                        "source_model": "source",
                        "evaluator_model": "judge",
                        "with_target": True,
                        "scores": {
                            "issue": 1.0,
                            "method": 1.0,
                            "stance": 1.0,
                            "length": 1.0,
                            "overall": 1.0,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = summary.summarize_results(
                inputs=[str(input_path)],
                output_path=output_path,
            )

            content = result.read_text(encoding="utf-8")

        self.assertEqual(result, output_path)
        self.assertIn("multi_agent", content)
        self.assertIn("pass_rate", content)
        self.assertIn("1.0", content)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest import mock

from yuyi_eval.eval import evaluate_comments as evaluation


class MultiAgentEvaluationTests(unittest.TestCase):
    def test_multi_agent_returns_all_reviewer_records(self) -> None:
        guidance = {
            "issue": "测试议题",
            "propaganda_method": "重复",
            "stance": {
                "label": "支持",
                "target": "测试议题",
            },
        }

        with mock.patch.object(
            evaluation,
            "chat_completion",
            return_value=(
                '{"issue": 1, "method": 0.5, "stance": 1}',
                {"total_tokens": 10},
            ),
        ):
            scores, usage, details = evaluation.evaluate_semantic_scores(
                guidance=guidance,
                generated_text="测试文本",
                evaluator="DeepSeek-R1-Distill-Qwen-14B",
                base_url="http://127.0.0.1:1/v1",
                with_target=True,
                method="multi_agent",
            )

        self.assertEqual(scores["issue"], 1.0)
        self.assertIn("adjudicator", usage)
        self.assertIsNotNone(details)
        self.assertEqual(len(details["agents"]), 3)


if __name__ == "__main__":
    unittest.main()

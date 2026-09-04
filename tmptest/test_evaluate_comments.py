from __future__ import annotations

import io
import unittest
import urllib.error
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

    def test_eval_chat_completion_uses_small_token_budget(self) -> None:
        captured = {}

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"'
                    b'{\\"issue\\": 1, \\"method\\": 1, \\"stance\\": 1}'
                    b'"}}],"usage":{"total_tokens":12}}'
                )

        def fake_urlopen(request, timeout):
            captured["body"] = request.data
            captured["timeout"] = timeout
            return FakeResponse()

        with mock.patch.object(
            evaluation.urllib.request,
            "urlopen",
            side_effect=fake_urlopen,
        ):
            text, usage = evaluation.chat_completion(
                base_url="http://127.0.0.1:1/v1",
                model="Qwen2.5-1.5B-Instruct",
                prompt="测试",
            )

        self.assertIn('"max_tokens": 128', captured["body"].decode("utf-8"))
        self.assertEqual(captured["timeout"], 300)
        self.assertIn('"issue"', text)
        self.assertEqual(usage["total_tokens"], 12)

    def test_eval_chat_completion_uses_default_token_budget_for_large_models(self) -> None:
        captured = {}

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"'
                    b'{\\"issue\\": 1, \\"method\\": 1, \\"stance\\": 1}'
                    b'"}}],"usage":{"total_tokens":12}}'
                )

        def fake_urlopen(request, timeout):
            captured["body"] = request.data
            return FakeResponse()

        with mock.patch.object(
            evaluation.urllib.request,
            "urlopen",
            side_effect=fake_urlopen,
        ):
            evaluation.chat_completion(
                base_url="http://127.0.0.1:1/v1",
                model="DeepSeek-R1-Distill-Qwen-32B",
                prompt="测试",
            )

        self.assertIn('"max_tokens": 1024', captured["body"].decode("utf-8"))

    def test_qwen3_evaluator_request_disables_thinking(self) -> None:
        captured = {}

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"'
                    b'{\\"issue\\": 1, \\"method\\": 1, \\"stance\\": 1}'
                    b'"}}],"usage":{"total_tokens":12}}'
                )

        def fake_urlopen(request, timeout):
            captured["body"] = request.data
            return FakeResponse()

        with mock.patch.object(
            evaluation.urllib.request,
            "urlopen",
            side_effect=fake_urlopen,
        ):
            evaluation.chat_completion(
                base_url="http://127.0.0.1:1/v1",
                model="Qwen3-32B",
                prompt="测试",
            )

        body = captured["body"].decode("utf-8")
        self.assertIn("/no_think", body)
        self.assertIn("语义一致性评价器", body)

    def test_empty_content_response_raises_clear_error(self) -> None:
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":null,'
                    b'"reasoning_content":"thinking"}}],"usage":{}}'
                )

        with (
            mock.patch.object(
                evaluation.urllib.request,
                "urlopen",
                return_value=FakeResponse(),
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            evaluation.chat_completion(
                base_url="http://127.0.0.1:1/v1",
                model="DeepSeek-R1-Distill-Qwen-32B",
                prompt="测试",
            )

        message = str(raised.exception)
        self.assertIn("空 content", message)
        self.assertIn("reasoning_content", message)

    def test_http_error_includes_vllm_response_body(self) -> None:
        error = urllib.error.HTTPError(
            url="http://127.0.0.1:1/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=io.BytesIO(
                b'{"error":"prompt plus max_tokens exceeds max_model_len"}'
            ),
        )

        with (
            mock.patch.object(
                evaluation.urllib.request,
                "urlopen",
                side_effect=error,
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            evaluation.chat_completion(
                base_url="http://127.0.0.1:1/v1",
                model="Qwen2.5-1.5B-Instruct",
                prompt="测试",
            )

        message = str(raised.exception)
        self.assertIn("HTTP 400", message)
        self.assertIn("Qwen2.5-1.5B-Instruct", message)
        self.assertIn("prompt plus max_tokens", message)

    def test_multi_agent_output_paths_are_descriptive(self) -> None:
        final_path, detail_path = evaluation.build_output_paths(
            method="multi_agent",
            source_model="DeepSeek-R1-Distill-Qwen-32B",
            evaluator="Qwen2.5-1.5B-Instruct",
            mode="target",
        )

        self.assertIn("multi_agent_final_scores", final_path.name)
        self.assertIn("multi_agent_role_details", detail_path.name)
        self.assertEqual(final_path.parent.as_posix(), "data/evaluation")
        self.assertEqual(
            detail_path.parent.as_posix(),
            "data/evaluation_details",
        )


if __name__ == "__main__":
    unittest.main()

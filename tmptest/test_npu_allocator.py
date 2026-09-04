from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm_service import model_service as ms


NPU_SMI_SAMPLE = """
| 1     910B4-1             | OK            | 79.4                 54                      0    / 0                |
| 0                         | 0000:0E:00.0  | 0                    0    / 0                58408/ 65536            |
| 7     910B4-1             | OK            | 79.0                 52                      0    / 0                |
| 0                         | 0000:03:00.0  | 0                    0    / 0                3401 / 65536            |
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
| 1       0                 | 3774161       | VLLMWorker_TP      | 55048                 | NA                      |
| No running processes found in NPU 7                                                                              |
"""


TOPOLOGY_SAMPLE = """
           NPU1       NPU3       NPU5       NPU7       NPU8       NPU9       NPU10      NPU11      CPU Affinity
NPU1       X          PIX        PIX        PIX        PHB        PHB        PHB        PHB        0-23
NPU3       PIX        X          PIX        PIX        PHB        PHB        PHB        PHB        0-23
NPU5       PIX        PIX        X          PIX        PHB        PHB        PHB        PHB        0-23
NPU7       PIX        PIX        PIX        X          PHB        PHB        PHB        PHB        0-23
NPU8       PHB        PHB        PHB        PHB        X          PIX        PIX        PIX        0-23
NPU9       PHB        PHB        PHB        PHB        PIX        X          PIX        PIX        0-23
NPU10      PHB        PHB        PHB        PHB        PIX        PIX        X          PIX        0-23
NPU11      PHB        PHB        PHB        PHB        PIX        PIX        PIX        X          0-23
"""


class NPUAllocatorTests(unittest.TestCase):
    def test_parse_real_server_output(self) -> None:
        npus = {npu.device_id: npu for npu in ms.parse_npu_smi(NPU_SMI_SAMPLE)}

        self.assertEqual(npus[1].free_hbm_mb, 7128)
        self.assertTrue(npus[1].has_process)
        self.assertTrue(npus[1].healthy)
        self.assertEqual(npus[7].free_hbm_mb, 62135)
        self.assertFalse(npus[7].has_process)

    def test_required_memory_has_vllm_gate_and_reserve(self) -> None:
        model = ms.MODELS["DeepSeek-R1-Distill-Qwen-14B"]
        npu = ms.NPUInfo(7, 65536, 0, 0)

        self.assertEqual(
            ms.required_free_hbm_mb(model, npu, model.runtime_profiles[0]),
            33588,
        )
        self.assertEqual(
            ms.required_free_hbm_mb(model, npu, model.runtime_profiles[-1]),
            27034,
        )

    def test_same_pix_group_wins_before_more_free_cross_group(self) -> None:
        model = ms.ModelConfig(
            path="/tmp/model",
            tp=2,
            pp=1,
            min_free_hbm_mb=0,
            reserve_hbm_mb=0,
            reserve_hbm_ratio=0,
            runtime_profiles=(ms.RuntimeProfile("test", 0.1, 1024),),
        )
        npus = [
            ms.NPUInfo(1, 65536, 15536, 0),
            ms.NPUInfo(3, 65536, 15536, 0),
            ms.NPUInfo(7, 65536, 5536, 0),
            ms.NPUInfo(11, 65536, 5536, 0),
        ]
        topology = ms.parse_npu_topology(TOPOLOGY_SAMPLE)

        selected = ms.select_npus(
            model,
            npus,
            model.runtime_profiles[0],
            topology=topology,
        )

        self.assertEqual([npu.device_id for npu in selected], [1, 3])

    def test_lease_is_visible_and_released(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lease_dir = Path(temp_dir) / "leases"
            with mock.patch.object(ms, "LEASE_DIR", lease_dir):
                token = ms.acquire_npu_leases(
                    [ms.NPUInfo(7, 65536, 0, 0)],
                    "test-container",
                )
                self.assertEqual(ms.active_leased_npu_ids(), {7})
                ms.release_npu_leases(token=token)
                self.assertEqual(ms.active_leased_npu_ids(), set())

    def test_low_memory_profile_reaches_docker_command(self) -> None:
        model = ms.MODELS["DeepSeek-R1-Distill-Qwen-14B"]
        profile = model.runtime_profiles[-1]
        command = ms.build_docker_command(
            model_name="DeepSeek-R1-Distill-Qwen-14B",
            model=model,
            parallel_plan=model.parallel_plans[0],
            selected_npus=[
                ms.NPUInfo(7, 65536, 0, 0),
                ms.NPUInfo(8, 65536, 0, 0),
            ],
            port=18000,
            container_name="test-container",
            profile=profile,
            lease_token="test-token",
        )

        utilization_index = command.index("--gpu-memory-utilization")
        context_index = command.index("--max-model-len")
        seq_index = command.index("--max-num-seqs")
        tp_index = command.index("--tensor-parallel-size")
        self.assertEqual(command[tp_index + 1], "2")
        self.assertEqual(command[utilization_index + 1], "0.35")
        self.assertEqual(command[context_index + 1], "2048")
        self.assertEqual(command[seq_index + 1], "1")
        self.assertIn("--enforce-eager", command)

    def test_emergency_profile_enables_eager_single_sequence(self) -> None:
        model = ms.MODELS["DeepSeek-R1-Distill-Qwen-32B"]
        plan = model.parallel_plans[0]
        profile = plan.runtime_profiles[-1]
        command = ms.build_docker_command(
            model_name="DeepSeek-R1-Distill-Qwen-32B",
            model=model,
            parallel_plan=plan,
            selected_npus=[
                ms.NPUInfo(1, 65536, 0, 0),
                ms.NPUInfo(3, 65536, 0, 0),
            ],
            port=18000,
            container_name="test-container",
            profile=profile,
            lease_token="test-token",
        )

        self.assertIn("--enforce-eager", command)
        seq_index = command.index("--max-num-seqs")
        context_index = command.index("--max-model-len")
        self.assertEqual(command[seq_index + 1], "1")
        self.assertEqual(command[context_index + 1], "2048")

    def test_small_evaluator_models_are_single_card_eager(self) -> None:
        for model_name in (
            "Qwen2.5-1.5B-Instruct",
            "Qwen2.5-0.5B-Instruct",
        ):
            model = ms.MODELS[model_name]
            profile = model.runtime_profiles[0]

            self.assertEqual(model.num_npus, 1)
            self.assertEqual(profile.max_model_len, 2048)
            self.assertEqual(profile.max_num_seqs, 1)
            self.assertTrue(profile.enforce_eager)

            command = ms.build_docker_command(
                model_name=model_name,
                model=model,
                parallel_plan=model.parallel_plans[0],
                selected_npus=[ms.NPUInfo(7, 65536, 0, 0)],
                port=18000,
                container_name="test-container",
                profile=profile,
                lease_token="test-token",
            )

            self.assertIn("--enforce-eager", command)
            self.assertIn("--max-num-seqs", command)
            self.assertIn("--max-num-batched-tokens", command)

    def test_startup_logs_are_compacted_and_classified(self) -> None:
        logs = "\n".join(
            [
                "(EngineCore pid=278) noise",
                "rtsMallocHost execution failed, reason=driver error:out of memory",
                "[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]",
                "Traceback (most recent call last):",
            ]
        )

        failure_type, hint = ms.classify_startup_failure(logs)
        compact = ms.compact_startup_logs(logs)

        self.assertEqual(failure_type, "host_memory_oom")
        self.assertIn("host memory", hint)
        self.assertTrue(compact[0].startswith("... 已隐藏"))
        self.assertTrue(
            any("rtsMallocHost" in line for line in compact)
        )
        self.assertTrue(
            any("Traceback" in line for line in compact)
        )

    def test_memory_startup_failure_retries_next_profile(self) -> None:
        service = ms.ModelService(
            "DeepSeek-R1-Distill-Qwen-14B",
            wait_for_npu=False,
        )
        selected = [ms.NPUInfo(7, 65536, 0, 0)]

        with (
            mock.patch.object(ms, "container_exists", return_value=False),
            mock.patch.object(
                service,
                "_launch_profile",
                side_effect=[
                    (18000, selected, False),
                    (18001, selected, False),
                ],
            ) as launch,
            mock.patch.object(
                ms,
                "wait_health",
                side_effect=[RuntimeError("failed"), None],
            ),
            mock.patch.object(
                ms,
                "get_container_logs",
                return_value="Free memory on device is less than desired",
            ),
            mock.patch.object(
                ms,
                "save_startup_error_log",
                return_value=Path("/tmp/error.log"),
            ),
            mock.patch.object(service, "stop"),
        ):
            base_url = service.start()

        self.assertEqual(base_url, "http://127.0.0.1:18001/v1")
        self.assertEqual(service.runtime_profile.name, "tp2_low_eager")
        self.assertEqual(launch.call_count, 2)

    def test_32b_uses_2_3_4_5_8_card_plans(self) -> None:
        model = ms.MODELS["DeepSeek-R1-Distill-Qwen-32B"]

        self.assertEqual(
            [
                (plan.name, plan.tp, plan.pp, plan.min_free_hbm_mb)
                for plan in model.parallel_plans
            ],
            [
                ("tp2_pp1", 2, 1, 40000),
                ("pp3", 1, 3, 28000),
                ("tp4", 4, 1, 22000),
                ("pp5", 1, 5, 16000),
                ("tp8", 8, 1, 14000),
            ],
        )

        self.assertEqual(
            [plan.num_npus for plan in model.parallel_plans],
            [2, 3, 4, 5, 8],
        )

        for plan_name in ("pp3", "pp5"):
            plan = next(
                plan
                for plan in model.parallel_plans
                if plan.name == plan_name
            )
            self.assertTrue(
                all(profile.enforce_eager for profile in plan.runtime_profiles)
            )

        self.assertEqual(
            [
                profile.gpu_memory_utilization
                for plan in model.parallel_plans
                for profile in plan.runtime_profiles
            ],
            [
                0.65,
                0.60,
                0.55,
                0.50,
                0.45,
                0.40,
                0.40,
                0.35,
                0.30,
                0.35,
                0.30,
                0.25,
                0.30,
                0.25,
                0.20,
            ],
        )

        tp4 = next(plan for plan in model.parallel_plans if plan.name == "tp4")
        tp4_low = tp4.runtime_profiles[1]
        required = ms.required_free_hbm_mb(
            model,
            ms.NPUInfo(7, 65536, 0, 0),
            tp4_low,
            tp4,
        )
        self.assertEqual(required, 27034)

    def test_qwen3_32b_uses_dense_32b_plans_without_reasoning_parser(self) -> None:
        model = ms.MODELS["Qwen3-32B"]

        self.assertEqual(model.path, "/user_home/linlujun/linlujun/model/Qwen3-32B")
        self.assertIsNone(model.reasoning_parser)
        self.assertEqual(
            [(plan.name, plan.tp, plan.pp) for plan in model.parallel_plans],
            [
                ("tp2_pp1", 2, 1),
                ("pp3", 1, 3),
                ("tp4", 4, 1),
                ("pp5", 1, 5),
                ("tp8", 8, 1),
            ],
        )

        command = ms.build_docker_command(
            model_name="Qwen3-32B",
            model=model,
            parallel_plan=model.parallel_plans[0],
            selected_npus=[
                ms.NPUInfo(1, 65536, 0, 0),
                ms.NPUInfo(5, 65536, 0, 0),
            ],
            port=18000,
            container_name="test-container",
            profile=model.runtime_profiles[0],
            lease_token="test-token",
        )

        self.assertNotIn("--reasoning-parser", command)
        self.assertIn("--enforce-eager", command)

    def test_14b_models_use_tp2_conservative_profiles(self) -> None:
        for model_name in (
            "Qwen2.5-14B-Instruct",
            "DeepSeek-R1-Distill-Qwen-14B",
        ):
            model = ms.MODELS[model_name]
            plan = model.parallel_plans[0]

            self.assertEqual(plan.tp, 2)
            self.assertEqual(plan.pp, 1)
            self.assertEqual(plan.num_npus, 2)
            self.assertTrue(
                all(profile.enforce_eager for profile in plan.runtime_profiles)
            )
            self.assertEqual(
                [profile.name for profile in plan.runtime_profiles],
                [
                    "tp2_safe_eager",
                    "tp2_low_eager",
                    "tp2_emergency_eager",
                ],
            )

    def test_only_requested_card_counts_are_allowed(self) -> None:
        with self.assertRaises(ValueError):
            ms.ModelConfig(
                path="/tmp/model",
                tp=6,
                min_free_hbm_mb=1,
            )

    def test_shared_cards_use_extra_reserve(self) -> None:
        model = ms.MODELS["DeepSeek-R1-Distill-Qwen-32B"]
        plan = model.parallel_plans[0]
        profile = plan.runtime_profiles[0]
        npu = ms.NPUInfo(1, 65536, 10173, 0, has_process=True)

        self.assertEqual(
            ms.required_free_hbm_mb(
                model,
                npu,
                profile,
                plan,
                shared=True,
            ),
            50791,
        )

    def test_shared_tp2_prefers_pix_group_with_processes(self) -> None:
        model = ms.MODELS["DeepSeek-R1-Distill-Qwen-32B"]
        plan = model.parallel_plans[0]
        profile = plan.runtime_profiles[0]
        npus = [
            ms.NPUInfo(1, 65536, 10173, 0, has_process=True),
            ms.NPUInfo(5, 65536, 7797, 33, has_process=True),
            ms.NPUInfo(9, 65536, 5604, 21, has_process=True),
        ]

        with self.assertRaises(ms.NPUAllocationError):
            ms.select_npus(
                model,
                npus,
                profile,
                plan=plan,
                topology=ms.parse_npu_topology(TOPOLOGY_SAMPLE),
            )

        selected = ms.select_npus(
            model,
            npus,
            profile,
            plan=plan,
            topology=ms.parse_npu_topology(TOPOLOGY_SAMPLE),
            shared=True,
        )

        self.assertEqual([npu.device_id for npu in selected], [1, 5])

    def test_launch_falls_back_from_exclusive_to_stable_shared_cards(self) -> None:
        service = ms.ModelService(
            "DeepSeek-R1-Distill-Qwen-32B",
            wait_for_npu=False,
        )
        model = ms.MODELS[service.model_name]
        plan = model.parallel_plans[0]
        profile = plan.runtime_profiles[0]
        selected = [
            ms.NPUInfo(1, 65536, 10173, 0, has_process=True),
            ms.NPUInfo(5, 65536, 7797, 33, has_process=True),
        ]

        with (
            mock.patch.object(
                ms,
                "allocation_lock",
                return_value=contextlib.nullcontext(),
            ),
            mock.patch.object(
                ms,
                "get_plan",
                side_effect=[
                    ms.NPUAllocationError("no exclusive group"),
                    (model, selected),
                ],
            ) as get_plan,
            mock.patch.object(
                ms,
                "stabilize_shared_selection",
                return_value=selected,
            ) as stabilize,
            mock.patch.object(ms, "acquire_npu_leases", return_value="token"),
            mock.patch.object(ms.time, "sleep"),
            mock.patch.object(ms, "confirm_selection", return_value=selected),
            mock.patch.object(ms, "find_free_port", return_value=18000),
            mock.patch.object(ms, "build_docker_command", return_value=["docker"]),
            mock.patch.object(ms.subprocess, "run"),
        ):
            port, result, shared = service._launch_profile(
                model,
                plan,
                profile,
            )

        self.assertEqual(port, 18000)
        self.assertEqual(result, selected)
        self.assertTrue(shared)
        self.assertEqual(get_plan.call_count, 2)
        self.assertEqual(get_plan.call_args_list[0].kwargs["shared"], False)
        self.assertEqual(get_plan.call_args_list[1].kwargs["shared"], True)
        stabilize.assert_called_once()

    def test_best_fit_fills_tighter_cards_with_same_topology(self) -> None:
        model = ms.ModelConfig(
            path="/tmp/model",
            tp=2,
            min_free_hbm_mb=0,
            reserve_hbm_mb=0,
            reserve_hbm_ratio=0,
            runtime_profiles=(ms.RuntimeProfile("test", 0.1, 1024),),
        )
        npus = [
            ms.NPUInfo(1, 65536, 5536, 0),
            ms.NPUInfo(3, 65536, 5536, 0),
            ms.NPUInfo(5, 65536, 45536, 0),
            ms.NPUInfo(7, 65536, 45536, 0),
        ]

        selected = ms.select_npus(
            model,
            npus,
            model.runtime_profiles[0],
            topology=ms.parse_npu_topology(TOPOLOGY_SAMPLE),
        )

        self.assertEqual([npu.device_id for npu in selected], [5, 7])

    def test_shared_recheck_rejects_fast_hbm_growth(self) -> None:
        model = ms.ModelConfig(
            path="/tmp/model",
            tp=1,
            min_free_hbm_mb=0,
            reserve_hbm_mb=0,
            reserve_hbm_ratio=0,
            runtime_profiles=(ms.RuntimeProfile("test", 0.1, 1024),),
        )
        plan = model.parallel_plans[0]
        old = ms.NPUInfo(1, 65536, 10000, 0, has_process=True)
        current = ms.NPUInfo(1, 65536, 12000, 0, has_process=True)

        with (
            mock.patch.object(ms, "run_npu_smi", return_value="sample"),
            mock.patch.object(ms, "parse_npu_smi", return_value=[current]),
            self.assertRaises(ms.NPUAllocationError),
        ):
            ms.confirm_selection(
                model,
                plan,
                model.runtime_profiles[0],
                [old],
                shared=True,
                max_hbm_drop_mb=1024,
            )

    def test_small_model_budget_switches_to_more_cards(self) -> None:
        service = ms.ModelService(
            "DeepSeek-R1-Distill-Qwen-32B",
            wait_for_npu=False,
        )
        tp2_selected = [
            ms.NPUInfo(device_id, 65536, 0, 0)
            for device_id in (1, 3)
        ]
        pp3_selected = [
            ms.NPUInfo(device_id, 65536, 0, 0)
            for device_id in (1, 3, 5)
        ]

        with (
            mock.patch.object(ms, "container_exists", return_value=False),
            mock.patch.object(
                service,
                "_launch_profile",
                side_effect=[
                    (18000, tp2_selected, False),
                    (18001, pp3_selected, False),
                ],
            ) as launch,
            mock.patch.object(
                ms,
                "wait_health",
                side_effect=[RuntimeError("failed"), None],
            ),
            mock.patch.object(
                ms,
                "get_container_logs",
                return_value="No available memory for the cache blocks",
            ),
            mock.patch.object(
                ms,
                "save_startup_error_log",
                return_value=Path("/tmp/error.log"),
            ),
            mock.patch.object(service, "stop"),
        ):
            base_url = service.start()

        self.assertEqual(base_url, "http://127.0.0.1:18001/v1")
        self.assertEqual(service.parallel_plan.name, "pp3")
        self.assertEqual(service.runtime_profile.name, "pp3_safe")
        self.assertEqual(launch.call_count, 2)

    def test_experimental_pp_failure_falls_back_to_tp4(self) -> None:
        service = ms.ModelService(
            "DeepSeek-R1-Distill-Qwen-32B",
            wait_for_npu=False,
        )
        tp2_selected = [
            ms.NPUInfo(device_id, 65536, 0, 0)
            for device_id in (1, 3)
        ]
        pp3_selected = [
            ms.NPUInfo(device_id, 65536, 0, 0)
            for device_id in (1, 3, 5)
        ]
        tp4_selected = [
            ms.NPUInfo(device_id, 65536, 0, 0)
            for device_id in (1, 3, 5, 7)
        ]

        with (
            mock.patch.object(ms, "container_exists", return_value=False),
            mock.patch.object(
                service,
                "_launch_profile",
                side_effect=[
                    (18000, tp2_selected, False),
                    (18001, pp3_selected, False),
                    (18002, tp4_selected, False),
                ],
            ) as launch,
            mock.patch.object(
                ms,
                "wait_health",
                side_effect=[RuntimeError("failed"), RuntimeError("failed"), None],
            ),
            mock.patch.object(
                ms,
                "get_container_logs",
                side_effect=[
                    "No available memory for the cache blocks",
                    "pipeline parallel is not supported",
                ],
            ),
            mock.patch.object(
                ms,
                "save_startup_error_log",
                return_value=Path("/tmp/error.log"),
            ),
            mock.patch.object(service, "stop"),
        ):
            base_url = service.start()

        self.assertEqual(base_url, "http://127.0.0.1:18002/v1")
        self.assertEqual(service.parallel_plan.name, "tp4")
        self.assertEqual(service.runtime_profile.name, "tp4_safe_eager")
        self.assertEqual(launch.call_count, 3)

    def test_shorter_context_is_tried_for_kv_capacity_error(self) -> None:
        logs = (
            "max seq len is larger than the maximum number of tokens "
            "that can be stored in KV cache"
        )

        self.assertTrue(ms.is_memory_startup_failure(logs))
        self.assertFalse(ms.is_model_budget_too_small(logs))

    def test_32b_tp4_selects_complete_pix_group(self) -> None:
        model = ms.MODELS["DeepSeek-R1-Distill-Qwen-32B"]
        tp4 = next(plan for plan in model.parallel_plans if plan.name == "tp4")
        profile = tp4.runtime_profiles[-1]
        npus = [
            ms.NPUInfo(device_id, 65536, 25000, 0)
            for device_id in (1, 3, 5, 7, 8, 9, 10, 11)
        ]

        selected = ms.select_npus(
            model,
            npus,
            profile,
            plan=tp4,
            topology=ms.parse_npu_topology(TOPOLOGY_SAMPLE),
        )

        self.assertEqual(
            [npu.device_id for npu in selected],
            [1, 3, 5, 7],
        )

    def test_parallel_plan_checks_attention_head_divisibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "num_attention_heads": 40,
                        "num_key_value_heads": 8,
                        "num_hidden_layers": 64,
                    }
                ),
                encoding="utf-8",
            )
            model = ms.ModelConfig(
                path=temp_dir,
                tp=2,
                pp=1,
                min_free_hbm_mb=1,
            )
            ms.validate_parallel_plan(
                model,
                ms.ParallelPlan("tp8", 8, 1, 1, (ms.NORMAL_PROFILE,)),
            )
            ms.validate_parallel_plan(
                model,
                ms.ParallelPlan("pp3", 1, 3, 1, (ms.NORMAL_PROFILE,)),
            )
            ms.validate_parallel_plan(
                model,
                ms.ParallelPlan("pp5", 1, 5, 1, (ms.NORMAL_PROFILE,)),
            )
            with self.assertRaises(ValueError):
                ms.validate_parallel_plan(
                    model,
                    ms.ParallelPlan("tp3", 3, 1, 1, (ms.NORMAL_PROFILE,)),
                )


if __name__ == "__main__":
    unittest.main()

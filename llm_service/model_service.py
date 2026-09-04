from __future__ import annotations

import argparse
import contextlib
import fcntl
import itertools
import json
import math
import os
import re
import socket
import subprocess
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path


IMAGE = "quay.io/ascend/vllm-ascend:v0.22.1rc1"
START_TIMEOUT = 900
ALLOWED_NPU_COUNTS = (1, 2, 3, 4, 5, 8)
DEFAULT_RESERVE_HBM_MB = 4096
DEFAULT_RESERVE_HBM_RATIO = 0.05
EXCLUSIVE_MAX_AICORE_PCT = 5
SHARED_MAX_AICORE_PCT = 40
SHARED_RESERVE_HBM_MB = 8192
SHARED_STABILITY_SAMPLES = 3
SHARED_MAX_HBM_DROP_MB = 1024
NPU_STABILITY_CHECK_SECONDS = 0.5
NPU_WAIT_POLL_SECONDS = 5

ALLOCATOR_LOCK_PATH = Path("/tmp/linlujun-yuyi-npu/allocator.lock")
LEASE_DIR = ALLOCATOR_LOCK_PATH.parent / "leases"
ERROR_LOG_DIR = Path("logs")


@dataclass(frozen=True)
class RuntimeProfile:
    """一档 vLLM 显存配置；按顺序从性能优先降到低显存。"""

    name: str
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    enforce_eager: bool = False


@dataclass(frozen=True)
class ParallelPlan:
    """一个经过允许的 TP/PP 组合及其显存降档。"""

    name: str
    tp: int
    pp: int
    min_free_hbm_mb: int
    runtime_profiles: tuple[RuntimeProfile, ...]
    fallback_on_startup_error: bool = False

    @property
    def num_npus(self) -> int:
        return self.tp * self.pp


NORMAL_PROFILE = RuntimeProfile(
    name="normal",
    gpu_memory_utilization=0.75,
    max_model_len=8192,
)

LOW_MEMORY_PROFILE = RuntimeProfile(
    name="low_memory",
    gpu_memory_utilization=0.65,
    max_model_len=4096,
    max_num_seqs=8,
    max_num_batched_tokens=4096,
)

EMERGENCY_PROFILE = RuntimeProfile(
    name="emergency",
    gpu_memory_utilization=0.60,
    max_model_len=3072,
    max_num_seqs=4,
    max_num_batched_tokens=2048,
)

LARGE_SAFE_PROFILE = RuntimeProfile(
    name="large_safe_eager",
    gpu_memory_utilization=0.50,
    max_model_len=4096,
    max_num_seqs=4,
    max_num_batched_tokens=2048,
    enforce_eager=True,
)

LARGE_LOW_PROFILE = RuntimeProfile(
    name="large_low_eager",
    gpu_memory_utilization=0.40,
    max_model_len=3072,
    max_num_seqs=2,
    max_num_batched_tokens=1536,
    enforce_eager=True,
)

LARGE_EMERGENCY_PROFILE = RuntimeProfile(
    name="large_emergency_eager",
    gpu_memory_utilization=0.30,
    max_model_len=2048,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
    enforce_eager=True,
)

SMALL_EVAL_PROFILE = RuntimeProfile(
    name="small_eval_eager",
    gpu_memory_utilization=0.25,
    max_model_len=2048,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
    enforce_eager=True,
)

SMALL_EVAL_LOW_PROFILE = RuntimeProfile(
    name="small_eval_low_eager",
    gpu_memory_utilization=0.20,
    max_model_len=1536,
    max_num_seqs=1,
    max_num_batched_tokens=768,
    enforce_eager=True,
)

TINY_EVAL_PROFILE = RuntimeProfile(
    name="tiny_eval_eager",
    gpu_memory_utilization=0.20,
    max_model_len=2048,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
    enforce_eager=True,
)

TINY_EVAL_LOW_PROFILE = RuntimeProfile(
    name="tiny_eval_low_eager",
    gpu_memory_utilization=0.15,
    max_model_len=1536,
    max_num_seqs=1,
    max_num_batched_tokens=768,
    enforce_eager=True,
)


@dataclass(frozen=True)
class ModelConfig:
    path: str
    tp: int
    min_free_hbm_mb: int

    pp: int = 1
    reasoning_parser: str | None = None

    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    runtime_profiles: tuple[RuntimeProfile, ...] = (NORMAL_PROFILE,)
    reserve_hbm_mb: int = DEFAULT_RESERVE_HBM_MB
    reserve_hbm_ratio: float = DEFAULT_RESERVE_HBM_RATIO
    fallback_parallel_plans: tuple[ParallelPlan, ...] = ()

    @property
    def num_npus(self) -> int:
        return self.tp * self.pp

    @property
    def parallel_plans(self) -> tuple[ParallelPlan, ...]:
        primary = ParallelPlan(
            name=f"tp{self.tp}_pp{self.pp}",
            tp=self.tp,
            pp=self.pp,
            min_free_hbm_mb=self.min_free_hbm_mb,
            runtime_profiles=self.runtime_profiles,
        )
        return (primary, *self.fallback_parallel_plans)

    def __post_init__(self) -> None:
        if self.num_npus not in ALLOWED_NPU_COUNTS:
            raise ValueError(
                f"NPU 数量必须是 {ALLOWED_NPU_COUNTS} 之一: {self.num_npus}"
            )

        for plan in self.parallel_plans:
            if plan.num_npus not in ALLOWED_NPU_COUNTS:
                raise ValueError(
                    f"计划 {plan.name} 的 NPU 数量必须是 "
                    f"{ALLOWED_NPU_COUNTS} 之一"
                )
            if not plan.runtime_profiles:
                raise ValueError(f"计划 {plan.name} 的 runtime_profiles 不能为空")

            previous = 1.0
            for profile in plan.runtime_profiles:
                utilization = profile.gpu_memory_utilization
                if not 0 < utilization <= 1:
                    raise ValueError(
                        f"gpu_memory_utilization 非法: {utilization}"
                    )
                if utilization > previous:
                    raise ValueError(
                        f"计划 {plan.name} 的显存档位必须从高到低排列"
                    )
                previous = utilization


MODELS = {
    "Qwen2.5-1.5B-Instruct": ModelConfig(
        path="/user_home/linlujun/linlujun/model/Qwen2.5-1.5B-Instruct",
        tp=1,
        pp=1,
        min_free_hbm_mb=8000,
        runtime_profiles=(
            SMALL_EVAL_PROFILE,
            SMALL_EVAL_LOW_PROFILE,
        ),
    ),

    "Qwen2.5-0.5B-Instruct": ModelConfig(
        path="/user_home/linlujun/linlujun/model/Qwen2.5-0.5B-Instruct",
        tp=1,
        pp=1,
        min_free_hbm_mb=6000,
        runtime_profiles=(
            TINY_EVAL_PROFILE,
            TINY_EVAL_LOW_PROFILE,
        ),
    ),

    "Qwen2.5-14B-Instruct": ModelConfig(
        path="/user_home/linlujun/linlujun/model/Qwen2.5-14B-Instruct",
        tp=2,
        pp=1,
        min_free_hbm_mb=18000,
        runtime_profiles=(
            RuntimeProfile(
                "tp2_safe_eager",
                0.45,
                4096,
                4,
                2048,
                enforce_eager=True,
            ),
            RuntimeProfile(
                "tp2_low_eager",
                0.40,
                3072,
                2,
                1536,
                enforce_eager=True,
            ),
            RuntimeProfile(
                "tp2_emergency_eager",
                0.35,
                2048,
                1,
                1024,
                enforce_eager=True,
            ),
        ),
    ),

    "Qwen3-32B": ModelConfig(
        path="/user_home/linlujun/linlujun/model/Qwen3-32B",
        tp=2,
        pp=1,
        min_free_hbm_mb=50000,
        runtime_profiles=(
            RuntimeProfile(
                "tp2_fast",
                0.72,
                1536,
                1,
                1024,
            ),
            RuntimeProfile(
                "tp2_eager",
                0.68,
                1536,
                1,
                1024,
                enforce_eager=True,
            ),
        ),
        fallback_parallel_plans=(
            # Qwen3-32B 是 64 Q heads / 8 KV heads；不能用 TP3/TP5。
            # Ascend 上 PP3/PP5 容易在 worker/TCPStore 初始化阶段失败；
            # 先试省卡的 TP2，再退到更稳的 TP4/TP8。
            ParallelPlan(
                name="tp4",
                tp=4,
                pp=1,
                min_free_hbm_mb=26000,
                runtime_profiles=(
                    RuntimeProfile(
                        "tp4_safe_eager",
                        0.40,
                        4096,
                        4,
                        2048,
                        enforce_eager=True,
                    ),
                    RuntimeProfile(
                        "tp4_low_eager",
                        0.35,
                        3072,
                        2,
                        1536,
                        enforce_eager=True,
                    ),
                    RuntimeProfile(
                        "tp4_emergency_eager",
                        0.30,
                        2048,
                        1,
                        2048,
                        enforce_eager=True,
                    ),
                ),
            ),
            ParallelPlan(
                name="tp8",
                tp=8,
                pp=1,
                min_free_hbm_mb=14000,
                runtime_profiles=(
                    RuntimeProfile("tp8_safe", 0.30, 4096, 8, 4096),
                    RuntimeProfile("tp8_low", 0.25, 3072, 4, 2048),
                    RuntimeProfile(
                        "tp8_emergency_eager",
                        0.20,
                        2048,
                        1,
                        2048,
                        enforce_eager=True,
                    ),
                ),
            ),
        ),
    ),

    "DeepSeek-R1-Distill-Llama-8B": ModelConfig(
        path="/user_home/linlujun/linlujun/model/DeepSeek-R1-Distill-Llama-8B",
        tp=1,
        pp=1,
        min_free_hbm_mb=20000,
        reasoning_parser="deepseek_r1",
        runtime_profiles=(
            RuntimeProfile(
                "eval_safe_eager",
                0.40,
                3072,
                2,
                1536,
                enforce_eager=True,
            ),
            RuntimeProfile(
                "eval_low_eager",
                0.35,
                2048,
                1,
                1024,
                enforce_eager=True,
            ),
            LARGE_EMERGENCY_PROFILE,
        ),
    ),

    "DeepSeek-R1-Distill-Qwen-14B": ModelConfig(
        path="/user_home/linlujun/linlujun/model/DeepSeek-R1-Distill-Qwen-14B",
        tp=2,
        pp=1,
        min_free_hbm_mb=18000,
        reasoning_parser="deepseek_r1",
        runtime_profiles=(
            RuntimeProfile(
                "tp2_safe_eager",
                0.45,
                4096,
                4,
                2048,
                enforce_eager=True,
            ),
            RuntimeProfile(
                "tp2_low_eager",
                0.40,
                3072,
                2,
                1536,
                enforce_eager=True,
            ),
            RuntimeProfile(
                "tp2_emergency_eager",
                0.35,
                2048,
                1,
                1024,
                enforce_eager=True,
            ),
        ),
    ),

    "DeepSeek-R1-Distill-Qwen-32B": ModelConfig(
        path="/user_home/linlujun/linlujun/model/DeepSeek-R1-Distill-Qwen-32B",
        tp=2,
        pp=1,
        min_free_hbm_mb=55000,
        reasoning_parser="deepseek_r1",
        runtime_profiles=(
            RuntimeProfile(
                "tp2_safe_eager",
                0.65,
                4096,
                4,
                2048,
                enforce_eager=True,
            ),
            RuntimeProfile(
                "tp2_low_eager",
                0.60,
                3072,
                2,
                1536,
                enforce_eager=True,
            ),
            RuntimeProfile(
                "tp2_emergency_eager",
                0.55,
                2048,
                1,
                2048,
                enforce_eager=True,
            ),
        ),
        fallback_parallel_plans=(
            # Qwen-32B 的 40 个 attention heads 和 8 个 KV heads
            # 不能被 3/5 整除；奇数卡必须使用 TP1 + PP3/PP5。
            # PP 档统一关闭图捕获，以减少额外显存并提高 Ascend 兼容性。
            ParallelPlan(
                name="pp3",
                tp=1,
                pp=3,
                min_free_hbm_mb=28000,
                runtime_profiles=(
                    RuntimeProfile(
                        "pp3_safe",
                        0.50,
                        4096,
                        8,
                        4096,
                        enforce_eager=True,
                    ),
                    RuntimeProfile(
                        "pp3_low",
                        0.45,
                        3072,
                        4,
                        2048,
                        enforce_eager=True,
                    ),
                    RuntimeProfile(
                        "pp3_emergency_eager",
                        0.40,
                        2048,
                        1,
                        2048,
                        enforce_eager=True,
                    ),
                ),
                fallback_on_startup_error=True,
            ),
            ParallelPlan(
                name="tp4",
                tp=4,
                pp=1,
                min_free_hbm_mb=22000,
                runtime_profiles=(
                    RuntimeProfile(
                        "tp4_safe_eager",
                        0.40,
                        4096,
                        4,
                        2048,
                        enforce_eager=True,
                    ),
                    RuntimeProfile(
                        "tp4_low_eager",
                        0.35,
                        3072,
                        2,
                        1536,
                        enforce_eager=True,
                    ),
                    RuntimeProfile(
                        "tp4_emergency_eager",
                        0.30,
                        2048,
                        1,
                        2048,
                        enforce_eager=True,
                    ),
                ),
            ),
            ParallelPlan(
                name="pp5",
                tp=1,
                pp=5,
                min_free_hbm_mb=16000,
                runtime_profiles=(
                    RuntimeProfile(
                        "pp5_safe",
                        0.35,
                        4096,
                        8,
                        4096,
                        enforce_eager=True,
                    ),
                    RuntimeProfile(
                        "pp5_low",
                        0.30,
                        3072,
                        4,
                        2048,
                        enforce_eager=True,
                    ),
                    RuntimeProfile(
                        "pp5_emergency_eager",
                        0.25,
                        2048,
                        1,
                        2048,
                        enforce_eager=True,
                    ),
                ),
                fallback_on_startup_error=True,
            ),
            ParallelPlan(
                name="tp8",
                tp=8,
                pp=1,
                min_free_hbm_mb=14000,
                runtime_profiles=(
                    RuntimeProfile("tp8_safe", 0.30, 4096, 8, 4096),
                    RuntimeProfile("tp8_low", 0.25, 3072, 4, 2048),
                    RuntimeProfile(
                        "tp8_emergency_eager",
                        0.20,
                        2048,
                        1,
                        2048,
                        enforce_eager=True,
                    ),
                ),
            ),
        ),
    ),

    "DeepSeek-R1-Distill-Llama-70B": ModelConfig(
        path="/user_home/linlujun/linlujun/model/DeepSeek-R1-Distill-Llama-70B",
        tp=1,
        pp=3,
        min_free_hbm_mb=56000,
        reasoning_parser="deepseek_r1",
        runtime_profiles=(
            RuntimeProfile(
                "pp3_safe_eager",
                0.55,
                4096,
                4,
                2048,
                enforce_eager=True,
            ),
            RuntimeProfile(
                "pp3_low_eager",
                0.50,
                3072,
                2,
                1536,
                enforce_eager=True,
            ),
            RuntimeProfile(
                "pp3_emergency_eager",
                0.45,
                2048,
                1,
                1024,
                enforce_eager=True,
            ),
        ),
    ),
}


@dataclass
class NPUInfo:
    device_id: int
    total_hbm_mb: int
    used_hbm_mb: int
    aicore_pct: int
    healthy: bool = True
    has_process: bool = False
    docker_holders: tuple[str, ...] = ()

    @property
    def free_hbm_mb(self) -> int:
        return self.total_hbm_mb - self.used_hbm_mb


@dataclass(frozen=True)
class VLLMStartupMemoryError:
    free_gib: float
    total_gib: float
    utilization: float
    requested_gib: float

    @property
    def shortfall_gib(self) -> float:
        return max(0.0, self.requested_gib - self.free_gib)

    @property
    def max_passing_utilization(self) -> float:
        if self.total_gib <= 0:
            return 0.0
        return self.free_gib / self.total_gib


def run_npu_smi() -> str:
    result = subprocess.run(
        ["npu-smi", "info"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_npu_smi(text: str) -> list[NPUInfo]:
    devices: dict[int, NPUInfo] = {}
    lines = text.splitlines()

    for i, line in enumerate(lines):
        m = re.match(r"\|\s*(\d+)\s+910", line)
        if not m or i + 1 >= len(lines):
            continue

        device_id = int(m.group(1))
        detail = lines[i + 1]

        hbm_match = re.search(
            r"(\d+)\s*/\s*(\d+)\s*\|?\s*$",
            detail,
        )
        if not hbm_match:
            continue

        used_hbm = int(hbm_match.group(1))
        total_hbm = int(hbm_match.group(2))

        bus_match = re.search(
            r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d+"
            r"\s*\|\s*(\d+)",
            detail,
        )
        aicore = int(bus_match.group(1)) if bus_match else 0

        devices[device_id] = NPUInfo(
            device_id=device_id,
            total_hbm_mb=total_hbm,
            used_hbm_mb=used_hbm,
            aicore_pct=aicore,
            healthy=bool(re.search(r"\|\s*OK\s*\|", line)),
        )

    process_re = re.compile(
        r"\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|"
    )

    for line in lines:
        m = process_re.match(line)
        if not m:
            continue

        device_id = int(m.group(1))
        pid = int(m.group(2))

        if pid > 0 and device_id in devices:
            devices[device_id].has_process = True

    return sorted(devices.values(), key=lambda x: x.device_id)


class NPUAllocationError(RuntimeError):
    """当前没有满足条件的 NPU 卡组。"""


TOPOLOGY_PRIORITY = {
    "X": 0,
    "HCCS": 0,
    "PIX": 1,
    "PXB": 2,
    "PHB": 3,
    "SYS": 4,
    "NA": 5,
}


def run_npu_topology(device_id: int) -> str:
    result = subprocess.run(
        [
            "npu-smi",
            "info",
            "-t",
            "topo",
            "-i",
            str(device_id),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_npu_topology(text: str) -> dict[tuple[int, int], str]:
    """解析 npu-smi 的拓扑矩阵，返回任意两张物理卡的连接类型。"""
    headers: list[int] = []
    topology: dict[tuple[int, int], str] = {}

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if not headers and stripped.startswith("NPU"):
            header_matches = re.findall(r"NPU(\d+)", stripped)
            if len(header_matches) >= 2:
                headers = [int(value) for value in header_matches]
                continue

        row_match = re.match(r"NPU(\d+)\s+(.+)", stripped)
        if not row_match or not headers:
            continue

        row_id = int(row_match.group(1))
        values = row_match.group(2).split()
        for column_id, relation in zip(headers, values):
            if relation in TOPOLOGY_PRIORITY:
                topology[(row_id, column_id)] = relation

    return topology


def required_free_hbm_mb(
    model: ModelConfig,
    npu: NPUInfo,
    profile: RuntimeProfile,
    plan: ParallelPlan | None = None,
    shared: bool = False,
) -> int:
    """vLLM 启动门槛、模型实测下限与安全预留三者共同决定准入量。"""
    reserve = max(
        model.reserve_hbm_mb,
        math.ceil(npu.total_hbm_mb * model.reserve_hbm_ratio),
    )
    if shared:
        reserve = max(reserve, SHARED_RESERVE_HBM_MB)
    vllm_requested = math.ceil(
        npu.total_hbm_mb * profile.gpu_memory_utilization
    )
    model_floor = (
        plan.min_free_hbm_mb
        if plan is not None
        else model.min_free_hbm_mb
    )
    return max(
        model_floor,
        vllm_requested + reserve,
    )


def requested_hbm_mb(npu: NPUInfo, profile: RuntimeProfile) -> int:
    return math.ceil(npu.total_hbm_mb * profile.gpu_memory_utilization)


def format_signed_mb(value: int) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value} MB"


def format_npu_memory_diagnostics(
    model: ModelConfig,
    selected: list[NPUInfo],
    profile: RuntimeProfile,
    plan: ParallelPlan,
    shared: bool = False,
) -> list[str]:
    lines = ["NPU显存诊断:"]
    for npu in selected:
        requested = requested_hbm_mb(npu, profile)
        required = required_free_hbm_mb(
            model,
            npu,
            profile,
            plan,
            shared=shared,
        )
        process_text = "有" if npu.has_process else "无"
        holders = ",".join(npu.docker_holders) or "无"
        lines.append(
            f"  NPU {npu.device_id}: "
            f"total={npu.total_hbm_mb} MB, "
            f"used={npu.used_hbm_mb} MB, "
            f"free={npu.free_hbm_mb} MB, "
            f"vLLM请求≈{requested} MB, "
            f"准入={required} MB, "
            f"准入余量={format_signed_mb(npu.free_hbm_mb - required)}, "
            f"启动后预计剩余={format_signed_mb(npu.free_hbm_mb - requested)}, "
            f"AICore={npu.aicore_pct}%, "
            f"process={process_text}, "
            f"Docker映射={holders}"
        )
    return lines


def _group_topology_score(
    group: tuple[NPUInfo, ...],
    topology: dict[tuple[int, int], str],
) -> int:
    if len(group) <= 1:
        return 0

    worst = 0
    for left, right in itertools.combinations(group, 2):
        relation = topology.get(
            (left.device_id, right.device_id),
            topology.get((right.device_id, left.device_id), "NA"),
        )
        worst = max(worst, TOPOLOGY_PRIORITY.get(relation, 5))
    return worst


def select_npus(
    model: ModelConfig,
    npus: list[NPUInfo],
    profile: RuntimeProfile,
    plan: ParallelPlan | None = None,
    topology: dict[tuple[int, int], str] | None = None,
    excluded_ids: set[int] | None = None,
    leased_ids: set[int] | None = None,
    shared: bool = False,
) -> list[NPUInfo]:
    topology = topology or {}
    excluded_ids = excluded_ids or set()
    leased_ids = leased_ids or set()
    plan = plan or model.parallel_plans[0]

    max_aicore_pct = (
        SHARED_MAX_AICORE_PCT
        if shared
        else EXCLUSIVE_MAX_AICORE_PCT
    )

    candidates = [
        npu
        for npu in npus
        if npu.healthy
        and npu.device_id not in excluded_ids
        and npu.device_id not in leased_ids
        and (shared or not npu.has_process)
        and npu.aicore_pct <= max_aicore_pct
        and npu.free_hbm_mb >= required_free_hbm_mb(
            model, npu, profile, plan, shared=shared
        )
    ]

    if len(candidates) < plan.num_npus:
        mode = "共享" if shared else "独占"
        raise NPUAllocationError(
            f"计划 {plan.name}/{profile.name}({mode}) "
            f"需要 {plan.num_npus} 张候选 NPU "
            f"(TP={plan.tp}, PP={plan.pp})，"
            f"但当前只有 {len(candidates)} 张满足显存、负载和租约条件"
        )

    groups = list(itertools.combinations(candidates, plan.num_npus))

    def group_rank(group: tuple[NPUInfo, ...]) -> tuple:
        headrooms = [
            npu.free_hbm_mb
            - required_free_hbm_mb(
                model,
                npu,
                profile,
                plan,
                shared=shared,
            )
            for npu in group
        ]
        mapped_count = sum(bool(npu.docker_holders) for npu in group)
        process_count = sum(npu.has_process for npu in group)
        max_aicore = max(npu.aicore_pct for npu in group)
        sum_aicore = sum(npu.aicore_pct for npu in group)
        spread = max(headrooms) - min(headrooms)
        ids = tuple(sorted(npu.device_id for npu in group))
        return (
            _group_topology_score(group, topology),
            process_count,
            max_aicore,
            sum_aicore,
            sum(headrooms),
            spread,
            mapped_count,
            ids,
        )

    selected = min(groups, key=group_rank)
    return sorted(selected, key=lambda npu: npu.device_id)


def find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def build_npu_device_args(
    selected_npus: list[NPUInfo],
) -> list[str]:
    args: list[str] = []

    for logical_id, npu in enumerate(selected_npus):
        physical_id = npu.device_id

        args += [
            "--device",
            f"/dev/davinci{physical_id}:/dev/davinci{logical_id}",
        ]

    return args


def container_name(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()
    return f"linlujun-{slug}"


def container_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def container_running(name: str) -> bool:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "-f",
            "{{.State.Running}}",
            name,
        ],
        capture_output=True,
        text=True,
    )

    return (
        result.returncode == 0
        and result.stdout.strip() == "true"
    )


def container_label(name: str, label: str) -> str | None:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "-f",
            f"{{{{index .Config.Labels {json.dumps(label)}}}}}",
            name,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value if value and value != "<no value>" else None


@contextlib.contextmanager
def allocation_lock():
    """只协调本项目进程；外部用户需要服务器级调度器才能被约束。"""
    ALLOCATOR_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with ALLOCATOR_LOCK_PATH.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_lease(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def active_leased_npu_ids(
    exclude_container: str | None = None,
) -> set[int]:
    """返回有效租约，并回收已无进程且无容器的陈旧租约。"""
    LEASE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    active: set[int] = set()

    for path in LEASE_DIR.glob("npu-*.json"):
        data = _read_lease(path)
        if data is None:
            path.unlink(missing_ok=True)
            continue

        container = str(data.get("container", ""))
        if exclude_container and container == exclude_container:
            continue

        try:
            pid = int(data.get("owner_pid", 0))
            device_id = int(data["device_id"])
        except (KeyError, TypeError, ValueError):
            path.unlink(missing_ok=True)
            continue

        if _pid_alive(pid) or (container and container_running(container)):
            active.add(device_id)
        else:
            path.unlink(missing_ok=True)

    return active


def acquire_npu_leases(
    selected_npus: list[NPUInfo],
    container: str,
) -> str:
    """调用者必须持有 allocation_lock。"""
    LEASE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    token = uuid.uuid4().hex
    created: list[Path] = []

    try:
        for npu in selected_npus:
            path = LEASE_DIR / f"npu-{npu.device_id}.json"
            payload = {
                "device_id": npu.device_id,
                "owner_pid": os.getpid(),
                "container": container,
                "token": token,
                "created_at": time.time(),
            }
            fd = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as lease_file:
                json.dump(payload, lease_file, ensure_ascii=False)
                lease_file.flush()
                os.fsync(lease_file.fileno())
            created.append(path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise

    return token


def release_npu_leases(
    *,
    token: str | None = None,
    container: str | None = None,
) -> None:
    if not LEASE_DIR.is_dir():
        return

    for path in LEASE_DIR.glob("npu-*.json"):
        data = _read_lease(path)
        if data is None:
            continue
        token_matches = token is not None and data.get("token") == token
        container_matches = (
            container is not None and data.get("container") == container
        )
        if token_matches or container_matches:
            path.unlink(missing_ok=True)


def wait_health(
    container: str,
    port: int,
    timeout: int = START_TIMEOUT,
) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout

    while time.time() < deadline:
        if not container_running(container):
            raise RuntimeError(
                f"模型容器已退出: {container}"
            )

        try:
            with urllib.request.urlopen(
                url,
                timeout=2,
            ) as response:
                if response.status == 200:
                    return

        except Exception:
            pass

        time.sleep(2)

    raise TimeoutError(
        f"等待模型服务启动超时: {timeout} 秒"
    )


def validate_parallel_plan(
    model: ModelConfig,
    plan: ParallelPlan,
) -> None:
    """在启动前用本地 config.json 验证 TP 是否能整除注意力头。"""
    config_path = Path(model.path) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"无法读取模型配置: {config_path}") from exc

    num_hidden_layers = config.get("num_hidden_layers")
    if num_hidden_layers is not None and int(num_hidden_layers) < plan.pp:
        raise ValueError(
            f"计划 {plan.name} 不兼容: num_hidden_layers={num_hidden_layers} "
            f"小于 PP={plan.pp}"
        )

    for field in ("num_attention_heads", "num_key_value_heads"):
        value = config.get(field)
        if value is None:
            continue
        if int(value) % plan.tp != 0:
            raise ValueError(
                f"计划 {plan.name} 不兼容: {field}={value} 不能被 TP={plan.tp} 整除"
            )


def get_plan(
    model_name: str,
    parallel_plan: ParallelPlan,
    profile: RuntimeProfile,
    excluded_ids: set[int] | None = None,
    shared: bool = False,
) -> tuple[ModelConfig, list[NPUInfo]]:
    if model_name not in MODELS:
        raise ValueError(
            f"未知模型: {model_name}"
        )

    model = MODELS[model_name]

    if not Path(model.path).exists():
        raise FileNotFoundError(
            f"模型目录不存在: {model.path}"
        )

    validate_parallel_plan(model, parallel_plan)

    npus = parse_npu_smi(
        run_npu_smi()
    )

    if not npus:
        raise NPUAllocationError("npu-smi 没有返回可用设备")

    holders = get_running_container_npu_holders()
    for npu in npus:
        npu.docker_holders = tuple(sorted(holders.get(npu.device_id, [])))

    topology = parse_npu_topology(
        run_npu_topology(npus[0].device_id)
    )

    selected = select_npus(
        model,
        npus,
        profile,
        plan=parallel_plan,
        topology=topology,
        excluded_ids=excluded_ids,
        leased_ids=active_leased_npu_ids(),
        shared=shared,
    )

    return model, selected


def confirm_selection(
    model: ModelConfig,
    parallel_plan: ParallelPlan,
    profile: RuntimeProfile,
    selected_npus: list[NPUInfo],
    shared: bool = False,
    max_hbm_drop_mb: int | None = None,
) -> list[NPUInfo]:
    """租约落盘后复检同一批卡，防止测量窗口内状态变化。"""
    refreshed = {
        npu.device_id: npu
        for npu in parse_npu_smi(run_npu_smi())
    }
    confirmed: list[NPUInfo] = []

    for old in selected_npus:
        current = refreshed.get(old.device_id)
        if current is None:
            raise NPUAllocationError(f"NPU {old.device_id} 在复检时消失")
        required = required_free_hbm_mb(
            model,
            current,
            profile,
            parallel_plan,
            shared=shared,
        )
        max_aicore_pct = (
            SHARED_MAX_AICORE_PCT
            if shared
            else EXCLUSIVE_MAX_AICORE_PCT
        )
        hbm_drop_mb = old.free_hbm_mb - current.free_hbm_mb
        if (
            not current.healthy
            or (not shared and current.has_process)
            or current.aicore_pct > max_aicore_pct
            or current.free_hbm_mb < required
            or (
                max_hbm_drop_mb is not None
                and hbm_drop_mb > max_hbm_drop_mb
            )
        ):
            raise NPUAllocationError(
                f"NPU {old.device_id} 状态在测量后变化: "
                f"free={current.free_hbm_mb}MB, required={required}MB, "
                f"drop={hbm_drop_mb}MB, AICore={current.aicore_pct}%, "
                f"process={current.has_process}, shared={shared}"
            )
        confirmed.append(current)

    return confirmed


def stabilize_shared_selection(
    model: ModelConfig,
    parallel_plan: ParallelPlan,
    profile: RuntimeProfile,
    selected_npus: list[NPUInfo],
) -> list[NPUInfo]:
    """共享卡在落租约前连续复测，避免撞上正在快速扩张的任务。"""
    baseline = selected_npus
    current = selected_npus
    for _ in range(1, SHARED_STABILITY_SAMPLES):
        time.sleep(NPU_STABILITY_CHECK_SECONDS)
        current = confirm_selection(
            model,
            parallel_plan,
            profile,
            baseline,
            shared=True,
            max_hbm_drop_mb=SHARED_MAX_HBM_DROP_MB,
        )
    return current


def build_docker_command(
    model_name: str,
    model: ModelConfig,
    parallel_plan: ParallelPlan,
    selected_npus: list[NPUInfo],
    port: int,
    container_name: str,
    profile: RuntimeProfile,
    lease_token: str,
) -> list[str]:
    physical_ids = [
        npu.device_id
        for npu in selected_npus
    ]

    physical_devices = ",".join(
        str(device_id)
        for device_id in physical_ids
    )

    logical_devices = ",".join(
        str(logical_id)
        for logical_id in range(len(selected_npus))
    )

    command = [
        "docker",
        "run",
        "-d",

        "--name",
        container_name,

        "--label",
        f"yuyi.npu-lease-token={lease_token}",

        "--label",
        "yuyi.npus=" + ",".join(str(device_id) for device_id in physical_ids),

        "--runtime=ascend",

        "-e",
        f"ASCEND_VISIBLE_DEVICES={physical_devices}",

        "-e",
        f"ASCEND_RT_VISIBLE_DEVICES={logical_devices}",

        "--device",
        "/dev/davinci_manager",

        "--device",
        "/dev/devmm_svm",

        "--device",
        "/dev/hisi_hdc",

        "-v",
        "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",

        "-v",
        "/var/log/npu:/var/log/npu",

        "-v",
        "/etc/ascend_install.info:/etc/ascend_install.info:ro",

        "--ulimit",
        "memlock=-1:-1",

        "--shm-size",
        "32g",

        "--cap-add=ALL",

        "-p",
        f"{port}:{port}",

        "-v",
        f"{model.path}:/model:ro",
    ]

    command += build_npu_device_args(selected_npus)

    command += [
        IMAGE,

        "vllm",
        "serve",
        "/model",

        "--host",
        "0.0.0.0",

        "--port",
        str(port),

        "--served-model-name",
        model_name,

        "--tensor-parallel-size",
        str(parallel_plan.tp),

        "--pipeline-parallel-size",
        str(parallel_plan.pp),

        "--max-model-len",
        str(profile.max_model_len),

        "--gpu-memory-utilization",
        str(profile.gpu_memory_utilization),
    ]

    if model.reasoning_parser:
        command += [
            "--reasoning-parser",
            model.reasoning_parser,
        ]

    max_num_seqs = (
        profile.max_num_seqs
        if profile.max_num_seqs is not None
        else model.max_num_seqs
    )
    if max_num_seqs is not None:
        command += [
            "--max-num-seqs",
            str(max_num_seqs),
        ]

    max_num_batched_tokens = (
        profile.max_num_batched_tokens
        if profile.max_num_batched_tokens is not None
        else model.max_num_batched_tokens
    )
    if max_num_batched_tokens is not None:
        command += [
            "--max-num-batched-tokens",
            str(max_num_batched_tokens),
        ]
    if profile.enforce_eager:
        command.append("--enforce-eager")
    return command

def print_container_logs(
    container: str,
    tail: int = 100,
) -> None:
    if not container_exists(container):
        return

    subprocess.run(
        [
            "docker",
            "logs",
            "--tail",
            str(tail),
            container,
        ],
        check=False,
    )


def get_container_logs(
    container: str,
    tail: int | None = 500,
) -> str:
    if not container_exists(container):
        return ""
    command = ["docker", "logs"]
    if tail is not None:
        command += ["--tail", str(tail)]
    command.append(container)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def safe_log_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-")


def save_startup_error_log(model_name: str, logs: str) -> Path:
    ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    stem = f"error-{safe_log_model_name(model_name)}-{timestamp}"
    path = ERROR_LOG_DIR / f"{stem}.log"
    suffix = 2
    while path.exists():
        path = ERROR_LOG_DIR / f"{stem}-{suffix}.log"
        suffix += 1
    path.write_text(logs, encoding="utf-8")
    return path


def gib_to_mb(value: float) -> int:
    return round(value * 1024)


def parse_vllm_startup_memory_error(
    logs: str,
) -> VLLMStartupMemoryError | None:
    match = re.search(
        r"Free memory on device "
        r"\((?P<free>[0-9.]+)/(?P<total>[0-9.]+) GiB\) "
        r"on startup is less than desired GPU memory utilization "
        r"\((?P<utilization>[0-9.]+), (?P<requested>[0-9.]+) GiB\)",
        logs,
        flags=re.I,
    )
    if match is None:
        return None

    return VLLMStartupMemoryError(
        free_gib=float(match.group("free")),
        total_gib=float(match.group("total")),
        utilization=float(match.group("utilization")),
        requested_gib=float(match.group("requested")),
    )


def format_vllm_memory_error_diagnostics(logs: str) -> list[str]:
    error = parse_vllm_startup_memory_error(logs)
    if error is None:
        return []

    return [
        "vLLM启动显存诊断: "
        f"实际空闲={error.free_gib:.2f}/{error.total_gib:.2f} GiB "
        f"({gib_to_mb(error.free_gib)}/{gib_to_mb(error.total_gib)} MB), "
        f"期望占用={error.requested_gib:.2f} GiB "
        f"({gib_to_mb(error.requested_gib)} MB), "
        f"差额={error.shortfall_gib:.2f} GiB "
        f"({gib_to_mb(error.shortfall_gib)} MB)",
        "vLLM按当前实际空闲最多只能接受 "
        f"gpu_memory_utilization≈{error.max_passing_utilization:.2f}；"
        "如果外层选卡余量充足而这里很低，优先检查残留进程、"
        "ASCEND_VISIBLE_DEVICES映射和容器内外NPU编号是否一致。",
    ]


def classify_startup_failure(logs: str) -> tuple[str, str]:
    lowered = logs.lower()

    if "rtsmallochost" in lowered or "alloc host memory" in lowered:
        return (
            "host_memory_oom",
            "Ascend runtime 申请 host memory 失败，通常需要降低并发/上下文或启用 eager。",
        )

    if "no available memory for the cache blocks" in lowered:
        return (
            "kv_cache_budget_too_small",
            "当前并行方式下模型/KV预算过小，继续降低同一档显存只会更难启动。",
        )

    if (
        "free memory on device" in lowered
        or "aclrtmalloc" in lowered
        or "failed to allocate" in lowered
        or "out of memory" in lowered
    ):
        if parse_vllm_startup_memory_error(logs) is not None:
            return (
                "vllm_device_memory_gate",
                "vLLM启动时看到的NPU空闲显存低于本档期望占用；请对照下方诊断确认差额。",
            )
        return (
            "device_memory_oom",
            "NPU显存或运行时内存不足，已按配置尝试后续低资源档。",
        )

    if "max seq len is larger than" in lowered:
        return (
            "context_too_long_for_kv_cache",
            "最大上下文超过当前KV Cache容量，需要缩短 max_model_len。",
        )

    if "traceback" in lowered:
        return (
            "python_traceback",
            "vLLM服务启动时抛出Python异常，请查看完整错误日志。",
        )

    return (
        "startup_failed",
        "模型服务启动失败，请查看完整错误日志。",
    )


def compact_startup_logs(
    logs: str,
    max_lines: int = 80,
) -> list[str]:
    noisy_patterns = (
        "EngineCore pid=",
        "APIServer pid=",
        "[FUNC:",
        "[FILE:",
        "[LINE:",
        "FuncErrorReason",
        "ReportCallError",
        "error_message_manage.cc",
        "log_inner.cpp",
    )
    important_patterns = (
        "Traceback",
        "RuntimeError",
        "ValueError",
        "Error:",
        "ERROR",
        "Exception",
        "out of memory",
        "No available memory",
        "Free memory on device",
        "max seq len",
        "failed",
        "Failed",
    )

    kept: list[str] = []
    skipped = 0

    for line in logs.splitlines():
        stripped = line.rstrip()
        lowered = stripped.lower()
        is_noisy = any(pattern in stripped for pattern in noisy_patterns)
        is_important = any(
            pattern.lower() in lowered
            for pattern in important_patterns
        )

        if is_noisy and not is_important:
            skipped += 1
            continue

        if stripped:
            kept.append(stripped)

    if skipped:
        kept.insert(0, f"... 已隐藏 {skipped} 行 Ascend/vLLM 底层重复日志")

    if len(kept) <= max_lines:
        return kept

    head = max_lines // 2
    tail = max_lines - head
    return [
        *kept[:head],
        f"... 日志摘要过长，省略 {len(kept) - max_lines} 行；完整内容见错误日志文件",
        *kept[-tail:],
    ]


def is_memory_startup_failure(logs: str) -> bool:
    lowered = logs.lower()
    patterns = (
        "free memory on device",
        "out of memory",
        "no available memory for the cache blocks",
        "failed to allocate",
        "aclrtmalloc",
        "maximum number of tokens that can be stored in kv cache",
        "max seq len is larger than",
    )
    return any(pattern in lowered for pattern in patterns)


def is_model_budget_too_small(logs: str) -> bool:
    """这类错误继续降低同一TP的利用率只会更糟，应增加TP。"""
    lowered = logs.lower()
    patterns = (
        "no available memory for the cache blocks",
        "no available memory for cache blocks",
    )
    return any(pattern in lowered for pattern in patterns)


class ModelService:
    """
    一个模型服务实例。

    使用方式：

        with ModelService("Qwen2.5-14B-Instruct") as service:
            print(service.base_url)

    离开 with 后自动停止并删除 Docker 容器。
    """

    def __init__(
        self,
        model_name: str,
        *,
        wait_for_npu: bool = True,
        npu_wait_timeout: int = 0,
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"未知模型: {model_name}"
            )

        self.model_name = model_name
        self.container = container_name(model_name)
        self.wait_for_npu = wait_for_npu
        self.npu_wait_timeout = npu_wait_timeout

        self.port: int | None = None
        self.base_url: str | None = None
        self.lease_token: str | None = None
        self.runtime_profile: RuntimeProfile | None = None
        self.parallel_plan: ParallelPlan | None = None

    def _launch_profile(
        self,
        model: ModelConfig,
        parallel_plan: ParallelPlan,
        profile: RuntimeProfile,
    ) -> tuple[int, list[NPUInfo], bool]:
        """持锁完成测量、租约、复检和真正容器的启动。"""
        with allocation_lock():
            allocation_errors: list[str] = []
            selected: list[NPUInfo] | None = None
            shared = False

            for candidate_shared in (False, True):
                try:
                    _, selected = get_plan(
                        self.model_name,
                        parallel_plan,
                        profile,
                        shared=candidate_shared,
                    )
                    shared = candidate_shared
                    break
                except NPUAllocationError as exc:
                    allocation_errors.append(str(exc))

            if selected is None:
                raise NPUAllocationError("；".join(allocation_errors))

            if shared:
                selected = stabilize_shared_selection(
                    model,
                    parallel_plan,
                    profile,
                    selected,
                )

            lease_token = acquire_npu_leases(
                selected,
                self.container,
            )
            self.lease_token = lease_token

            try:
                time.sleep(NPU_STABILITY_CHECK_SECONDS)
                selected = confirm_selection(
                    model,
                    parallel_plan,
                    profile,
                    selected,
                    shared=shared,
                    max_hbm_drop_mb=(
                        SHARED_MAX_HBM_DROP_MB
                        if shared
                        else None
                    ),
                )
                port = find_free_port()
                cmd = build_docker_command(
                    model_name=self.model_name,
                    model=model,
                    parallel_plan=parallel_plan,
                    selected_npus=selected,
                    container_name=self.container,
                    port=port,
                    profile=profile,
                    lease_token=lease_token,
                )
                subprocess.run(cmd, check=True)
            except BaseException:
                if (
                    container_exists(self.container)
                    and container_label(
                        self.container,
                        "yuyi.npu-lease-token",
                    ) == lease_token
                ):
                    subprocess.run(
                        ["docker", "rm", "-f", self.container],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                release_npu_leases(token=lease_token)
                self.lease_token = None
                raise

        return port, selected, shared

    def start(self) -> str:
        """
        自动选择 NPU、启动模型并等待健康检查通过。
        """
        if container_exists(self.container):
            raise RuntimeError(
                f"容器 {self.container} 已存在，"
                "请先确认它是否仍在使用"
            )

        model = MODELS[self.model_name]
        launch_attempts = [
            (parallel_plan, profile)
            for parallel_plan in model.parallel_plans
            for profile in parallel_plan.runtime_profiles
        ]
        deadline = (
            None
            if self.npu_wait_timeout <= 0
            else time.monotonic() + self.npu_wait_timeout
        )
        last_wait_message_at = 0.0
        unusable_plans: set[str] = set()

        while True:
            allocation_errors: list[str] = []

            for attempt_index, (parallel_plan, profile) in enumerate(
                launch_attempts
            ):
                if parallel_plan.name in unusable_plans:
                    continue
                try:
                    port, selected, shared = self._launch_profile(
                        model,
                        parallel_plan,
                        profile,
                    )
                except NPUAllocationError as exc:
                    allocation_errors.append(str(exc))
                    continue

                selected_text = ",".join(
                    str(npu.device_id) for npu in selected
                )
                required_text = ",".join(
                    str(
                        required_free_hbm_mb(
                            model,
                            npu,
                            profile,
                            parallel_plan,
                            shared=shared,
                        )
                    )
                    for npu in selected
                )

                print()
                print(f"模型: {self.model_name}")
                print(f"并行计划: {parallel_plan.name}")
                print(f"运行档位: {profile.name}")
                print(f"选卡模式: {'共享卡' if shared else '独占卡'}")
                print(
                    "显存利用率: "
                    f"{profile.gpu_memory_utilization:.2f}"
                )
                print(f"最大上下文: {profile.max_model_len}")
                if profile.enforce_eager:
                    print("执行模式: eager（节省图缓存，速度可能较低）")
                print(f"TP: {parallel_plan.tp}")
                print(f"PP: {parallel_plan.pp}")
                print(f"NPU 数量: {parallel_plan.num_npus}")
                print(f"NPU: {selected_text}")
                print(f"每卡准入显存: {required_text} MB")
                print(
                    "\n".join(
                        format_npu_memory_diagnostics(
                            model,
                            selected,
                            profile,
                            parallel_plan,
                            shared=shared,
                        )
                    )
                )
                if shared:
                    print(f"共享安全预留: {SHARED_RESERVE_HBM_MB} MB")
                print(f"端口: {port}")
                print(f"容器: {self.container}")

                if model.reasoning_parser:
                    print(
                        "Reasoning parser: "
                        f"{model.reasoning_parser}"
                    )

                print()
                print("正在启动模型...")

                try:
                    wait_health(self.container, port)
                except BaseException as exc:
                    logs = get_container_logs(self.container, tail=None)
                    error_log_path = save_startup_error_log(
                        self.model_name,
                        logs,
                    )
                    failure_type, failure_hint = classify_startup_failure(logs)
                    compact_logs = compact_startup_logs(logs)
                    print()
                    print("模型服务启动失败")
                    print(f"错误类型: {failure_type}")
                    print(f"处理建议: {failure_hint}")
                    print(f"完整日志: {error_log_path.resolve()}")
                    memory_diagnostics = format_vllm_memory_error_diagnostics(
                        logs
                    )
                    if memory_diagnostics:
                        print("\n".join(memory_diagnostics))
                    if compact_logs:
                        print("日志摘要：")
                        print("\n".join(compact_logs))
                    memory_failure = is_memory_startup_failure(logs)
                    budget_too_small = is_model_budget_too_small(logs)
                    self.stop()

                    if not isinstance(exc, Exception):
                        raise

                    if budget_too_small:
                        unusable_plans.add(parallel_plan.name)
                        next_attempt = next(
                            (
                                candidate
                                for candidate in launch_attempts[attempt_index + 1:]
                                if candidate[0].name != parallel_plan.name
                            ),
                            None,
                        )
                        if next_attempt is None:
                            raise
                        next_plan, next_profile = next_attempt
                        print()
                        print(
                            "当前TP的模型/KV预算过小，"
                            "跳过更低利用率并切换更多卡/并行方式: "
                            f"{parallel_plan.name}/{profile.name} -> "
                            f"{next_plan.name}/{next_profile.name}"
                        )
                        continue

                    if (
                        not memory_failure
                        and parallel_plan.fallback_on_startup_error
                    ):
                        unusable_plans.add(parallel_plan.name)
                        next_attempt = next(
                            (
                                candidate
                                for candidate in launch_attempts[attempt_index + 1:]
                                if candidate[0].name != parallel_plan.name
                            ),
                            None,
                        )
                        if next_attempt is None:
                            raise
                        next_plan, next_profile = next_attempt
                        print()
                        print(
                            "流水线计划启动失败，"
                            "自动回退到下一种卡数/并行方式: "
                            f"{parallel_plan.name}/{profile.name} -> "
                            f"{next_plan.name}/{next_profile.name}"
                        )
                        continue

                    next_attempt = (
                        launch_attempts[attempt_index + 1]
                        if attempt_index + 1 < len(launch_attempts)
                        else None
                    )
                    if memory_failure and next_attempt is not None:
                        next_plan, next_profile = next_attempt
                        print()
                        print(
                            "检测到显存启动失败，自动降档重试: "
                            f"{parallel_plan.name}/{profile.name} -> "
                            f"{next_plan.name}/{next_profile.name}"
                        )
                        continue
                    raise

                self.port = port
                self.base_url = f"http://127.0.0.1:{port}/v1"
                self.runtime_profile = profile
                self.parallel_plan = parallel_plan

                print()
                print("模型服务启动成功")
                print(f"BASE_URL={self.base_url}")
                return self.base_url

            if not self.wait_for_npu:
                raise NPUAllocationError("；".join(allocation_errors))

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"等待可用 NPU 超时: {self.npu_wait_timeout} 秒；"
                    + "；".join(allocation_errors)
                )

            now = time.monotonic()
            if now - last_wait_message_at >= 30:
                detail = allocation_errors[-1] if allocation_errors else "无候选卡组"
                print(
                    f"没有可立即启动的 NPU，{NPU_WAIT_POLL_SECONDS} 秒后重试: "
                    f"{detail}"
                )
                last_wait_message_at = now
            time.sleep(NPU_WAIT_POLL_SECONDS)

    def stop(self) -> None:
        """
        强制停止并删除模型容器。
        """
        exists = container_exists(self.container)
        if exists:
            print(
                "正在释放模型服务: "
                f"{self.container}"
            )

            subprocess.run(
                [
                    "docker",
                    "rm",
                    "-f",
                    self.container,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

        release_npu_leases(
            token=self.lease_token,
            container=self.container,
        )
        self.lease_token = None

        self.port = None
        self.base_url = None
        self.runtime_profile = None
        self.parallel_plan = None

        if exists:
            print("模型服务已释放")

    def __enter__(self) -> "ModelService":
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:
        self.stop()

        # False 表示如果 with 内发生异常，
        # 清理完容器后仍继续向外抛出原异常。
        return False
def plan(model_name: str) -> None:
    if model_name not in MODELS:
        raise ValueError(f"未知模型: {model_name}")

    model = MODELS[model_name]

    npus = parse_npu_smi(run_npu_smi())
    if not npus:
        raise NPUAllocationError("npu-smi 没有返回可用设备")
    holders = get_running_container_npu_holders()
    leased = active_leased_npu_ids()

    for npu in npus:
        npu.docker_holders = tuple(sorted(holders.get(npu.device_id, [])))

    print("当前 NPU：")

    for npu in npus:
        if npu.device_id in leased:
            state = "本项目已有租约"
        elif not npu.healthy:
            state = "设备状态异常"
        elif not npu.has_process and npu.aicore_pct <= EXCLUSIVE_MAX_AICORE_PCT:
            state = "可参与独占档"
        elif npu.aicore_pct <= SHARED_MAX_AICORE_PCT:
            state = "可参与共享档"
        else:
            state = "当前计算负载过高"

        process_text = "有" if npu.has_process else "无"
        holder_text = (
            ", Docker已映射:" + ",".join(holders[npu.device_id])
            if npu.device_id in holders
            else ""
        )

        print(
            f"  NPU {npu.device_id}: "
            f"free={npu.free_hbm_mb} MB, "
            f"AICore={npu.aicore_pct}%, "
            f"process={process_text}, {state}{holder_text}"
        )

    print()
    print("开始按拓扑和显存档位规划（不会启动探测容器）...")

    topology = parse_npu_topology(run_npu_topology(npus[0].device_id))
    selected = None
    selected_profile = None
    selected_plan = None
    selected_shared = None
    for candidate_plan in model.parallel_plans:
        for candidate_profile in candidate_plan.runtime_profiles:
            for candidate_shared in (False, True):
                try:
                    selected = select_npus(
                        model,
                        npus,
                        candidate_profile,
                        plan=candidate_plan,
                        topology=topology,
                        leased_ids=leased,
                        shared=candidate_shared,
                    )
                    selected_profile = candidate_profile
                    selected_plan = candidate_plan
                    selected_shared = candidate_shared
                    break
                except NPUAllocationError:
                    continue
            if selected is not None:
                break
        if selected is not None:
            break

    if selected is None or selected_profile is None or selected_plan is None:
        raise NPUAllocationError("所有运行档位都没有合格 NPU 卡组")

    print()
    print(f"模型: {model_name}")
    print(f"并行计划: {selected_plan.name}")
    print(f"TP: {selected_plan.tp}")
    print(f"PP: {selected_plan.pp}")
    print(f"需要 NPU: {selected_plan.num_npus}")
    print(f"运行档位: {selected_profile.name}")
    print(f"选卡模式: {'共享卡' if selected_shared else '独占卡'}")
    print(
        "显存利用率: "
        f"{selected_profile.gpu_memory_utilization:.2f}"
    )
    if selected_profile.enforce_eager:
        print("执行模式: eager（节省图缓存，速度可能较低）")

    print(
        "选择 NPU: "
        + ",".join(
            str(n.device_id)
            for n in selected
        )
    )
    print(
        "每卡准入显存: "
        + ",".join(
            str(
                required_free_hbm_mb(
                    model,
                    npu,
                    selected_profile,
                    selected_plan,
                    shared=bool(selected_shared),
                )
            )
            for npu in selected
        )
        + " MB"
    )
    print(
        "\n".join(
            format_npu_memory_diagnostics(
                model,
                selected,
                selected_profile,
                selected_plan,
                shared=bool(selected_shared),
            )
        )
    )

def get_running_container_npu_holders() -> dict[int, list[str]]:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )

    holders: dict[int, list[str]] = {}

    for line in result.stdout.splitlines():
        parts = line.split("\t", 1)
        if not parts:
            continue
        container_id = parts[0]
        container_display_name = parts[1] if len(parts) > 1 else container_id
        inspect = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}",
                container_id,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        for match in re.finditer(
            r"/dev/davinci(\d+)",
            inspect.stdout,
        ):
            device_id = int(match.group(1))
            holders.setdefault(device_id, []).append(container_display_name)

    return holders


def get_running_container_npus() -> set[int]:
    return set(get_running_container_npu_holders())


def main() -> None:
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    for command in [
        "plan",
        "start",
        "stop",
    ]:
        sub_parser = sub.add_parser(
            command
        )

        sub_parser.add_argument(
            "--model",
            required=True,
            choices=MODELS,
        )

        if command == "start":
            sub_parser.add_argument(
                "--no-wait",
                action="store_true",
                help="没有可用 NPU 时立即失败",
            )
            sub_parser.add_argument(
                "--npu-wait-timeout",
                type=int,
                default=0,
                help="等待 NPU 的秒数；0 表示一直等待",
            )

    args = parser.parse_args()

    if args.command == "plan":
        plan(args.model)

    elif args.command == "start":
        ModelService(
            args.model,
            wait_for_npu=not args.no_wait,
            npu_wait_timeout=args.npu_wait_timeout,
        ).start()

    elif args.command == "stop":
        ModelService(
            args.model
        ).stop()


if __name__ == "__main__":
    main()

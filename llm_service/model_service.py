from __future__ import annotations

import argparse
import re
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


IMAGE = "quay.io/ascend/vllm-ascend:v0.22.1rc1"
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.70
START_TIMEOUT = 900


@dataclass(frozen=True)
class ModelConfig:
    path: str
    tp: int
    min_free_hbm_mb: int

    pp: int = 1
    reasoning_parser: str | None = None

    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None

    @property
    def num_npus(self) -> int:
        return self.tp * self.pp


MODELS = {
    "Qwen2.5-14B-Instruct": ModelConfig(
        path="/user_home/linlujun/linlujun/model/Qwen2.5-14B-Instruct",
        tp=1,
        pp=1,
        min_free_hbm_mb=40000,
    ),

    "DeepSeek-R1-Distill-Llama-8B": ModelConfig(
        path="/user_home/linlujun/linlujun/model/DeepSeek-R1-Distill-Llama-8B",
        tp=1,
        pp=1,
        min_free_hbm_mb=30000,
        reasoning_parser="deepseek_r1",
    ),

    "DeepSeek-R1-Distill-Qwen-14B": ModelConfig(
        path="/user_home/linlujun/linlujun/model/DeepSeek-R1-Distill-Qwen-14B",
        tp=1,
        pp=1,
        min_free_hbm_mb=40000,
        reasoning_parser="deepseek_r1",
    ),

    "DeepSeek-R1-Distill-Qwen-32B": ModelConfig(
        path="/user_home/linlujun/linlujun/model/DeepSeek-R1-Distill-Qwen-32B",
        tp=2,
        pp=1,
        min_free_hbm_mb=55000,
        reasoning_parser="deepseek_r1",
    ),

    "DeepSeek-R1-Distill-Llama-70B": ModelConfig(
        path="/user_home/linlujun/linlujun/model/DeepSeek-R1-Distill-Llama-70B",
        tp=1,
        pp=3,
        min_free_hbm_mb=60000,
        reasoning_parser="deepseek_r1",
        max_num_seqs=32,
        max_num_batched_tokens=2048,
    ),
}


@dataclass
class NPUInfo:
    device_id: int
    total_hbm_mb: int
    used_hbm_mb: int
    aicore_pct: int
    has_process: bool = False

    @property
    def free_hbm_mb(self) -> int:
        return self.total_hbm_mb - self.used_hbm_mb


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


def select_npus(
    model: ModelConfig,
    npus: list[NPUInfo],
) -> list[NPUInfo]:
    candidates = [
        npu
        for npu in npus
        if not npu.has_process
        and npu.aicore_pct <= 5
        and npu.free_hbm_mb >= model.min_free_hbm_mb
    ]

    candidates.sort(
        key=lambda n: (
            -n.free_hbm_mb,
            n.aicore_pct,
            n.device_id,
        )
    )
    if len(candidates) < model.num_npus:
        raise RuntimeError(
            f"模型需要 {model.num_npus} 张候选 NPU "
            f"(TP={model.tp}, PP={model.pp})，"
            f"但当前只有 {len(candidates)} 张满足基础条件"
        )

    return candidates

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
def probe_npu(device_id: int) -> bool:
    command = [
        "docker",
        "run",
        "--rm",

        "--runtime=ascend",

        "-e",
        f"ASCEND_VISIBLE_DEVICES={device_id}",

        "-e",
        "ASCEND_RT_VISIBLE_DEVICES=0",

        "--device",
        f"/dev/davinci{device_id}:/dev/davinci0",

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

        "--entrypoint",
        "python",

        IMAGE,

        "-c",
        (
            "import torch, torch_npu; "
            "assert torch.npu.is_available(); "
            "assert torch.npu.device_count() == 1; "
            "x = torch.zeros(1024 * 1024 * 1024,dtype=torch.float16).npu();"
            "assert x.device.type == 'npu'"
        ),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode == 0:
        return True

    print(
        f"NPU {device_id} 探测失败:"
    )

    output_parts = []

    if result.stdout:
        output_parts.extend(
            result.stdout.strip().splitlines()
        )

    if result.stderr:
        output_parts.extend(
            result.stderr.strip().splitlines()
        )

    for line in output_parts[-30:]:
        print(f"  {line}")

    return False
def select_usable_npus(
    model: ModelConfig,
    npus: list[NPUInfo],
) -> list[NPUInfo]:
    candidates = select_npus(
        model,
        npus,
    )

    selected: list[NPUInfo] = []

    for npu in candidates:
        print(
            f"探测 NPU {npu.device_id} ... ",
            end="",
            flush=True,
        )

        if probe_npu(npu.device_id):
            print("可用")
            selected.append(npu)
        else:
            print("不可用，跳过")

        if len(selected) >= model.num_npus:
            break

    if len(selected) < model.num_npus:
        raise RuntimeError(
            f"模型需要 {model.num_npus} 张可用 NPU "
            f"(TP={model.tp}, PP={model.pp})，"
            f"但实际探测后只有 {len(selected)} 张可用"
        )

    return selected
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

def get_plan(
    model_name: str,
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

    npus = parse_npu_smi(
        run_npu_smi()
    )

    selected = select_usable_npus(
        model,
        npus,
    )

    return model, selected
def build_docker_command(
    model_name: str,
    model: ModelConfig,
    selected_npus: list[NPUInfo],
    port: int,
    container_name: str,
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
        str(model.tp),

        "--pipeline-parallel-size",
        str(model.pp),

        "--max-model-len",
        str(MAX_MODEL_LEN),

        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
    ]

    if model.reasoning_parser:
        command += [
            "--reasoning-parser",
            model.reasoning_parser,
        ]

    if model.max_num_seqs is not None:
        command += [
            "--max-num-seqs",
            str(model.max_num_seqs),
        ]

    if model.max_num_batched_tokens is not None:
        command += [
            "--max-num-batched-tokens",
            str(model.max_num_batched_tokens),
        ]
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

class ModelService:
    """
    一个模型服务实例。

    使用方式：

        with ModelService("Qwen2.5-14B-Instruct") as service:
            print(service.base_url)

    离开 with 后自动停止并删除 Docker 容器。
    """

    def __init__(self, model_name: str):
        if model_name not in MODELS:
            raise ValueError(
                f"未知模型: {model_name}"
            )

        self.model_name = model_name
        self.container = container_name(model_name)

        self.port: int | None = None
        self.base_url: str | None = None

    def start(self) -> str:
        """
        自动选择 NPU、启动模型并等待健康检查通过。
        """
        if container_exists(self.container):
            raise RuntimeError(
                f"容器 {self.container} 已存在，"
                "请先确认它是否仍在使用"
            )

        model, selected = get_plan(
            self.model_name
        )

        port = find_free_port()

        cmd = build_docker_command(
            model_name=self.model_name,
            model=model,
            selected_npus=selected,
            container_name=self.container,
            port=port,
        )

        print()
        print(f"模型: {self.model_name}")
        print(f"TP: {model.tp}")
        print(f"PP: {model.pp}")
        print(f"NPU 数量: {model.num_npus}")
        selected_text = ",".join(
            str(npu.device_id)
            for npu in selected
        )

        print(f"NPU: {selected_text}")
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
            subprocess.run(
                cmd,
                check=True,
            )

            wait_health(
                self.container,
                port,
            )

        except Exception:
            print()
            print("模型服务启动失败，最近日志：")

            print_container_logs(
                self.container,
                tail=500
            )

            self.stop()

            raise

        self.port = port
        self.base_url = (
            f"http://127.0.0.1:{port}/v1"
        )

        print()
        print("模型服务启动成功")
        print(
            f"BASE_URL={self.base_url}"
        )

        return self.base_url

    def stop(self) -> None:
        """
        强制停止并删除模型容器。
        """
        if not container_exists(self.container):
            return

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

        self.port = None
        self.base_url = None

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
    docker_occupied = get_running_container_npus()

    print("当前 NPU：")

    for npu in npus:
        reasons = []

        if npu.has_process:
            reasons.append("npu-smi进程")

        if npu.aicore_pct > 5:
            reasons.append(f"AICore={npu.aicore_pct}%")

        if npu.free_hbm_mb < model.min_free_hbm_mb:
            reasons.append("显存不足")

        notes = []

        if npu.device_id in docker_occupied:
            notes.append("Docker已映射")

        if reasons:
            state = "不可候选(" + ",".join(reasons) + ")"
        else:
            state = "候选"

        if notes:
            state += "[" + ",".join(notes) + "]"

        print(
            f"  NPU {npu.device_id}: "
            f"free={npu.free_hbm_mb} MB, "
            f"AICore={npu.aicore_pct}%, "
            f"{state}"
        )

    print()
    print("开始实际可用性探测...")

    selected = select_usable_npus(model, npus)

    print()
    print(f"模型: {model_name}")
    print(f"TP: {model.tp}")
    print(f"PP: {model.pp}")
    print(f"需要 NPU: {model.num_npus}")

    print(
        "选择 NPU: "
        + ",".join(
            str(n.device_id)
            for n in selected
        )
    )

def get_running_container_npus() -> set[int]:
    result = subprocess.run(
        ["docker", "ps", "-q"],
        capture_output=True,
        text=True,
        check=True,
    )

    occupied: set[int] = set()

    for container_id in result.stdout.split():
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
            occupied.add(int(match.group(1)))

    return occupied
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

    args = parser.parse_args()

    if args.command == "plan":
        plan(args.model)

    elif args.command == "start":
        ModelService(
            args.model
        ).start()

    elif args.command == "stop":
        ModelService(
            args.model
        ).stop()


if __name__ == "__main__":
    main()

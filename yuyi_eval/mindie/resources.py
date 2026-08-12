"""空闲 NPU / 端口自动分配。"""

from __future__ import annotations

import json
import re
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class NpuStatus:
    npu_id: int
    healthy: bool
    hbm_capacity_mb: int
    hbm_usage_pct: float
    aicore_usage_pct: float
    process_mem_mb: int
    process_names: list[str] = field(default_factory=list)
    docker_holders: list[str] = field(default_factory=list)

    @property
    def free_hbm_mb(self) -> float:
        return self.hbm_capacity_mb * (1.0 - self.hbm_usage_pct / 100.0)

    @property
    def device_path(self) -> str:
        return f"/dev/davinci{self.npu_id}"


@dataclass
class Allocation:
    tmodel: str
    physical_npu_ids: list[int]
    logical_npu_ids: list[int]
    port: int
    management_port: int
    metrics_port: int
    inter_comm_port: int
    multi_nodes_infer_port: int
    management_ip: str
    container_name: str
    base_url: str
    log_file: str
    config_host: str
    free_hbm_mb: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(cmd: list[str], timeout: float = 30.0) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if r.returncode != 0:
        raise RuntimeError(
            f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr or r.stdout}"
        )
    return r.stdout


def _parse_kv_block(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def list_npu_ids() -> list[int]:
    text = _run(["npu-smi", "info", "-m"])
    ids: list[int] = []
    for line in text.splitlines():
        cols = [c.strip() for c in line.split() if c.strip()]
        if len(cols) < 4:
            continue
        if not cols[0].isdigit():
            continue
        # Chip Logic ID == "-" 表示 MCU 行
        if cols[2] == "-":
            continue
        if "Ascend" not in cols[3] and "910" not in cols[3]:
            # 兼容 Chip Name 列
            if not any("Ascend" in c or "910" in c for c in cols[3:]):
                continue
        nid = int(cols[0])
        if nid not in ids:
            ids.append(nid)
    if not ids:
        raise RuntimeError("npu-smi info -m 未解析到 NPU")
    return ids


def docker_davinci_holders() -> dict[int, list[str]]:
    """running 容器 → 挂载的物理 davinci id。"""
    holders: dict[int, list[str]] = {}
    try:
        names = _run(["docker", "ps", "--format", "{{.Names}}"]).splitlines()
    except Exception:
        return holders
    for name in names:
        name = name.strip()
        if not name:
            continue
        try:
            raw = _run(
                ["docker", "inspect", name, "--format", "{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}"]
            )
        except Exception:
            continue
        for tok in raw.split():
            m = re.search(r"/dev/davinci(\d+)$", tok)
            if not m:
                continue
            nid = int(m.group(1))
            holders.setdefault(nid, []).append(name)
    return holders


def query_npu(npu_id: int, holders: Optional[dict[int, list[str]]] = None) -> NpuStatus:
    holders = holders if holders is not None else docker_davinci_holders()
    health = _parse_kv_block(_run(["npu-smi", "info", "-t", "health", "-i", str(npu_id)]))
    usages = _parse_kv_block(_run(["npu-smi", "info", "-t", "usages", "-i", str(npu_id)]))
    memory = _parse_kv_block(_run(["npu-smi", "info", "-t", "memory", "-i", str(npu_id)]))
    proc_text = _run(["npu-smi", "info", "-t", "proc-mem", "-i", str(npu_id)])

    healthy = health.get("Health", "").upper() == "OK"
    cap = int(float(memory.get("HBM Capacity(MB)", "0") or 0))
    usage_pct = float(usages.get("HBM Usage Rate(%)", "0") or 0)
    aicore = float(usages.get("Aicore Usage Rate(%)", "0") or 0)

    process_names: list[str] = []
    process_mem = 0
    for m in re.finditer(
        r"Process id:(\d+)\s+Process name:(\S+)\s+Process memory\(MB\):(\d+)",
        proc_text,
    ):
        process_names.append(m.group(2))
        process_mem += int(m.group(3))

    return NpuStatus(
        npu_id=npu_id,
        healthy=healthy,
        hbm_capacity_mb=cap,
        hbm_usage_pct=usage_pct,
        aicore_usage_pct=aicore,
        process_mem_mb=process_mem,
        process_names=process_names,
        docker_holders=list(holders.get(npu_id, [])),
    )


def list_npu_status() -> list[NpuStatus]:
    holders = docker_davinci_holders()
    return [query_npu(i, holders) for i in list_npu_ids()]


def _port_free(port: int, host: str = "0.0.0.0") -> bool:
    # 同时探测 IPv4 bind；management 可能绑 127.0.0.x
    for bind_host in (host, "127.0.0.1"):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((bind_host, port))
        except OSError:
            return False
        finally:
            s.close()
    return True


def pick_free_ports(count: int = 3, start: int = 18500, end: int = 18999) -> list[int]:
    """选一段连续空闲端口：api / management / metrics。"""
    for p in range(start, end - count + 2):
        cand = list(range(p, p + count))
        if all(_port_free(x) for x in cand):
            # interComm / multiNodes 也尽量空闲
            if not _port_free(p + 100) or not _port_free(p + 99):
                continue
            return cand
    raise RuntimeError(f"在 {start}-{end} 找不到 {count} 个连续空闲端口")


def pick_management_ip(used: Optional[set[str]] = None) -> str:
    used = used or set()
    for i in range(2, 254):
        ip = f"127.0.0.{i}"
        if ip in used:
            continue
        # 粗测：该 IP 上常见 management 端口是否已被占用由 port 选择兜底
        return ip
    raise RuntimeError("无可用 management_ip")


def _runtime_claimed_npus(runtime_dir: Path, exclude_tmodel: str) -> set[int]:
    claimed: set[int] = set()
    if not runtime_dir.is_dir():
        return claimed
    for path in runtime_dir.glob("*.json"):
        if path.stem == exclude_tmodel:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cname = data.get("container_name")
        if cname:
            try:
                running = _run(["docker", "ps", "--format", "{{.Names}}"])
                if cname not in running.splitlines():
                    continue
            except Exception:
                pass
        for nid in data.get("physical_npu_ids", []):
            claimed.add(int(nid))
    return claimed


def _foreign_holders(n: NpuStatus, our_containers: set[str]) -> list[str]:
    return [h for h in n.docker_holders if h not in our_containers]


def _is_candidate(
    n: NpuStatus,
    *,
    min_free_hbm_mb: int,
    max_aicore_pct: float,
    max_process_mem_mb: int,
    allow_foreign_mount: bool,
    our_containers: set[str],
) -> bool:
    if not n.healthy:
        return False
    if not Path(n.device_path).exists():
        return False
    if n.free_hbm_mb < min_free_hbm_mb:
        return False
    if n.aicore_usage_pct > max_aicore_pct:
        return False
    if n.process_mem_mb > max_process_mem_mb:
        return False
    if _foreign_holders(n, our_containers) and not allow_foreign_mount:
        return False
    return True


def select_npus(
    world_size: int,
    *,
    min_free_hbm_mb: int,
    max_aicore_pct: float = 5.0,
    max_process_mem_mb: int = 512,
    prefer_unmounted: bool = True,
    exclude_ids: Optional[set[int]] = None,
    our_containers: Optional[set[str]] = None,
) -> list[NpuStatus]:
    exclude_ids = exclude_ids or set()
    our_containers = our_containers or set()
    status = [n for n in list_npu_status() if n.npu_id not in exclude_ids]

    def rank(n: NpuStatus) -> tuple:
        foreign = _foreign_holders(n, our_containers)
        return (
            len(foreign),
            -n.free_hbm_mb,
            n.aicore_usage_pct,
            n.process_mem_mb,
            n.npu_id,
        )

    strict = [
        n
        for n in status
        if _is_candidate(
            n,
            min_free_hbm_mb=min_free_hbm_mb,
            max_aicore_pct=max_aicore_pct,
            max_process_mem_mb=max_process_mem_mb,
            allow_foreign_mount=False,
            our_containers=our_containers,
        )
    ]
    pool = strict
    if len(pool) < world_size:
        # 回退：允许被其他容器挂载、但无实际进程且 HBM 足够的卡
        # （本机常见：他人 mindie 容器 --device 挂了全部卡但未在该卡跑进程）
        pool = [
            n
            for n in status
            if _is_candidate(
                n,
                min_free_hbm_mb=min_free_hbm_mb,
                max_aicore_pct=max_aicore_pct,
                max_process_mem_mb=max_process_mem_mb,
                allow_foreign_mount=True,
                our_containers=our_containers,
            )
        ]

    pool = sorted(pool, key=rank)
    if len(pool) < world_size:
        detail = "\n".join(
            f"  NPU{n.npu_id}: free_hbm={n.free_hbm_mb:.0f}MB "
            f"aicore={n.aicore_usage_pct}% proc_mem={n.process_mem_mb}MB "
            f"procs={n.process_names} holders={n.docker_holders}"
            for n in status
        )
        raise RuntimeError(
            f"需要 {world_size} 张空闲卡（每卡 ≥{min_free_hbm_mb}MB 空闲 HBM），"
            f"仅找到 {len(pool)} 张。\n当前 NPU:\n{detail}"
        )
    return pool[:world_size]


def allocate_for_model(
    tmodel: str,
    mindie_cfg: dict[str, Any],
    *,
    runtime_dir: Path,
    config_host: Path,
    container_name: Optional[str] = None,
    port_start: int = 18500,
    port_end: int = 18999,
) -> Allocation:
    world_size = int(mindie_cfg["world_size"])
    min_free = int(mindie_cfg.get("min_free_hbm_mb", 50000))
    max_aicore = float(mindie_cfg.get("max_aicore_pct", 5.0))
    max_proc = int(mindie_cfg.get("max_process_mem_mb", 512))

    claimed = _runtime_claimed_npus(runtime_dir, exclude_tmodel=tmodel)
    # 本模型旧容器占用的卡可回收（启动脚本会先 docker rm）
    cname = container_name or f"mindie-{tmodel.lower()}"
    npus = select_npus(
        world_size,
        min_free_hbm_mb=min_free,
        max_aicore_pct=max_aicore,
        max_process_mem_mb=max_proc,
        exclude_ids=claimed,
        our_containers={cname},
    )
    # 若选到的卡仍被本容器占着，允许（即将重建）
    ports = pick_free_ports(3, start=port_start, end=port_end)
    # management_ip：避开 runtime 已占用
    used_ips: set[str] = set()
    if runtime_dir.is_dir():
        for path in runtime_dir.glob("*.json"):
            try:
                used_ips.add(json.loads(path.read_text(encoding="utf-8")).get("management_ip", ""))
            except Exception:
                pass
    mgmt_ip = pick_management_ip(used_ips)

    physical = [n.npu_id for n in npus]
    logical = list(range(world_size))
    port, mgmt_port, metrics_port = ports
    return Allocation(
        tmodel=tmodel,
        physical_npu_ids=physical,
        logical_npu_ids=logical,
        port=port,
        management_port=mgmt_port,
        metrics_port=metrics_port,
        inter_comm_port=port + 100,
        multi_nodes_infer_port=port + 99,
        management_ip=mgmt_ip,
        container_name=cname,
        base_url=f"http://127.0.0.1:{port}/v1",
        log_file=mindie_cfg.get("log_file") or f"/tmp/mindieservice_{tmodel}.log",
        config_host=str(config_host),
        free_hbm_mb=[round(n.free_hbm_mb, 1) for n in npus],
    )

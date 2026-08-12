"""MindIE 服务配置渲染与 runtime endpoint 持久化。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from yuyi_eval.mindie.resources import Allocation


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_runtime_dir() -> Path:
    return repo_root() / "config" / "mindie" / "runtime"


def runtime_path(tmodel: str, runtime_dir: Optional[Path] = None) -> Path:
    return (runtime_dir or default_runtime_dir()) / f"{tmodel}.json"


def load_runtime(tmodel: str, runtime_dir: Optional[Path] = None) -> Optional[dict[str, Any]]:
    path = runtime_path(tmodel, runtime_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_runtime(alloc: Allocation, extra: Optional[dict[str, Any]] = None) -> Path:
    path = Path(alloc.config_host).parent.parent / "runtime" / f"{alloc.tmodel}.json"
    # config_host 形如 config/mindie/runtime/../generated → 统一写到 runtime_dir
    path = default_runtime_dir() / f"{alloc.tmodel}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = alloc.to_dict()
    if extra:
        data.update(extra)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def render_mindie_service_config(
    *,
    weight_path: str,
    service_model_name: str,
    world_size: int,
    logical_npu_ids: list[int],
    port: int,
    management_port: int,
    metrics_port: int,
    management_ip: str,
    inter_comm_port: int,
    multi_nodes_infer_port: int,
    max_seq_len: int = 4096,
    max_input_token_len: int = 2048,
    max_batch_size: int = 8,
    npu_mem_size: int = -1,
) -> dict[str, Any]:
    return {
        "Version": "1.0.0",
        "ServerConfig": {
            "ipAddress": "0.0.0.0",
            "managementIpAddress": management_ip,
            "port": port,
            "managementPort": management_port,
            "metricsPort": metrics_port,
            "allowAllZeroIpListening": True,
            "maxLinkNum": 1000,
            "httpsEnabled": False,
            "fullTextEnabled": False,
            "inferMode": "standard",
            "interCommTLSEnabled": False,
            "interCommPort": inter_comm_port,
            "openAiSupport": "vllm",
            "tokenTimeout": 600,
            "e2eTimeout": 600,
            "distDPServerEnabled": False,
        },
        "BackendConfig": {
            "backendName": "mindieservice_llm_engine",
            "modelInstanceNumber": 1,
            "npuDeviceIds": [logical_npu_ids],
            "tokenizerProcessNumber": 8,
            "multiNodesInferEnabled": False,
            "multiNodesInferPort": multi_nodes_infer_port,
            "interNodeTLSEnabled": False,
            "ModelDeployConfig": {
                "maxSeqLen": max_seq_len,
                "maxInputTokenLen": max_input_token_len,
                "truncation": False,
                "ModelConfig": [
                    {
                        "modelInstanceType": "Standard",
                        "modelName": service_model_name,
                        "modelWeightPath": weight_path,
                        "worldSize": world_size,
                        "cpuMemSize": 0,
                        "npuMemSize": npu_mem_size,
                        "backendType": "atb",
                        "trustRemoteCode": True,
                        "async_scheduler_wait_time": 120,
                        "kv_trans_timeout": 10,
                        "kv_link_timeout": 1080,
                    }
                ],
            },
            "ScheduleConfig": {
                "templateType": "Standard",
                "templateName": "Standard_LLM",
                "cacheBlockSize": 128,
                "maxPrefillBatchSize": min(16, max_batch_size),
                "maxPrefillTokens": max_seq_len,
                "prefillTimeMsPerReq": 150,
                "prefillPolicyType": 0,
                "decodeTimeMsPerReq": 50,
                "decodePolicyType": 0,
                "maxBatchSize": max_batch_size,
                "maxIterTimes": min(4096, max_seq_len),
                "maxPreemptCount": 0,
                "supportSelectBatch": False,
                "maxQueueDelayMicroseconds": 5000,
                "maxFirstTokenWaitTime": 60000,
            },
        },
        "LogConfig": {
            "dynamicLogLevel": "",
            "dynamicLogLevelValidHours": 2,
            "dynamicLogLevelValidTime": "",
        },
        "EnableDynamicAdjustTimeoutConfig": False,
    }


def build_service_config_from_registry(
    tmodel: str,
    alloc: Allocation,
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    mindie = model_cfg["mindie"]
    max_seq = int(mindie.get("max_seq_len", 4096))
    max_input = int(mindie.get("max_input_token_len", min(2048, max_seq)))
    max_batch = int(mindie.get("max_batch_size", 8))
    weight = mindie.get("weight_path_override") or model_cfg["weight_path"]
    return render_mindie_service_config(
        weight_path=weight,
        service_model_name=mindie["service_model_name"],
        world_size=int(mindie["world_size"]),
        logical_npu_ids=alloc.logical_npu_ids,
        port=alloc.port,
        management_port=alloc.management_port,
        metrics_port=alloc.metrics_port,
        management_ip=alloc.management_ip,
        inter_comm_port=alloc.inter_comm_port,
        multi_nodes_infer_port=alloc.multi_nodes_infer_port,
        max_seq_len=max_seq,
        max_input_token_len=max_input,
        max_batch_size=max_batch,
        npu_mem_size=int(mindie.get("npu_mem_size", -1)),
    )

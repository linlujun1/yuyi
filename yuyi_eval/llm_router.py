"""统一 LLM 客户端；模型注册表见 config/models.yaml。

MindIE 本地服务的实际 base_url 以 config/mindie/runtime/<tmodel>.json 为准
（由 scripts/mindie/start.sh 自动分配端口后写入）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from openai import OpenAI


@dataclass
class LLMEndpoint:
    tmodel: str
    base_url: str
    model: str
    api_key: str = "EMPTY"
    display_name: str = ""


def default_models_path() -> str:
    root = Path(__file__).resolve().parent.parent
    return str(root / "config" / "models.yaml")


def load_models_registry(config_path: Optional[str] = None) -> dict[str, Any]:
    path = config_path or default_models_path()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_api_key(raw: str) -> str:
    if raw.startswith("${") and raw.endswith("}"):
        return os.environ.get(raw[2:-1], "EMPTY")
    return raw or "EMPTY"


def get_tmodel_names(config_path: Optional[str] = None) -> list[str]:
    reg = load_models_registry(config_path)
    return list(reg.get("models", {}).keys())


def get_endpoint(tmodel: str, config_path: Optional[str] = None) -> LLMEndpoint:
    reg = load_models_registry(config_path)
    models = reg.get("models", {})
    if tmodel not in models:
        raise KeyError(f"未知 tmodel '{tmodel}'，可选: {list(models)}")
    cfg = models[tmodel]
    ep = cfg["endpoint"]
    base_url = ep["base_url"]

    # MindIE runtime 覆盖（自动选端口后的真实地址）
    if cfg.get("mindie"):
        try:
            from yuyi_eval.mindie.config import load_runtime

            rt = load_runtime(tmodel)
            if rt and rt.get("base_url"):
                base_url = rt["base_url"]
        except Exception:
            pass

    if base_url.rstrip("/").endswith("://127.0.0.1:0") or base_url.endswith(":0/v1"):
        raise RuntimeError(
            f"{tmodel} 尚未启动或无 runtime endpoint。"
            f"请先运行: bash scripts/mindie/start.sh --tmodel {tmodel}"
        )

    return LLMEndpoint(
        tmodel=tmodel,
        base_url=base_url,
        model=ep["model"],
        api_key=resolve_api_key(ep.get("api_key", "EMPTY")),
        display_name=cfg.get("display_name", tmodel),
    )


def list_mindie_models(config_path: Optional[str] = None) -> dict[str, dict[str, Any]]:
    reg = load_models_registry(config_path)
    out: dict[str, dict[str, Any]] = {}
    for name, cfg in reg.get("models", {}).items():
        mindie = cfg.get("mindie")
        if mindie:
            out[name] = {
                "weight_path": cfg["weight_path"],
                **mindie,
            }
    return out


def render_mindie_config(tmodel: str, config_path: Optional[str] = None) -> dict[str, Any]:
    """基于 registry + runtime（若有）渲染 MindIE 服务 JSON。"""
    from yuyi_eval.mindie.config import build_service_config_from_registry, load_runtime
    from yuyi_eval.mindie.resources import Allocation

    reg = load_models_registry(config_path)
    cfg = reg["models"][tmodel]
    if not cfg.get("mindie"):
        raise ValueError(f"{tmodel} 无 mindie 配置")
    rt = load_runtime(tmodel)
    if not rt:
        raise RuntimeError(
            f"{tmodel} 无 runtime，无法渲染。请先: python -m yuyi_eval.mindie.cli plan --tmodel {tmodel}"
        )
    alloc = Allocation(
        tmodel=tmodel,
        physical_npu_ids=list(rt["physical_npu_ids"]),
        logical_npu_ids=list(rt["logical_npu_ids"]),
        port=int(rt["port"]),
        management_port=int(rt["management_port"]),
        metrics_port=int(rt["metrics_port"]),
        inter_comm_port=int(rt.get("inter_comm_port", int(rt["port"]) + 100)),
        multi_nodes_infer_port=int(rt.get("multi_nodes_infer_port", int(rt["port"]) + 99)),
        management_ip=rt["management_ip"],
        container_name=rt["container_name"],
        base_url=rt["base_url"],
        log_file=rt["log_file"],
        config_host=rt.get("config_host", ""),
        free_hbm_mb=list(rt.get("free_hbm_mb", [])),
    )
    return build_service_config_from_registry(tmodel, alloc, cfg)


class LLMRouter:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or default_models_path()
        self._clients: dict[str, OpenAI] = {}

    def list_tmodels(self) -> list[str]:
        return get_tmodel_names(self.config_path)

    def _client(self, tmodel: str) -> tuple[OpenAI, LLMEndpoint]:
        ep = get_endpoint(tmodel, self.config_path)
        # base_url 可能因重新部署变化，始终按最新 endpoint 建/换客户端
        cached = self._clients.get(tmodel)
        if cached is None or getattr(cached, "_yuyi_base_url", None) != ep.base_url:
            client = OpenAI(api_key=ep.api_key or "EMPTY", base_url=ep.base_url)
            client._yuyi_base_url = ep.base_url  # type: ignore[attr-defined]
            self._clients[tmodel] = client
        return self._clients[tmodel], ep

    def chat(
        self,
        tmodel: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
    ) -> str:
        client, ep = self._client(tmodel)
        resp = client.chat.completions.create(
            model=ep.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return resp.choices[0].message.content or ""

    def dump_mindie_config(self, tmodel: str, out_path: str) -> None:
        cfg = render_mindie_config(tmodel, self.config_path)
        Path(out_path).write_text(
            json.dumps(cfg, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )

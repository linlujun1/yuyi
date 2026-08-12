"""MindIE 部署 CLI：plan / wait / status。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from yuyi_eval.llm_router import load_models_registry
from yuyi_eval.mindie.config import (
    build_service_config_from_registry,
    default_runtime_dir,
    load_runtime,
    save_runtime,
)
from yuyi_eval.mindie.resources import allocate_for_model, list_npu_status


def cmd_plan(args: argparse.Namespace) -> int:
    reg = load_models_registry()
    if args.tmodel not in reg.get("models", {}):
        print(f"unknown tmodel: {args.tmodel}", file=sys.stderr)
        return 1
    model_cfg = reg["models"][args.tmodel]
    mindie = model_cfg.get("mindie")
    if not mindie:
        print(f"{args.tmodel} 无 mindie 部署段", file=sys.stderr)
        return 1

    runtime_dir = Path(args.runtime_dir) if args.runtime_dir else default_runtime_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config_host = runtime_dir / f"config_{args.tmodel}.json"
    container = args.container or f"mindie-{args.tmodel.lower()}"

    alloc = allocate_for_model(
        args.tmodel,
        mindie,
        runtime_dir=runtime_dir,
        config_host=config_host,
        container_name=container,
    )
    service_cfg = build_service_config_from_registry(args.tmodel, alloc, model_cfg)
    config_host.write_text(json.dumps(service_cfg, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    # MindIE daemon 会对配置 chown/chmod，宿主机侧先放宽
    config_host.chmod(0o644)

    rt_path = save_runtime(
        alloc,
        extra={
            "service_model_name": mindie["service_model_name"],
            "weight_path": mindie.get("weight_path_override") or model_cfg["weight_path"],
            "world_size": int(mindie["world_size"]),
        },
    )

    # 给 bash 用的一行导出
    print(f"TMODEL={alloc.tmodel}")
    print(f"CONTAINER_NAME={alloc.container_name}")
    print(f"PHYSICAL_NPU_IDS={','.join(map(str, alloc.physical_npu_ids))}")
    print(f"LOGICAL_NPU_IDS={','.join(map(str, alloc.logical_npu_ids))}")
    print(f"ASCEND_VISIBLE={','.join(map(str, alloc.logical_npu_ids))}")
    print(f"PORT={alloc.port}")
    print(f"MANAGEMENT_PORT={alloc.management_port}")
    print(f"METRICS_PORT={alloc.metrics_port}")
    print(f"MANAGEMENT_IP={alloc.management_ip}")
    print(f"BASE_URL={alloc.base_url}")
    print(f"LOG_FILE={alloc.log_file}")
    print(f"CONFIG_HOST={config_host}")
    print(f"RUNTIME_JSON={rt_path}")
    print(f"FREE_HBM_MB={','.join(map(str, alloc.free_hbm_mb))}")
    return 0


def cmd_npu_status(_: argparse.Namespace) -> int:
    rows = list_npu_status()
    print(json.dumps([r.__dict__ for r in rows], ensure_ascii=False, indent=2))
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    rt = load_runtime(args.tmodel)
    if not rt:
        print(f"no runtime for {args.tmodel}", file=sys.stderr)
        return 1
    url = rt["base_url"].rstrip("/") + "/models"
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    print(f"READY {url}")
                    print(body)
                    return 0
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(args.interval)
        print(f"  ... waiting {url}", flush=True)
    print(f"TIMEOUT waiting {url}", file=sys.stderr)
    return 1


def cmd_endpoint(args: argparse.Namespace) -> int:
    rt = load_runtime(args.tmodel)
    if not rt:
        print(f"no runtime for {args.tmodel}", file=sys.stderr)
        return 1
    print(rt["base_url"])
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="yuyi_eval.mindie")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="自动选卡/端口并写出 runtime + MindIE config")
    p_plan.add_argument("--tmodel", required=True)
    p_plan.add_argument("--container", default="")
    p_plan.add_argument("--runtime-dir", default="")
    p_plan.set_defaults(func=cmd_plan)

    p_st = sub.add_parser("npu-status", help="打印各卡空闲情况")
    p_st.set_defaults(func=cmd_npu_status)

    p_wait = sub.add_parser("wait", help="等待 runtime endpoint 就绪")
    p_wait.add_argument("--tmodel", required=True)
    p_wait.add_argument("--timeout", type=int, default=600)
    p_wait.add_argument("--interval", type=int, default=10)
    p_wait.set_defaults(func=cmd_wait)

    p_ep = sub.add_parser("endpoint", help="打印 runtime base_url")
    p_ep.add_argument("--tmodel", required=True)
    p_ep.set_defaults(func=cmd_endpoint)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

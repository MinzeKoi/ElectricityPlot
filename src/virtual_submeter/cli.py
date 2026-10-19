"""Initialization commands; no data import, training, or predictions yet."""

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="低交互可信虚拟分表：初始化入口")
    parser.add_argument("command", choices=("status", "validate-config"), nargs="?", default="status")
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--json", action="store_true", help="输出结构化状态")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as error:
        if args.json:
            print(json.dumps({"status": "invalid_config", "error": str(error)}, ensure_ascii=False))
        else:
            print(f"配置检查失败：{error}", file=sys.stderr)
        return 2

    payload = {
        "status": "configuration_valid",
        "project": config.name,
        "version": __version__,
        "stage": "scaffold",
        "dataset_candidate": config.dataset,
        "target_appliances": list(config.target_appliances),
        "query_budgets": list(config.query_budgets),
        "data_validation": "not_run",
        "training": "not_implemented",
        "experiments": "not_run",
        "notice": "仅完成配置结构检查；尚未核实数据、训练模型或取得实验结果。",
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "validate-config":
        print("配置结构检查通过。" + payload["notice"])
    else:
        print(config.name)
        print(f"阶段：项目骨架 {__version__}")
        print(f"候选数据：{config.dataset}；暂定设备：{', '.join(config.target_appliances)}")
        print(payload["notice"])
    return 0

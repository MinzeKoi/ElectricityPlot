"""ElectricityPlot configuration, real-data preparation and protocol commands."""

import argparse
import json
from pathlib import Path
import sys
from urllib.error import URLError

from . import __version__
from .config import ConfigError, load_config
from .data.common import DataError, read_json, safe_child, sha256, write_json


def _required(value, name):
    if value is None:
        raise DataError(f"{name} is required")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ElectricityPlot：数据、质量审计与实验协议")
    parser.add_argument("command", choices=("status", "validate-config", "env-report", "download-refit", "prepare-refit", "prepare-aggregate", "audit-data", "freeze-protocol", "validate-confirmations", "train-baseline", "predict", "evaluate"), nargs="?", default="status")
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--json", action="store_true", help="输出结构化状态")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path, help="新产物目录；不会覆盖已有数据集或协议")
    parser.add_argument("--households", nargs="+", default=["2", "3", "4", "5", "11"])
    parser.add_argument("--household-id")
    parser.add_argument("--unit", help="用户导入必须显式声明 W")
    parser.add_argument("--sample-bytes", type=int, default=2097152, help="每户下载字节数；0 为完整下载并校验出版方哈希")
    parser.add_argument("--catalog", type=Path, default=Path("configs/refit_channels.json"))
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan", type=Path, default=Path("configs/m1_split.json"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/m2_baseline.json"))
    parser.add_argument("--model-kind", choices=("small_seq2point", "constant_mean", "zero"), default="small_seq2point")
    parser.add_argument("--appliances", nargs="+")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--partition")
    parser.add_argument("--prediction", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as error:
        if args.json:
            print(json.dumps({"status": "invalid_config", "error": str(error)}, ensure_ascii=False))
        else:
            print(f"配置检查失败：{error}", file=sys.stderr)
        return 2

    if args.command not in ("status", "validate-config"):
        try:
            if args.command == "train-baseline":
                from .models.training import train
                result = train(_required(args.manifest, "--manifest"), args.training_config, _required(args.output, "--output"), args.model_kind, args.appliances)
            elif args.command == "predict":
                from .models.prediction import predict
                result = predict(_required(args.model, "--model"), _required(args.output, "--output"), args.manifest, args.partition, args.input)
            elif args.command == "evaluate":
                from .evaluation.runner import evaluate
                result = evaluate(_required(args.prediction, "--prediction"), _required(args.manifest, "--manifest"), _required(args.output, "--output"))
            elif args.command == "env-report":
                from .environment import environment_report
                result = environment_report()
                if args.output:
                    if args.output.exists():
                        raise DataError("Environment report already exists; choose a new path")
                    write_json(args.output, result)
            elif args.command == "download-refit":
                from .data.acquire import fetch_refit
                result = fetch_refit(_required(args.output, "--output"), args.households, args.sample_bytes)
            elif args.command in ("prepare-refit", "prepare-aggregate"):
                from .data.prepare import prepare
                if args.command == "prepare-refit":
                    directory = _required(args.input_dir, "--input-dir")
                    if len(set(args.households)) != len(args.households):
                        raise DataError("Duplicate household IDs")
                    inputs = {house: directory / f"CLEAN_House{house}.csv" for house in args.households}
                    kind = "refit"
                else:
                    if args.unit != "W":
                        raise DataError("Only explicitly declared active power in W is supported")
                    inputs = {_required(args.household_id, "--household-id"): _required(args.input, "--input")}
                    kind = "aggregate"
                result = prepare(inputs, _required(args.output, "--output"), args.catalog, config.sample_interval_seconds, kind)
            elif args.command == "audit-data":
                manifest_path = _required(args.manifest, "--manifest")
                result = read_json(manifest_path)
                for house in result["households"].values():
                    for artifact in house["artifacts"].values():
                        if sha256(safe_child(manifest_path.parent, artifact["path"])) != artifact["sha256"]:
                            raise DataError("Prepared artifact hash mismatch")
                result = {"status": "audit_verified", "audit": result}
            elif args.command == "freeze-protocol":
                from .data.protocol import freeze
                result = freeze(_required(args.prepared, "--prepared"), args.plan, _required(args.output, "--output"), config)
            else:
                from .data.protocol import validate_confirmation_history
                records = json.loads(_required(args.input, "--input").read_text(encoding="utf-8"))
                if not isinstance(records, list):
                    raise DataError("Confirmation history must be a JSON array")
                validate_confirmation_history(records, read_json(_required(args.manifest, "--manifest")))
                result = {"status": "confirmations_valid", "records": len(records)}
        except (DataError, OSError, URLError, ValueError, KeyError, TypeError, OverflowError, ImportError, RuntimeError) as error:
            if isinstance(error, ImportError):
                error = DataError("M2 dependency missing; install requirements-m2-cpu.lock in the project environment. " + str(error))
            if args.json:
                print(json.dumps({"status": "data_operation_failed", "error": str(error)}, ensure_ascii=False))
            else:
                print(f"数据操作失败：{error}", file=sys.stderr)
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    payload = {
        "status": "configuration_valid",
        "project": config.name,
        "version": __version__,
        "stage": "m2_baseline",
        "dataset_candidate": config.dataset,
        "target_appliances": list(config.target_appliances),
        "query_budgets": list(config.query_budgets),
        "data_validation": "not_run",
        "training": "available_not_run_by_status",
        "experiments": "not_run",
        "notice": "本状态命令仅检查配置，不读取运行成绩。M2 可使用 train-baseline、predict、evaluate；状态不代表已有模型或有效预测。",
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "validate-config":
        print("配置结构检查通过。" + payload["notice"])
    else:
        print(config.name)
        print(f"阶段：M2 基线闭环 {__version__}")
        print(f"候选数据：{config.dataset}；暂定设备：{', '.join(config.target_appliances)}")
        print(payload["notice"])
    return 0

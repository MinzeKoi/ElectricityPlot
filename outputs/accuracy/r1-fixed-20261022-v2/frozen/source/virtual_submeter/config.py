"""Validate the initial experiment scope without reading data or model weights."""

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import tomllib


class ConfigError(ValueError):
    """The configuration is invalid or claims an unsupported scope."""


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    target_appliances: tuple[str, ...]
    dataset: str
    sample_interval_seconds: int
    raw_dir: str
    processed_dir: str
    query_budgets: tuple[int, ...]
    random_seeds: tuple[int, ...]
    query_window_seconds: int
    split_gap_seconds: int


def _keys(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ConfigError(f"{name} 字段不完整或含未知字段。")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{name} 必须为正整数。")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} 必须为非空字符串。")
    return value


def _relative_dir(value: object, name: str) -> str:
    value = _text(value, name)
    windows = PureWindowsPath(value)
    if windows.drive or windows.root or ".." in windows.parts or "\x00" in value:
        raise ConfigError(f"{name} 必须为项目内不含上级跳转的相对目录。")
    return value


def _int_list(value: object, name: str) -> tuple[int, ...]:
    if (not isinstance(value, list) or not value
            or any(type(item) is not int or item < 0 for item in value)
            or len(set(value)) != len(value)):
        raise ConfigError(f"{name} 必须为非空、不重复的非负整数列表。")
    return tuple(value)


def load_config(path: Path) -> ProjectConfig:
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"无法读取配置：{error}") from error

    _keys(raw, {"schema_version", "project", "scope", "data", "protocol"}, "配置")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ConfigError("仅支持 schema_version = 1。")
    project = _keys(raw["project"], {"name"}, "project")
    scope = _keys(raw["scope"], {
        "input_kind", "target_appliances", "supports_pv", "supports_storage"
    }, "scope")
    data = _keys(raw["data"], {
        "dataset", "sample_interval_seconds", "raw_dir", "processed_dir"
    }, "data")
    protocol = _keys(raw["protocol"], {
        "query_budgets", "random_seeds", "query_window_seconds", "split_gap_seconds"
    }, "protocol")

    if (scope["input_kind"] != "aggregate_active_power"
            or scope["supports_pv"] is not False
            or scope["supports_storage"] is not False):
        raise ConfigError("首版仅配置无光伏、无储能的总有功功率场景。")
    targets = scope["target_appliances"]
    if (not isinstance(targets, list) or not targets
            or any(not isinstance(item, str) or not item.strip() for item in targets)
            or len(set(targets)) != len(targets)):
        raise ConfigError("target_appliances 必须为非空、不重复的设备标识列表。")

    interval = _positive_int(data["sample_interval_seconds"], "sample_interval_seconds")
    window = _positive_int(protocol["query_window_seconds"], "query_window_seconds")
    gap = _positive_int(protocol["split_gap_seconds"], "split_gap_seconds")
    if window % interval or gap % interval or gap < window:
        raise ConfigError("确认窗口和隔离带须为采样间隔的整数倍，且隔离带不能短于确认窗口。")
    budgets = _int_list(protocol["query_budgets"], "query_budgets")
    if budgets[0] != 0 or tuple(sorted(budgets)) != budgets:
        raise ConfigError("query_budgets 必须从 0 开始严格递增，以保留不适配对照。")

    return ProjectConfig(
        name=_text(project["name"], "project.name"),
        target_appliances=tuple(targets),
        dataset=_text(data["dataset"], "data.dataset"),
        sample_interval_seconds=interval,
        raw_dir=_relative_dir(data["raw_dir"], "data.raw_dir"),
        processed_dir=_relative_dir(data["processed_dir"], "data.processed_dir"),
        query_budgets=budgets,
        random_seeds=_int_list(protocol["random_seeds"], "random_seeds"),
        query_window_seconds=window,
        split_gap_seconds=gap,
    )

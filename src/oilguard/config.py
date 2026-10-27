from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "experiment.json"


@dataclass(frozen=True)
class EventSpec:
    class_id: int
    name: str
    steady_label: int
    transient_label: int
    target: str
    exog: str
    target_scale: float
    exog_scale: float
    target_unit: str
    exog_unit: str


@dataclass(frozen=True)
class PreprocessingSpec:
    resample_rule: str
    ffill_limit_raw: int
    min_normal_samples: int
    split: tuple[float, float, float]
    min_calibration_samples: int
    min_holdout_samples: int
    max_missing_fraction: float
    max_gap_seconds: float


@dataclass(frozen=True)
class ModelSpec:
    max_p: int
    max_q: int
    max_order: int
    d_max: int
    trend: str


@dataclass(frozen=True)
class WarningSpec:
    quantile: float
    persistence: int
    cooldown_minutes: int
    detection_grace_minutes: int
    pre_event_grace_minutes: int
    threshold_floor: float
    sensitivity_quantiles: tuple[float, ...]
    sensitivity_persistence: tuple[int, ...]


@dataclass(frozen=True)
class EDASpec:
    acf_lags: int
    decomposition_period: int
    max_examples: int


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_root: Path
    project_root: Path
    results_dir: Path
    data_dir: Path
    figures_dir: Path
    metrics_dir: Path
    tables_dir: Path
    logs_dir: Path
    processed_dir: Path
    report_dir: Path
    event: EventSpec
    preprocessing: PreprocessingSpec
    models: ModelSpec
    warning: WarningSpec
    eda: EDASpec
    seed: int
    examples: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def event_id(self) -> str:
        return f"class_{self.event.class_id}"

    def ensure_directories(self) -> None:
        for path in (
            self.results_dir,
            self.figures_dir,
            self.metrics_dir,
            self.tables_dir,
            self.logs_dir,
            self.processed_dir,
            self.processed_dir / "events",
            self.report_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    dataset_root = _resolve_path(raw["dataset_root"], PROJECT_ROOT)
    results_dir = PROJECT_ROOT / "results"
    data_dir = PROJECT_ROOT / "data"
    event = EventSpec(**raw["primary_event"])
    preprocessing_raw = dict(raw["preprocessing"])
    preprocessing_raw["split"] = tuple(float(v) for v in preprocessing_raw["split"])
    preprocessing = PreprocessingSpec(**preprocessing_raw)
    models = ModelSpec(**raw["models"])
    warning_raw = dict(raw["warning"])
    warning_raw["sensitivity_quantiles"] = tuple(
        float(v) for v in warning_raw["sensitivity_quantiles"]
    )
    warning_raw["sensitivity_persistence"] = tuple(
        int(v) for v in warning_raw["sensitivity_persistence"]
    )
    warning = WarningSpec(**warning_raw)
    eda = EDASpec(**raw["eda"])

    if abs(sum(preprocessing.split) - 1.0) > 1e-6:
        raise ValueError("preprocessing.split must sum to 1.0")
    if any(v <= 0 for v in preprocessing.split):
        raise ValueError("preprocessing.split values must be positive")
    if not dataset_root.exists():
        if data_dir.exists():
            dataset_root = data_dir
        else:
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    config = ExperimentConfig(
        dataset_root=dataset_root,
        project_root=PROJECT_ROOT,
        results_dir=results_dir,
        data_dir=data_dir,
        figures_dir=results_dir / "figures",
        metrics_dir=results_dir / "metrics",
        tables_dir=results_dir / "tables",
        logs_dir=results_dir / "logs",
        processed_dir=data_dir / "processed",
        report_dir=PROJECT_ROOT / "report",
        event=event,
        preprocessing=preprocessing,
        models=models,
        warning=warning,
        eda=eda,
        seed=int(raw.get("seed", 42)),
        examples=tuple(raw.get("examples", [])),
        raw=raw,
    )
    config.ensure_directories()
    return config


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    packages = [
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pytest",
        "python-docx",
        "python-pptx",
        "nbformat",
        "nbconvert",
        "jupyter",
    ]
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions

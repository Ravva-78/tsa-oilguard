from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import ExperimentConfig


@dataclass(frozen=True)
class SensorQuality:
    column: str
    n_rows: int
    n_present: int
    missing_fraction: float
    n_unique: int
    all_zero: bool
    constant: bool
    absurd: bool
    max_gap_seconds: float | None
    status: str
    usable: bool


@dataclass
class EventDataset:
    event_id: str
    stem: str
    well_id: str | None
    path: Path
    frame: pd.DataFrame
    quality_target: SensorQuality
    quality_exog: SensorQuality
    transient_onset: pd.Timestamp
    steady_onset: pd.Timestamp | None
    transient_pos: int
    steady_pos: int | None
    train_end_pos: int
    calib_start_pos: int
    calib_end_pos: int
    holdout_start_pos: int
    resample_seconds: float
    normal_samples: int
    holdout_samples: int
    transient_samples: int
    steady_samples: int
    normal_hours: float
    holdout_hours: float
    transient_minutes: float
    steady_minutes: float
    total_hours: float
    excluded_reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.excluded_reason is None


def sensor_quality(
    series: pd.Series,
    column: str,
    max_missing_fraction: float,
    max_gap_seconds: float,
) -> SensorQuality:
    n_rows = len(series)
    present = series.notna()
    n_present = int(present.sum())
    missing_fraction = float(1.0 - (n_present / n_rows)) if n_rows else 1.0

    if n_present == 0:
        return SensorQuality(
            column=column,
            n_rows=n_rows,
            n_present=0,
            missing_fraction=1.0,
            n_unique=0,
            all_zero=False,
            constant=False,
            absurd=False,
            max_gap_seconds=None,
            status="absent",
            usable=False,
        )

    values = series.dropna()
    n_unique = int(values.nunique())
    all_zero = bool(np.allclose(values.to_numpy(dtype=float), 0.0, equal_nan=False))
    constant = bool(n_unique <= 1)
    absurd = bool(np.isinf(values.to_numpy(dtype=float)).any()) or bool(
        (np.abs(values.to_numpy(dtype=float)) > 1e12).any()
    )

    present_index = series.index[present]
    if len(present_index) > 1:
        deltas = np.diff(present_index.to_numpy()).astype("timedelta64[s]").astype(float)
        max_gap = float(np.nanmax(deltas))
    else:
        max_gap = None

    reasons: list[str] = []
    if all_zero:
        reasons.append("all-zero")
    if constant and not all_zero:
        reasons.append("constant")
    if absurd:
        reasons.append("absurd")
    if missing_fraction > max_missing_fraction:
        reasons.append("gappy")
    if max_gap is not None and max_gap > max_gap_seconds:
        reasons.append("long-gap")

    return SensorQuality(
        column=column,
        n_rows=n_rows,
        n_present=n_present,
        missing_fraction=round(missing_fraction, 6),
        n_unique=n_unique,
        all_zero=all_zero,
        constant=constant,
        absurd=absurd,
        max_gap_seconds=None if max_gap is None else round(max_gap, 3),
        status="ok" if not reasons else ",".join(reasons),
        usable=not reasons,
    )


def read_labeled_frame(
    path: str | Path,
    target: str,
    exog: str,
    target_scale: float,
    exog_scale: float,
    max_missing_fraction: float,
    max_gap_seconds: float,
) -> tuple[pd.DataFrame, SensorQuality, SensorQuality]:
    parquet_file = pq.ParquetFile(Path(path))
    columns = set(parquet_file.schema_arrow.names)
    missing = {name for name in (target, exog, "class") if name not in columns}
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    read_columns = [target, exog, "class"]
    if "timestamp" in columns:
        read_columns.append("timestamp")
    table = pq.read_table(path, columns=read_columns)
    frame = table.to_pandas()
    if "timestamp" in frame.columns:
        frame = frame.set_index("timestamp")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"Timestamp index is not datetime for {path}")
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[frame.index.notna()]
    frame = frame[frame["class"].notna()]
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    if frame.empty:
        raise ValueError(f"No labeled rows in {path}")

    quality_target = sensor_quality(
        pd.to_numeric(frame[target], errors="coerce"),
        target,
        max_missing_fraction,
        max_gap_seconds,
    )
    quality_exog = sensor_quality(
        pd.to_numeric(frame[exog], errors="coerce"),
        exog,
        max_missing_fraction,
        max_gap_seconds,
    )

    labeled = pd.DataFrame(
        {
            "target": pd.to_numeric(frame[target], errors="coerce") * target_scale,
            "exog": pd.to_numeric(frame[exog], errors="coerce") * exog_scale,
            "class": frame["class"].astype("int16"),
        },
        index=frame.index,
    )
    return labeled, quality_target, quality_exog


def find_onsets(
    labeled: pd.DataFrame,
    transient_label: int,
    steady_label: int,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None]:
    labels = labeled["class"]
    abnormal = labels != 0
    if not abnormal.any():
        raise ValueError("No abnormal labels found")
    first_abnormal = labels.index[abnormal][0]

    transient = labels[labels == transient_label]
    transient_onset = transient.index[0] if len(transient) else first_abnormal
    steady = labels[labels == steady_label]
    steady_onset = steady.index[0] if len(steady) else None
    return first_abnormal, transient_onset, steady_onset


def resample_frame(
    labeled: pd.DataFrame,
    rule: str,
    ffill_limit_raw: int,
) -> pd.DataFrame:
    frame = labeled.copy()
    frame[["target", "exog"]] = frame[["target", "exog"]].ffill(limit=ffill_limit_raw)
    resampled = frame.resample(rule).agg(
        {"target": "mean", "exog": "mean", "class": "last"}
    )
    resampled["class"] = resampled["class"].ffill(limit=1)
    resampled[["target", "exog"]] = resampled[["target", "exog"]].ffill(limit=1)
    resampled = resampled.dropna(subset=["target", "exog", "class"])
    if resampled.empty:
        raise ValueError("No rows remain after resampling")
    resampled["class"] = resampled["class"].astype("int16")
    resampled["exog_lag1"] = resampled["exog"].shift(1)
    resampled = resampled.dropna(subset=["exog_lag1"])
    return resampled


def _position(index: pd.DatetimeIndex, timestamp: pd.Timestamp) -> int:
    return int(index.searchsorted(timestamp, side="left"))


def build_event_dataset(
    path: str | Path,
    config: ExperimentConfig,
    event_id: str | None = None,
    stem: str | None = None,
    well_id: str | None = None,
) -> EventDataset:
    spec = config.event
    prep = config.preprocessing
    path = Path(path)
    stem = stem or path.stem
    event_id = event_id or f"{config.event_id}/{stem}"

    labeled, quality_target, quality_exog = read_labeled_frame(
        path=path,
        target=spec.target,
        exog=spec.exog,
        target_scale=spec.target_scale,
        exog_scale=spec.exog_scale,
        max_missing_fraction=prep.max_missing_fraction,
        max_gap_seconds=prep.max_gap_seconds,
    )
    _, transient_onset, steady_onset = find_onsets(
        labeled,
        transient_label=spec.transient_label,
        steady_label=spec.steady_label,
    )
    if quality_target.status == "absent" or quality_exog.status == "absent":
        raise ValueError(f"target {quality_target.status}, exog {quality_exog.status}")
    frame = resample_frame(labeled, prep.resample_rule, prep.ffill_limit_raw)
    resample_seconds = float(pd.Timedelta(prep.resample_rule).total_seconds())

    transient_pos = _position(frame.index, transient_onset)
    if transient_pos >= len(frame):
        raise ValueError(f"Transient onset outside resampled frame for {stem}")
    steady_pos = None
    if steady_onset is not None:
        candidate = _position(frame.index, steady_onset)
        if candidate < len(frame):
            steady_pos = candidate

    normal_mask = (frame.index < transient_onset) & (frame["class"] == 0)
    normal = frame.loc[normal_mask]
    if normal.empty:
        raise ValueError(f"No normal samples before the event for {stem}")

    n_normal = len(normal)
    n_train = int(n_normal * prep.split[0])
    n_calib = int(n_normal * prep.split[1])
    n_holdout = n_normal - n_train - n_calib
    n_calib = max(n_calib, prep.min_calibration_samples)
    n_holdout = max(n_holdout, prep.min_holdout_samples)
    n_train = n_normal - n_calib - n_holdout

    excluded_reason: str | None = None
    if not quality_target.usable:
        excluded_reason = f"target {quality_target.status}"
    elif not quality_exog.usable:
        excluded_reason = f"exog {quality_exog.status}"
    elif n_train < 10:
        excluded_reason = "insufficient-normal-samples"

    if n_train < 1:
        n_train = 1
        n_calib = max(1, n_normal - 1)
        n_holdout = 0

    train = normal.iloc[:n_train]
    calib = normal.iloc[n_train : n_train + n_calib]
    holdout = normal.iloc[n_train + n_calib :]
    if len(calib) < 1 or len(holdout) < 1:
        excluded_reason = excluded_reason or "insufficient-split-samples"

    train_target = train["target"].dropna().to_numpy(dtype=float)
    train_exog = train["exog"].dropna().to_numpy(dtype=float)
    if len(train_target) < 2 or np.nanstd(train_target) <= 1e-12:
        excluded_reason = excluded_reason or "constant-normal-target"
    if len(train_exog) < 2 or np.nanstd(train_exog) <= 1e-12:
        excluded_reason = excluded_reason or "constant-normal-exog"

    train_end_pos = int(frame.index.get_loc(train.index[-1])) + 1
    calib_start_pos = int(frame.index.get_loc(calib.index[0]))
    calib_end_pos = int(frame.index.get_loc(calib.index[-1])) + 1
    holdout_start_pos = int(frame.index.get_loc(holdout.index[0]))

    dt = pd.Timedelta(prep.resample_rule)
    transient_end = steady_onset if steady_onset is not None else frame.index[-1] + dt
    transient_samples = int(((frame.index >= transient_onset) & (frame.index < transient_end)).sum())
    steady_samples = (
        int((frame.index >= steady_onset).sum()) if steady_onset is not None else 0
    )
    normal_hours = float(len(normal) * resample_seconds / 3600.0)
    holdout_hours = float(len(holdout) * resample_seconds / 3600.0)
    transient_minutes = float(transient_samples * resample_seconds / 60.0)
    steady_minutes = float(steady_samples * resample_seconds / 60.0)
    total_hours = float(len(frame) * resample_seconds / 3600.0)

    return EventDataset(
        event_id=event_id,
        stem=stem,
        well_id=well_id,
        path=path,
        frame=frame,
        quality_target=quality_target,
        quality_exog=quality_exog,
        transient_onset=transient_onset,
        steady_onset=steady_onset,
        transient_pos=transient_pos,
        steady_pos=steady_pos,
        train_end_pos=train_end_pos,
        calib_start_pos=calib_start_pos,
        calib_end_pos=calib_end_pos,
        holdout_start_pos=holdout_start_pos,
        resample_seconds=resample_seconds,
        normal_samples=len(normal),
        holdout_samples=len(holdout),
        transient_samples=transient_samples,
        steady_samples=steady_samples,
        normal_hours=normal_hours,
        holdout_hours=holdout_hours,
        transient_minutes=transient_minutes,
        steady_minutes=steady_minutes,
        total_hours=total_hours,
        excluded_reason=excluded_reason,
    )


def build_candidate_table(
    config: ExperimentConfig,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    spec = config.event
    subset = manifest[
        (manifest["class_dir"] == spec.class_id) & (manifest["source"] == "real")
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        path = Path(row["file_path"])
        if not bool(row["has_target"]) or not bool(row["has_exog"]):
            rows.append(
                {
                    "event_id": f"{config.event_id}/{row['stem']}",
                    "stem": row["stem"],
                    "well_id": row["well_id"],
                    "path": str(path),
                    "target_status": "missing-column",
                    "exog_status": "missing-column",
                    "target_usable": False,
                    "exog_usable": False,
                    "gold": False,
                    "exclusion_reason": "missing-column",
                }
            )
            continue
        try:
            event = build_event_dataset(
                path=path,
                config=config,
                event_id=f"{config.event_id}/{row['stem']}",
                stem=row["stem"],
                well_id=row["well_id"],
            )
        except Exception as exc:
            rows.append(
                {
                    "event_id": f"{config.event_id}/{row['stem']}",
                    "stem": row["stem"],
                    "well_id": row["well_id"],
                    "path": str(path),
                    "target_status": "error",
                    "exog_status": "error",
                    "target_usable": False,
                    "exog_usable": False,
                    "gold": False,
                    "exclusion_reason": f"load-error: {exc}",
                }
            )
            continue

        train_frame = event.frame.iloc[: event.train_end_pos]
        rows.append(
            {
                "event_id": event.event_id,
                "stem": event.stem,
                "well_id": event.well_id,
                "path": str(event.path),
                "target_status": event.quality_target.status,
                "target_missing_fraction": event.quality_target.missing_fraction,
                "target_max_gap_seconds": event.quality_target.max_gap_seconds,
                "target_usable": event.quality_target.usable,
                "exog_status": event.quality_exog.status,
                "exog_missing_fraction": event.quality_exog.missing_fraction,
                "exog_max_gap_seconds": event.quality_exog.max_gap_seconds,
                "exog_usable": event.quality_exog.usable,
                "train_target_std": round(float(train_frame["target"].std()), 8),
                "train_target_unique": int(train_frame["target"].nunique()),
                "train_exog_std": round(float(train_frame["exog"].std()), 8),
                "train_exog_unique": int(train_frame["exog"].nunique()),
                "normal_samples": event.normal_samples,
                "holdout_samples": event.holdout_samples,
                "normal_hours": round(event.normal_hours, 4),
                "holdout_hours": round(event.holdout_hours, 4),
                "transient_minutes": round(event.transient_minutes, 2),
                "steady_minutes": round(event.steady_minutes, 2),
                "total_hours": round(event.total_hours, 4),
                "has_steady": event.steady_onset is not None,
                "gold": event.usable,
                "exclusion_reason": event.excluded_reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["well_id", "stem"]).reset_index(drop=True)


def save_event_frame(event: EventDataset, path: str | Path) -> None:
    frame = event.frame.copy()
    frame.index.name = "timestamp"
    frame["split"] = "train"
    frame.iloc[event.calib_start_pos : event.calib_end_pos, frame.columns.get_loc("split")] = "calibration"
    frame.iloc[
        event.holdout_start_pos : event.transient_pos, frame.columns.get_loc("split")
    ] = "holdout"
    if event.steady_pos is not None:
        frame.iloc[
            event.transient_pos : event.steady_pos, frame.columns.get_loc("split")
        ] = "transient"
        frame.iloc[event.steady_pos :, frame.columns.get_loc("split")] = "steady"
    else:
        frame.iloc[event.transient_pos :, frame.columns.get_loc("split")] = "transient"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)

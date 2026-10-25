from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

CLASS_DIR_RE = re.compile(r"^\d+$")
WELL_RE = re.compile(r"^(WELL-\d+)_(\d{14})$")
SIMULATED_RE = re.compile(r"^SIMULATED_(\d+)$")
DRAWN_RE = re.compile(r"^DRAWN_(\d+)$")


def parse_source(stem: str) -> str:
    if stem.startswith("WELL-"):
        return "real"
    if stem.startswith("SIMULATED_"):
        return "simulated"
    if stem.startswith("DRAWN_"):
        return "drawn"
    return "unknown"


def parse_well_id(stem: str) -> str | None:
    match = WELL_RE.match(stem)
    return match.group(1) if match else None


def parse_instance_time(stem: str) -> datetime | None:
    match = WELL_RE.match(stem)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def build_manifest(
    dataset_root: str | Path,
    target: str = "P-TPT",
    exog: str = "T-TPT",
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    class_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and CLASS_DIR_RE.match(p.name)],
        key=lambda p: int(p.name),
    )
    rows: list[dict[str, Any]] = []
    for class_dir in class_dirs:
        files = sorted(class_dir.glob("*.parquet"))
        for file_path in files:
            parquet_file = pq.ParquetFile(file_path)
            columns = list(parquet_file.schema_arrow.names)
            stem = file_path.stem
            source = parse_source(stem)
            well_id = parse_well_id(stem)
            instance_time = parse_instance_time(stem)
            rows.append(
                {
                    "class_dir": int(class_dir.name),
                    "source": source,
                    "well_id": well_id,
                    "instance_time": instance_time,
                    "stem": stem,
                    "file_name": file_path.name,
                    "relative_path": str(file_path.relative_to(root)),
                    "file_path": str(file_path),
                    "rows": int(parquet_file.metadata.num_rows),
                    "row_groups": int(parquet_file.metadata.num_row_groups),
                    "size_mb": round(file_path.stat().st_size / 1_000_000, 3),
                    "columns": columns,
                    "has_target": target in columns,
                    "has_exog": exog in columns,
                }
            )
            if verbose and len(rows) % 250 == 0:
                print(f"manifest: {len(rows)} files indexed", flush=True)

    manifest = pd.DataFrame(rows)
    manifest["class_dir"] = manifest["class_dir"].astype("int16")
    manifest["rows"] = manifest["rows"].astype("int64")
    manifest["row_groups"] = manifest["row_groups"].astype("int32")
    manifest["has_target"] = manifest["has_target"].astype(bool)
    manifest["has_exog"] = manifest["has_exog"].astype(bool)
    manifest["instance_time"] = pd.to_datetime(manifest["instance_time"])
    manifest = manifest.sort_values(
        ["class_dir", "source", "instance_time", "file_name"],
        na_position="last",
    ).reset_index(drop=True)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_parquet(output, index=False)
    if summary_path is not None:
        summary = (
            manifest.groupby(["class_dir", "source"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        summary.loc["total"] = summary.sum()
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path)
    return manifest


def load_or_build_manifest(
    dataset_root: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    target: str = "P-TPT",
    exog: str = "T-TPT",
    refresh: bool = False,
) -> pd.DataFrame:
    output = Path(output_path)
    if output.exists() and not refresh:
        return pd.read_parquet(output)
    return build_manifest(
        dataset_root=dataset_root,
        target=target,
        exog=exog,
        output_path=output,
        summary_path=summary_path,
    )

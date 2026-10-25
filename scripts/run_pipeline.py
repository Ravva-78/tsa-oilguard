from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oilguard.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OILGUARD forecasting and warning pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment JSON")
    parser.add_argument("--refresh-manifest", action="store_true", help="Rebuild the file manifest")
    parser.add_argument("--skip-eda", action="store_true", help="Skip exploratory figures")
    args = parser.parse_args()
    summary = run_pipeline(
        config_path=args.config,
        refresh_manifest=args.refresh_manifest,
        skip_eda=args.skip_eda,
    )
    print("Pipeline finished")
    print(f"Modeled events: {summary['n_modeled_events']}")
    print(f"Runtime: {summary['runtime_seconds']:.1f} s")
    print(f"Results: {PROJECT_ROOT / 'results'}")


if __name__ == "__main__":
    main()

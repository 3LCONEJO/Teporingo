"""Command-line entry point for Teporingo-Demultiplexing."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    default_config = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    parser = argparse.ArgumentParser(
        prog="teporingo-demultiplexing",
        description="Run the Teporingo Demultiplexing pipeline.",
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to the YAML config file",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary = run_pipeline(args.config)
    pipeline = summary["pipeline"]

    print("Teporingo-Demultiplexing scaffold is ready.")
    print(f"Config: {summary['config_path']}")
    print(f"min_maf: {pipeline['min_maf']}")
    print(f"max_maf: {pipeline['max_maf']}")
    print(f"use_gt: {pipeline['use_gt']}")
    print(f"vcf: {pipeline['vcf']}")
    print(f"assignments: {pipeline['assignments']}")
    print(f"bams: {len(pipeline['bams'])} configured")
    print(f"vcf samples: {pipeline['vcf_metadata']['sample_count']}")
    print("batch plan:")
    for batch in pipeline["batch_plan"]:
        print(f"  {batch['batch_label']}: {batch['bam_path']}")

    tips = pipeline.get("tips", {})
    if tips:
        print("tips:")
        for key, value in tips.items():
            print(f"  {key}: {value}")

"""Command-line entry point for Teporingo-Demultiplexing."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teporingo-demultiplexing",
        description="Run the Teporingo Demultiplexing pipeline.",
    )
    parser.add_argument("--help-config", action="store_true", help="Show configuration guidance")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.help_config:
        print("Add your default pipeline configuration under configs/.")
        return

    print("Teporingo-Demultiplexing scaffold is ready.")

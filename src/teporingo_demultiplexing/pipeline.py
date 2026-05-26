"""Core pipeline orchestration for Teporingo-Demultiplexing."""

from __future__ import annotations

import gzip
import logging
import pickle
from pathlib import Path

import scipy.sparse as sp
import torch

from .config import load_simple_yaml
from buildnload import construir_matriz_bam, construir_matriz_vcf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run_pipeline(config_path: str | Path) -> dict:
    """Load the YAML config, validate inputs, and prepare per-batch outputs."""

    config = load_simple_yaml(config_path)
    pipeline_config = config.get("pipeline")

    if not isinstance(pipeline_config, dict):
        raise ValueError("Missing 'pipeline' section in config")

    required_keys = ("min_maf", "max_maf", "use_gt", "vcf", "assignments", "bams")
    missing_keys = [key for key in required_keys if key not in pipeline_config]
    if missing_keys:
        raise ValueError(f"Missing pipeline keys: {', '.join(missing_keys)}")

    bams = pipeline_config["bams"]
    if not isinstance(bams, dict) or not bams:
        raise ValueError("'bams' must be a non-empty mapping of batch labels to BAM paths")

    validated_bams = validate_bams(bams)
    vcf_path = validate_file_path(pipeline_config["vcf"], "VCF")
    assignments_path = validate_file_path(pipeline_config["assignments"], "assignments file")
    vcf_metadata = load_vcf_metadata(vcf_path)
    batch_plan = build_batch_plan(validated_bams)
    pair_mode = normalize_pair_mode(pipeline_config.get("pair_mode", "sampled"))
    training_config = config.get("training", config.get("model_training", {}))
    if training_config and not isinstance(training_config, dict):
        raise ValueError("'training' must be a mapping when provided")

    tips = pipeline_config.get("tips", {})
    if tips and not isinstance(tips, dict):
        raise ValueError("'tips' must be a mapping when provided, make sure you read them! :D")

    output_prefix = Path(pipeline_config.get("output_prefix", Path(config_path).parent / "output"))
    if output_prefix.exists() and not output_prefix.is_dir():
        raise ValueError(f"Output prefix path exists and is not a directory: {output_prefix}")
    if not output_prefix.exists():
        logging.info(f":o The output directory does not exist yet: {output_prefix}")
        output_prefix.mkdir(parents=True)
        logging.info(f":D Created output directory: {output_prefix}")
    logging.info(f":D Output prefix will be: {output_prefix}")

    summary = {
        "config_path": str(Path(config_path)),
        "pipeline": {
            "min_maf": pipeline_config["min_maf"],
            "max_maf": pipeline_config["max_maf"],
            "use_gt": pipeline_config["use_gt"],
            "vcf": str(vcf_path),
            "assignments": str(assignments_path),
            "bams": validated_bams,
            "batch_plan": batch_plan,
            "vcf_metadata": vcf_metadata,
            "negative_ratio": pipeline_config.get("negative_ratio", 1.0),
            "pair_mode": pair_mode,
            "hard_negatives_k": pipeline_config.get("hard_negatives_k", 5),
            "tips": tips,
            "output_prefix": str(output_prefix),
        },
        "training": training_config,
    }

    logging.info("\n[1/2] Building genotype matrix (X_VCF)...")
    X_VCF, donors, snps_info = construir_matriz_vcf(
        str(vcf_path),
        donantes_seleccionados=None,
        usar_ds=not bool(pipeline_config["use_gt"]),
        min_maf=pipeline_config["min_maf"],
        max_maf=pipeline_config["max_maf"],
    )

    summary["pipeline"]["vcf_matrix_shape"] = tuple(X_VCF.shape)
    summary["pipeline"]["donor_count"] = len(donors)
    summary["pipeline"]["snp_count"] = len(snps_info)

    prepared_batches = []
    for batch in batch_plan:
        batch_label = batch["batch_label"]
        batch_bam_path = batch["bam_path"]
        logging.info(f"  Processing batch '{batch_label}' with BAM: {batch_bam_path}")

        vcf_matrix_path = Path(f"{output_prefix}_vcf_{batch_label}_matrix.pt")
        bam_matrix_path = Path(f"{output_prefix}_bam_{batch_label}_matrix.npz")
        barcodes_path = Path(f"{output_prefix}_barcodes_{batch_label}.pkl")
        metadata_path = Path(f"{output_prefix}_metadata_{batch_label}.pt")

        output_paths = (vcf_matrix_path, bam_matrix_path, barcodes_path, metadata_path)
        if all(path.exists() for path in output_paths):
            logging.info(f"    Reusing existing outputs for batch '{batch_label}'")
            prepared_batches.append(
                {
                    "batch_label": batch_label,
                    "bam_path": batch_bam_path,
                    "vcf_matrix": str(vcf_matrix_path),
                    "bam_matrix": str(bam_matrix_path),
                    "barcodes": str(barcodes_path),
                    "metadata": str(metadata_path),
                    "reused": True,
                }
            )
            continue

        X_BAM_sparse, barcodes, _ = construir_matriz_bam(
            batch_bam_path,
            snps_info,
        )

        if not vcf_matrix_path.exists():
            torch.save(
                {
                    "X_VCF": X_VCF,
                    "donors": donors,
                    "snps": snps_info,
                },
                vcf_matrix_path,
            )

        if not bam_matrix_path.exists():
            sp.save_npz(bam_matrix_path, X_BAM_sparse)

        if not barcodes_path.exists():
            with barcodes_path.open("wb") as handle:
                pickle.dump(barcodes, handle)

        if not metadata_path.exists():
            torch.save(
                {
                    "batch_label": batch_label,
                    "bam_path": str(batch_bam_path),
                    "vcf_path": str(vcf_path),
                    "assignments_path": str(assignments_path),
                    "vcf_metadata": vcf_metadata,
                    "sample_count": vcf_metadata["sample_count"],
                    "snps": snps_info,
                    "barcodes": barcodes,
                },
                metadata_path,
            )

        prepared_batches.append(
            {
                "batch_label": batch_label,
                "bam_path": batch_bam_path,
                "vcf_matrix": str(vcf_matrix_path),
                "bam_matrix": str(bam_matrix_path),
                "barcodes": str(barcodes_path),
                "metadata": str(metadata_path),
                "reused": False,
            }
        )

    summary["pipeline"]["prepared_batches"] = prepared_batches
    logging.info("\n[3/3] Launching Siamese training from prepared outputs...")

    from train_siamese import build_evaluate_config, build_training_config, train_from_pipeline

    resolved_training_config = build_training_config(config)
    resolved_evaluate_config = build_evaluate_config(config, summary, resolved_training_config)
    summary["training"] = resolved_training_config
    summary["evaluate"] = resolved_evaluate_config
    train_from_pipeline(summary, resolved_training_config, resolved_evaluate_config)

    logging.info("\nPipeline preparation finished successfully")
    return summary


def validate_bams(bam_config):
    """Validate BAM file paths from the config."""

    validated_bams = {}
    for batch_label, bam_path in bam_config.items():
        bam_file = validate_file_path(bam_path, f"BAM for batch '{batch_label}'")
        validated_bams[batch_label] = str(bam_file)

    logging.info(":D Validated BAM files for batches: %s", ", ".join(validated_bams.keys()))
    return validated_bams


def build_batch_plan(validated_bams):
    """Build an ordered batch plan from validated BAM paths."""

    return [
        {"batch_label": batch_label, "bam_path": bam_path}
        for batch_label, bam_path in validated_bams.items()
    ]


def load_vcf_metadata(vcf_path):
    """Read the VCF header and extract sample metadata without extra dependencies."""

    vcf_path = Path(vcf_path)
    if vcf_path.suffix == ".gz":
        opener = gzip.open
        open_kwargs = {"mode": "rt", "encoding": "utf-8"}
    else:
        opener = open
        open_kwargs = {"mode": "rt", "encoding": "utf-8"}

    sample_names = []
    header_line = None

    with opener(vcf_path, **open_kwargs) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                header_line = line.rstrip("\n")
                columns = header_line.split("\t")
                sample_names = columns[9:]
                break

    if header_line is None:
        raise ValueError(f"VCF header not found in {vcf_path}")

    metadata = {
        "vcf_path": str(vcf_path.resolve()),
        "sample_count": len(sample_names),
        "sample_names": sample_names,
    }

    logging.info(":D Loaded VCF header with %s samples", metadata["sample_count"])
    return metadata


def validate_file_path(path_value, label):
    """Validate a file path and return its resolved absolute path."""

    if not path_value:
        raise ValueError(f":D Missing {label} path")

    file_path = Path(path_value)
    if not file_path.is_file():
        raise FileNotFoundError(f":D {label} not found: {file_path}")

    resolved_path = file_path.resolve()
    logging.info(":D Found and validated %s: %s", label, resolved_path)
    return resolved_path


def normalize_pair_mode(value):
    """Normalize pair_mode to one of the supported dataset modes."""

    if isinstance(value, str) and value in {"sampled", "exhaustive"}:
        return value

    logging.warning("Invalid pair_mode %r in config, defaulting to 'sampled'", value)
    return "sampled"

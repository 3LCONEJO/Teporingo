#!/usr/bin/env python3
"""Evaluate a trained Siamese network on a prepared batch or dataset."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from buildnload import GenotypeMatchingDataset
from teporingo_demultiplexing.models.siamese_network import (
    SiameseNetwork,
    ShallowSiameseNetwork,
)


def load_assignments_table(path):
    """Load a barcode-to-donor mapping from a TSV or CSV file."""

    assignments = {}
    path = Path(path)

    with path.open(newline='') as handle:
        first_line = handle.readline().strip()
        handle.seek(0)

        if first_line and first_line.split('\t')[0].upper() == 'BARCODE':
            reader = csv.DictReader(handle, delimiter='\t')
            for row in reader:
                barcode = (row.get('BARCODE') or '').strip()
                donor = (row.get('DONOR') or row.get('BEST') or '').strip()
                if barcode and donor:
                    assignments[barcode] = donor
        else:
            for line in handle:
                if not line.strip():
                    continue
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                barcode, donor = parts[0], parts[1]
                assignments[barcode] = donor

    return assignments

def load_prepared_data(vcf_path, bam_path, metadata_path):
    """Load matrices already prepared by the pipeline."""

    vcf_data = torch.load(vcf_path, weights_only=False)
    X_vcf = vcf_data['X_VCF']
    donors = vcf_data['donors']

    X_bam = load_npz(bam_path)

    metadata = torch.load(metadata_path, weights_only=False)
    barcodes = metadata.get('barcodes', [])
    assignments = metadata.get('assignments', {})

    if not assignments:
        assignments_path = metadata.get('assignments_path')
        if assignments_path:
            assignments = load_assignments_table(assignments_path)

    print(f"  X_VCF: {X_vcf.shape}")
    print(f"  X_BAM: {X_bam.shape}")
    print(f"  Donors: {len(donors)}")
    print(f"  Cells: {len(barcodes)}")
    print(f"  Assignments: {len(assignments)}")

    if not assignments:
        raise ValueError(
            'No barcode-to-donor assignments were found. ' \
            'Make sure the pipeline metadata contains assignments or assignments_path.'
        )

    return X_vcf, X_bam, donors, barcodes, assignments


def infer_embedding_dim(state_dict, default=512):
    """Try to infer embedding_dim from the checkpoint."""

    candidates = [
        'cell_encoder.encoder.6.weight',
        'genotype_encoder.encoder.6.weight',
        'cell_encoder.4.weight',
        'genotype_encoder.4.weight',
    ]

    for key in candidates:
        if key in state_dict:
            return state_dict[key].shape[0]

    return default


def infer_model_type(state_dict):
    """Infer which Siamese variant produced the checkpoint."""

    if any(key.startswith('cell_encoder.encoder.') or key.startswith('genotype_encoder.encoder.') for key in state_dict):
        return 'siamese'

    if any(key.startswith('cell_encoder.') or key.startswith('genotype_encoder.') for key in state_dict):
        return 'shallow'

    return 'siamese'


def scores_to_probabilities(raw_scores):
    """Convert model scores to probabilities for binary metrics.

    - If scores are already in [0, 1], keep them as-is.
    - If scores are in [-1, 1], rescale them linearly to [0, 1].
    - Otherwise, fall back to sigmoid.
    """

    if raw_scores.min() >= 0.0 and raw_scores.max() <= 1.0:
        return raw_scores.detach().cpu().numpy().reshape(-1)

    elif raw_scores.min() >= -1.0 and raw_scores.max() <= 1.0:
        return ((raw_scores + 1.0) / 2.0).detach().cpu().numpy().reshape(-1)
    else:
        return torch.sigmoid(raw_scores).detach().cpu().numpy().reshape(-1)


@torch.no_grad()
def evaluate(model, loader, device, decision_threshold=0.9):
    """Run evaluation and return metrics."""

    model.eval()

    all_probs = []
    all_raw_scores = []
    all_labels = []
    all_records = []

    dataset = loader.dataset
    cell_offset = 0

    for batch in loader:
        if len(batch) == 3:
            cell_batch, geno_batch, labels = batch
            batch_cell_idx = None
            batch_donor_idx = None
            batch_true_donor_idx = None
        else:
            cell_batch, geno_batch, labels, batch_cell_idx, batch_donor_idx, batch_true_donor_idx = batch

        cell_batch = cell_batch.to(device)
        geno_batch = geno_batch.to(device)

        raw_scores = model(cell_batch, geno_batch)
        raw_scores_np = raw_scores.detach().cpu().numpy().reshape(-1)
        probs = scores_to_probabilities(raw_scores)
        labels_np = labels.numpy().reshape(-1)

        all_raw_scores.extend(raw_scores_np.tolist())
        all_probs.extend(probs.tolist())
        all_labels.extend(labels_np.tolist())

        if batch_cell_idx is not None:
            batch_cell_idx = batch_cell_idx.numpy().reshape(-1)
            batch_donor_idx = batch_donor_idx.numpy().reshape(-1)
            batch_true_donor_idx = batch_true_donor_idx.numpy().reshape(-1)

            for i, (prob, label) in enumerate(zip(probs, labels_np)):
                cell_idx = int(batch_cell_idx[i])
                donor_idx = int(batch_donor_idx[i])
                true_donor_idx = int(batch_true_donor_idx[i])
                cell_barcode = dataset.barcodes[cell_idx]
                donor_name = dataset.donors[donor_idx]
                true_donor_name = dataset.donors[true_donor_idx]

                all_records.append({
                    'sample_idx': cell_offset + i,
                    'cell_idx': cell_idx,
                    'cell_barcode': cell_barcode,
                    'donor_idx': donor_idx,
                    'donor_name': donor_name,
                    'true_donor_idx': true_donor_idx,
                    'true_donor_name': true_donor_name,
                    'label': float(label),
                    'raw_score': float(raw_scores_np[i]),
                    'score': float(prob),
                    'prediction': float(raw_scores_np[i] >= decision_threshold),
                    'is_correct_pair': int(donor_idx == true_donor_idx),
                })

        cell_offset += len(probs)

    y_true = np.asarray(all_labels, dtype=np.float32)
    y_prob = np.asarray(all_probs, dtype=np.float32)
    y_pred = (np.asarray(all_raw_scores, dtype=np.float32) >= decision_threshold).astype(np.float32)

    if y_true.size == 0:
        raise ValueError('Evaluation produced no pairs. Check assignments and pair_mode.')

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob) if np.unique(y_true).size > 1 else float('nan'),
        'ap': average_precision_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'log_loss': log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)),
        'confidence': float(np.mean(np.maximum(y_prob, 1 - y_prob))),
    }

    p = np.clip(y_prob, 1e-7, 1 - 1e-7)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    metrics['entropy'] = float(np.mean(entropy))
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    metrics['score_table'] = pd.DataFrame(all_records)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate a Siamese model on another batch')
    parser.add_argument('--model', required=True, help='best_model.pt checkpoint')
    parser.add_argument('--vcf-matrix', required=True, help='.pt file containing X_VCF and donors')
    parser.add_argument('--bam-matrix', required=True, help='.npz file containing X_BAM')
    parser.add_argument('--metadata', required=True, help='.pt file containing barcodes and assignments')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default=None, help='cpu or cuda; auto by default')
    parser.add_argument('--embedding-dim', type=int, default=None, help='Optional embedding_dim override')
    parser.add_argument('--output-dir', default=None, help='Directory where evaluation CSV/JSON files will be saved')

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        checkpoint_meta = checkpoint
    else:
        state_dict = checkpoint
        checkpoint_meta = {}

    X_vcf, X_bam, donors, barcodes, assignments = load_prepared_data(
        args.vcf_matrix,
        args.bam_matrix,
        args.metadata,
    )

    n_snps = X_vcf.shape[1]
    network = checkpoint_meta.get('network') or infer_model_type(state_dict)
    embedding_dim = args.embedding_dim or checkpoint_meta.get('embedding_dim') or infer_embedding_dim(state_dict)

    dataset = GenotypeMatchingDataset(
        X_vcf=X_vcf,
        X_bam_sparse=X_bam,
        cell_to_donor=assignments,
        barcodes=barcodes,
        donors=donors,
        negative_ratio=1.0,
        pair_mode='exhaustive',
        return_pair_info=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    if network == 'shallow':
        model = ShallowSiameseNetwork(
            n_snps=n_snps,
            embedding_dim=embedding_dim,
            similarity='cosine',
        ).to(device)
    else:
        model = SiameseNetwork(
            n_snps=n_snps,
            embedding_dim=embedding_dim,
            similarity='cosine',
        ).to(device)

    model.load_state_dict(state_dict, strict=True)

    metrics = evaluate(model, loader, device, decision_threshold=0.9)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    score_table = metrics.pop('score_table')
    score_csv = output_dir / 'score_pairs.csv'
    summary_json = output_dir / 'evaluation_summary.json'

    score_table.to_csv(score_csv, index=False)

    with open(summary_json, 'w') as f:
        json.dump({k: v for k, v in metrics.items() if k != 'confusion_matrix'}, f, indent=2)

    print("\nEvaluation results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC:      {metrics['auc']:.4f}")
    print(f"  AP:       {metrics['ap']:.4f}")
    print(f"  Brier:    {metrics['brier']:.4f}")
    print(f"  Log loss: {metrics['log_loss']:.4f}")
    print(f"  Conf.:    {metrics['confidence']:.4f}")
    print(f"  Entropy:  {metrics['entropy']:.4f}")
    print(f"  CM:       {metrics['confusion_matrix']}")
    print(f"\nScore table saved to: {score_csv}")
    print(f"Summary saved to: {summary_json}")


if __name__ == '__main__':
    main()

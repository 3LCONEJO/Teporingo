#!/usr/bin/env python3
"""Training script for the Siamese network."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import load_npz
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from buildnload import GenotypeMatchingDataset
from teporingo_demultiplexing.config import load_simple_yaml
from teporingo_demultiplexing.pipeline import normalize_pair_mode, run_pipeline
from teporingo_demultiplexing.models.siamese_network import (
    SiameseNetwork,
    ShallowSiameseNetwork,
)


LOO_FOLDS = {
    'A': ('B_C', 'A'),
    'B': ('A_C', 'B'),
    'C': ('A_B', 'C'),
}

AVAILABLE_BATCHES = ('A', 'B', 'C')


def build_evaluate_config(config, summary, training_config):
    """Merge evaluation settings from YAML with defaults from the trained batch."""

    pipeline_config = config.get('pipeline', {}) or {}
    evaluate_config = config.get('evaluate', {}) or {}
    analysis_config = evaluate_config.get('analysis', {}) or {}

    prepared_batches = summary['pipeline'].get('prepared_batches', [])
    batch_label = training_config.get('batch_label')
    selected_batch = None
    for batch in prepared_batches:
        if batch['batch_label'] == batch_label:
            selected_batch = batch
            break

    if selected_batch is None and prepared_batches:
        selected_batch = prepared_batches[0]

    default_model_dir = Path(training_config['output_dir'])
    default_analysis_dir = Path(analysis_config.get('output_dir', default_model_dir))

    merged = {
        'model': str(evaluate_config.get('model', default_model_dir / 'best_model.pt')),
        'vcf_matrix': str(evaluate_config.get('vcf_matrix', selected_batch['vcf_matrix'] if selected_batch else pipeline_config.get('vcf_matrix', ''))),
        'bam_matrix': str(evaluate_config.get('bam_matrix', selected_batch['bam_matrix'] if selected_batch else pipeline_config.get('bam_matrix', ''))),
        'metadata': str(evaluate_config.get('metadata', selected_batch['metadata'] if selected_batch else pipeline_config.get('metadata', ''))),
        'output_dir': str(evaluate_config.get('output_dir', default_model_dir)),
        'batch_size': int(evaluate_config.get('batch_size', training_config.get('batch_size', 64))),
        'num_workers': int(evaluate_config.get('num_workers', training_config.get('num_workers', 4))),
        'device': evaluate_config.get('device'),
        'embedding_dim': evaluate_config.get('embedding_dim'),
        'analysis': {
            'score_csv': str(analysis_config.get('score_csv', default_analysis_dir / 'score_pairs.csv')),
            'output_dir': str(analysis_config.get('output_dir', default_analysis_dir)),
            'prefix': analysis_config.get('prefix', 'analyze_output'),
            'unsupervised_method': analysis_config.get('unsupervised_method', 'kmeans'),
            'random_state': int(analysis_config.get('random_state', 42)),
            'save_plot': bool(analysis_config.get('save_plot', False)),
        },
    }

    return merged


def run_post_training_evaluation(evaluate_config):
    """Run the evaluation and analysis CLIs using the YAML-driven arguments."""

    evaluation_cmd = [
        sys.executable,
        '-m',
        'teporingo_demultiplexing.models.evaluate_siamese',
        '--model',
        evaluate_config['model'],
        '--vcf-matrix',
        evaluate_config['vcf_matrix'],
        '--bam-matrix',
        evaluate_config['bam_matrix'],
        '--metadata',
        evaluate_config['metadata'],
        '--output-dir',
        evaluate_config['output_dir'],
    ]

    if evaluate_config.get('batch_size') is not None:
        evaluation_cmd.extend(['--batch-size', str(evaluate_config['batch_size'])])
    if evaluate_config.get('num_workers') is not None:
        evaluation_cmd.extend(['--num-workers', str(evaluate_config['num_workers'])])
    if evaluate_config.get('device'):
        evaluation_cmd.extend(['--device', str(evaluate_config['device'])])
    if evaluate_config.get('embedding_dim') is not None:
        evaluation_cmd.extend(['--embedding-dim', str(evaluate_config['embedding_dim'])])

    print('\n[4/4] Running evaluation...')
    subprocess.run(evaluation_cmd, check=True)

    analysis = evaluate_config['analysis']
    analysis_cmd = [
        sys.executable,
        '-m',
        'analyze_output',
        '--score-csv',
        analysis['score_csv'],
        '--output-dir',
        analysis['output_dir'],
        '--unsupervised-method',
        analysis['unsupervised_method'],
        '--prefix',
        analysis['prefix'],
        '--random-state',
        str(analysis['random_state']),
    ]

    if analysis.get('save_plot'):
        analysis_cmd.append('--save-plot')

    print('\n[5/4] Running output analysis...')
    subprocess.run(analysis_cmd, check=True)


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


def load_data(vcf_path, bam_path, metadata_path, assignments_path=None):
    print('Loading data...')

    vcf_data = torch.load(vcf_path, weights_only=False)
    X_VCF = vcf_data['X_VCF']
    donors = vcf_data['donors']

    X_BAM = load_npz(bam_path)

    metadata = torch.load(metadata_path, weights_only=False)
    barcodes = metadata.get('barcodes', [])
    assignments = metadata.get('assignments', {})
    if assignments_path is not None:
        assignments = load_assignments_table(assignments_path)

    print(f'  X_VCF: {X_VCF.shape}')
    print(f'  X_BAM: {X_BAM.shape}')
    print(f'  Donors: {len(donors)}')
    print(f'  Cells: {len(barcodes)}')
    print(f'  Assignments: {len(assignments)}')

    return X_VCF, X_BAM, donors, barcodes, assignments


def load_prepared_batch(batch_info, assignments_path):
    """Load one batch prepared by the pipeline."""

    return load_data(
        batch_info['vcf_matrix'],
        batch_info['bam_matrix'],
        batch_info['metadata'],
        assignments_path=assignments_path,
    )


def load_prepared_case(case_dir, case_prefix):
    case_dir = Path(case_dir)
    vcf_path = case_dir / f'{case_prefix}_vcf_matrix.pt'
    bam_path = case_dir / f'{case_prefix}_bam_matrix.npz'
    metadata_path = case_dir / f'{case_prefix}_metadata.pt'

    if not vcf_path.exists():
        raise FileNotFoundError(f'Could not find {vcf_path}')
    if not bam_path.exists():
        raise FileNotFoundError(f'Could not find {bam_path}')
    if not metadata_path.exists():
        raise FileNotFoundError(f'Could not find {metadata_path}')

    return load_data(vcf_path, bam_path, metadata_path)


def create_random_dataloaders(
    X_VCF,
    X_BAM,
    donors,
    barcodes,
    assignments,
    batch_size=64,
    train_ratio=0.7,
    val_ratio=0.15,
    num_workers=4,
    seed=42,
    negative_ratio=1.0,
    pair_mode='sampled',
    pin_memory=False,
    hard_negatives_k=0,
):
    print('\nCreating datasets...')

    full_dataset = GenotypeMatchingDataset(
        X_vcf=X_VCF,
        X_bam_sparse=X_BAM,
        cell_to_donor=assignments,
        barcodes=barcodes,
        donors=donors,
        negative_ratio=negative_ratio,
        pair_mode=pair_mode,
        hard_negatives_k=hard_negatives_k,
    )

    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=generator,
    )

    print(f'  Train: {len(train_dataset)} pairs')
    print(f'  Val:   {len(val_dataset)} pairs')
    print(f'  Test:  {len(test_dataset)} pairs')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def create_loo_dataloaders(
    train_data,
    test_data,
    batch_size=64,
    train_ratio=0.7,
    val_ratio=0.15,
    num_workers=4,
    seed=42,
    negative_ratio=1.0,
    pair_mode='sampled',
    pin_memory=False,
    hard_negatives_k=0,
):
    print('\nCreating leave-one-batch-out datasets...')

    X_VCF_train, X_BAM_train, donors_train, barcodes_train, assignments_train = train_data
    X_VCF_test, X_BAM_test, donors_test, barcodes_test, assignments_test = test_data

    train_dataset = GenotypeMatchingDataset(
        X_vcf=X_VCF_train,
        X_bam_sparse=X_BAM_train,
        cell_to_donor=assignments_train,
        barcodes=barcodes_train,
        donors=donors_train,
        negative_ratio=negative_ratio,
        pair_mode=pair_mode,
        hard_negatives_k=hard_negatives_k,
    )

    test_dataset = GenotypeMatchingDataset(
        X_vcf=X_VCF_test,
        X_bam_sparse=X_BAM_test,
        cell_to_donor=assignments_test,
        barcodes=barcodes_test,
        donors=donors_test,
        negative_ratio=negative_ratio,
        pair_mode=pair_mode,
        hard_negatives_k=0,  # do not compute hard negatives for test set
    )

    n_total = len(train_dataset)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1
    n_rest = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, _ = random_split(
        train_dataset,
        [n_train, n_val, n_rest],
        generator=generator,
    )

    print(f'  Train: {len(train_subset)} pairs')
    print(f'  Val:   {len(val_subset)} pairs')
    print(f'  Test:  {len(test_dataset)} pairs (batch holdout)')

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def resolve_loo_paths(demuxlet_root, holdout_batch):
    if holdout_batch not in LOO_FOLDS:
        raise ValueError(f'Unsupported batch: {holdout_batch}. Use A, B, or C.')

    train_combo, test_batch = LOO_FOLDS[holdout_batch]
    demuxlet_root = Path(demuxlet_root)

    train_dir = demuxlet_root / 'merges' / train_combo
    test_dir = demuxlet_root / 'batches' / test_batch

    return {
        'train_dir': train_dir,
        'test_dir': test_dir,
        'train_combo': train_combo,
        'test_batch': test_batch,
    }


def resolve_single_batch_paths(demuxlet_root, batch_name):
    if batch_name not in AVAILABLE_BATCHES:
        raise ValueError(f'Unsupported batch: {batch_name}. Use A, B, or C.')

    demuxlet_root = Path(demuxlet_root)
    batch_dir = demuxlet_root / 'batches' / batch_name

    return {
        'batch_dir': batch_dir,
        'batch_prefix': batch_name,
        'batch_name': batch_name,
    }


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')

    for cell_batch, geno_batch, labels in pbar:
        cell_batch = cell_batch.to(device)
        geno_batch = geno_batch.to(device)
        labels = labels.to(device)

        predictions = model(cell_batch, geno_batch)
        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        pred_labels = (predictions > 0.5).float()
        correct += (pred_labels == labels).sum().item()
        total += len(labels)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})

    return total_loss / total, correct / total


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, epoch, phase='Val'):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    pbar = tqdm(loader, desc=f'Epoch {epoch} [{phase}]')

    for cell_batch, geno_batch, labels in pbar:
        cell_batch = cell_batch.to(device)
        geno_batch = geno_batch.to(device)
        labels = labels.to(device)

        predictions = model(cell_batch, geno_batch)
        loss = criterion(predictions, labels)

        total_loss += loss.item() * len(labels)
        pred_labels = (predictions > 0.9).float()
        correct += (pred_labels == labels).sum().item()
        total += len(labels)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})

    from sklearn.metrics import average_precision_score, roc_auc_score

    avg_loss = total_loss / total
    accuracy = correct / total
    if len(set(int(label) for label in all_labels)) < 2:
        auc = float('nan')
    else:
        auc = roc_auc_score(all_labels, all_predictions)
    ap = average_precision_score(all_labels, all_predictions)

    return avg_loss, accuracy, auc, ap


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    output_dir,
    patience=10,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_ap': [],
    }

    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 60}")
        print(f'Epoch {epoch}/{epochs}')
        print(f"{'=' * 60}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_auc, val_ap = validate_epoch(model, val_loader, criterion, device, epoch, phase='Val')

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_ap'].append(val_ap)

        print('\nEpoch Summary:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}')
        print(f'  Val AUC:    {val_auc:.4f} | Val AP:    {val_ap:.4f}')
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                },
                output_dir / 'best_model.pt',
            )
            print(f'  ✓ New best model saved (val_loss: {val_loss:.4f})')
        else:
            epochs_no_improve += 1
            print(f'  No mejora por {epochs_no_improve} epochs')
            if epochs_no_improve >= patience:
                print(f'\nEarly stopping after {epoch} epochs')
                break

    print(f"\n{'=' * 60}")
    print('Evaluation on the test set')
    print(f"{'=' * 60}")

    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_auc, test_ap = validate_epoch(model, test_loader, criterion, device, epoch='Final', phase='Test')

    print('\nTest Results:')
    print(f'  Loss: {test_loss:.4f}')
    print(f'  Accuracy: {test_acc:.4f}')
    print(f'  AUC: {test_auc:.4f}')
    print(f'  AP: {test_ap:.4f}')

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return history


def build_training_config(config):
    """Merge training settings from dedicated and legacy YAML sections."""

    pipeline_config = config.get('pipeline', {}) or {}
    training_config = config.get('training', {}) or config.get('model_training', {}) or {}

    merged = {
        'output_dir': training_config.get('output_dir', pipeline_config.get('output_dir', 'models/siamese')),
        'split_mode': training_config.get('split_mode', pipeline_config.get('split_mode', 'single-batch')),
        'batch_label': training_config.get('batch_label', pipeline_config.get('batch_label')),
        'epochs': training_config.get('epochs', 50),
        'batch_size': training_config.get('batch_size', 64),
        'learning_rate': training_config.get('learning_rate', 1e-3),
        'optimizer': training_config.get('optimizer', 'adamw'),
        'momentum': training_config.get('momentum', 0.9),
        'embedding_dim': training_config.get('embedding_dim', 512),
        'network': training_config.get('network', 'siamese'),
        'patience': training_config.get('patience', 10),
        'overwrite': training_config.get('overwrite', pipeline_config.get('overwrite', False)),
        'num_workers': training_config.get('num_workers', pipeline_config.get('num_workers', 4)),
        'seed': training_config.get('seed', 42),
        'negative_ratio': training_config.get('negative_ratio', pipeline_config.get('negative_ratio', 1.0)),
        'hard_negatives_k': training_config.get('hard_negatives_k', pipeline_config.get('hard_negatives_k', 0)),
        'pair_mode': normalize_pair_mode(training_config.get('pair_mode', pipeline_config.get('pair_mode', 'sampled'))),
        'skip_dataloader_smoke_test': training_config.get('skip_dataloader_smoke_test', pipeline_config.get('skip_dataloader_smoke_test', False)),
        'single_batch': training_config.get('single_batch', pipeline_config.get('single_batch')),
        'holdout_batch': training_config.get('holdout_batch', pipeline_config.get('holdout_batch')),
        'demuxlet_root': training_config.get('demuxlet_root', pipeline_config.get('demuxlet_root', 'Proyecto/data/demuxlet')),
    }

    return merged


def should_skip_training(output_dir, overwrite):
    """Return True when an existing checkpoint should be reused."""

    best_model_path = Path(output_dir) / 'best_model.pt'
    return best_model_path.exists() and not overwrite


def train_from_pipeline(summary, training_config, evaluate_config=None):
    """Train immediately after pipeline preparation."""

    prepared_batches = summary['pipeline'].get('prepared_batches', [])
    if not prepared_batches:
        raise ValueError('Pipeline did not produce any prepared batches')

    batch_label = training_config.get('batch_label')
    if batch_label is None:
        batch_label = prepared_batches[0]['batch_label']

    selected_batch = None
    for batch in prepared_batches:
        if batch['batch_label'] == batch_label:
            selected_batch = batch
            break

    if selected_batch is None:
        available = ', '.join(batch['batch_label'] for batch in prepared_batches)
        raise ValueError(f"Batch '{batch_label}' not found. Available batches: {available}")

    output_dir = Path(training_config['output_dir'])
    if should_skip_training(output_dir, training_config.get('overwrite', False)):
        print(f"\nSkipping training because {output_dir / 'best_model.pt'} already exists and overwrite is False")
        if evaluate_config is not None:
            run_post_training_evaluation(evaluate_config)
        return

    assignments_path = summary['pipeline']['assignments']
    X_VCF, X_BAM, donors, barcodes, assignments = load_prepared_batch(selected_batch, assignments_path)

    split_mode = training_config['split_mode']
    if split_mode not in {'random', 'leave-one-batch-out', 'single-batch'}:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    batch_size = training_config['batch_size']
    num_workers = training_config['num_workers']
    seed = training_config['seed']
    negative_ratio = training_config['negative_ratio']
    hard_negatives_k = training_config['hard_negatives_k']
    pair_mode = training_config['pair_mode']

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin_memory = device.type == 'cuda'
    print(f'Using device: {device}')

    if split_mode == 'random':
        train_loader, val_loader, test_loader = create_random_dataloaders(
            X_VCF,
            X_BAM,
            donors,
            barcodes,
            assignments,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            negative_ratio=negative_ratio,
            pair_mode=pair_mode,
            hard_negatives_k=hard_negatives_k,
            pin_memory=pin_memory,
        )
    elif split_mode == 'leave-one-batch-out':
        holdout_batch = training_config.get('holdout_batch')
        if holdout_batch is None:
            raise ValueError('--holdout-batch is required when split_mode=leave-one-batch-out')

        paths = resolve_loo_paths(training_config['demuxlet_root'], holdout_batch)
        print(f"Leave-one-batch-out: holdout={paths['test_batch']} | train={paths['train_combo']}")

        train_data = load_prepared_case(paths['train_dir'], paths['train_combo'])
        test_data = load_prepared_case(paths['test_dir'], paths['test_batch'])

        train_loader, val_loader, test_loader = create_loo_dataloaders(
            train_data=train_data,
            test_data=test_data,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            negative_ratio=negative_ratio,
            pair_mode=pair_mode,
            hard_negatives_k=hard_negatives_k,
            pin_memory=pin_memory,
        )
        X_VCF = train_data[0]
    else:
        train_loader, val_loader, test_loader = create_random_dataloaders(
            X_VCF,
            X_BAM,
            donors,
            barcodes,
            assignments,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            negative_ratio=negative_ratio,
            pair_mode=pair_mode,
            hard_negatives_k=hard_negatives_k,
            pin_memory=pin_memory,
        )

    n_snps = X_VCF.shape[1]

    if training_config['network'] == 'shallow':
        model = ShallowSiameseNetwork(n_snps=n_snps, embedding_dim=min(256, training_config['embedding_dim']), similarity='cosine').to(device)
    else:
        model = SiameseNetwork(n_snps=n_snps, embedding_dim=training_config['embedding_dim'], similarity='cosine').to(device)

    print('\nModel created:')
    print(f"  Type: {training_config['network']}")
    print(f'  SNPs: {n_snps}')
    print(f"  Embedding dim: {training_config['embedding_dim']}")
    print(f'  Total parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.SmoothL1Loss()

    if training_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_config['learning_rate'],
            momentum=training_config['momentum'],
            weight_decay=1e-4,
            nesterov=True,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=1e-4,
        )

    print(f"  Optimizer: {training_config['optimizer']}")
    if training_config['optimizer'] == 'sgd':
        print(f"  Momentum: {training_config['momentum']}")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=training_config['epochs'],
        output_dir=training_config['output_dir'],
        patience=training_config['patience'],
    )

    print('\n✓ Training completed!')
    print(f"  Models saved in: {training_config['output_dir']}/")

    if evaluate_config is not None:
        run_post_training_evaluation(evaluate_config)


def run_from_config(config_path):
    """Run the pipeline and return its summary."""

    return run_pipeline(config_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='YAML config with pipeline and training settings')
    parser.add_argument('--vcf-matrix')
    parser.add_argument('--bam-matrix')
    parser.add_argument('--metadata')
    parser.add_argument('--output-dir')
    parser.add_argument(
        '--split-mode',
        choices=['random', 'leave-one-batch-out', 'single-batch'],
        default='random',
        help='Split mode: random, leave-one-batch-out, or single-batch',
    )
    parser.add_argument('--holdout-batch', choices=list(AVAILABLE_BATCHES))
    parser.add_argument('--single-batch', choices=list(AVAILABLE_BATCHES))
    parser.add_argument('--demuxlet-root', default='Proyecto/data/demuxlet')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', choices=['adamw', 'sgd'], default='adamw')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum used when --optimizer=sgd')
    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--network', choices=['siamese', 'shallow'], default='siamese')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--overwrite', action='store_true', help='Retrain even if best_model.pt already exists')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--negative-ratio', type=float, default=1.0)
    parser.add_argument('--hard-negatives-k', type=int, default=0, help='Number of hard negatives per cell (0 disables it)')
    parser.add_argument('--pair-mode', choices=['sampled', 'exhaustive'], default='sampled')
    parser.add_argument('--batch-label', help='Batch label to train on when using --config')

    args = parser.parse_args()

    if args.config:
        config = load_simple_yaml(args.config)
        training_config = build_training_config(config)
        if args.output_dir:
            training_config['output_dir'] = args.output_dir
        if args.batch_label:
            training_config['batch_label'] = args.batch_label
        if args.overwrite:
            training_config['overwrite'] = True

        summary = run_pipeline(args.config)
        evaluate_config = build_evaluate_config(config, summary, training_config)
        summary['evaluate'] = evaluate_config
        return train_from_pipeline(summary, training_config, evaluate_config)

    if not args.output_dir:
        raise ValueError('--output-dir is required when not using --config')

    if should_skip_training(args.output_dir, args.overwrite):
        print(f"Skipping training because {Path(args.output_dir) / 'best_model.pt'} already exists and overwrite is False")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin_memory = device.type == 'cuda'
    print(f'Using device: {device}')

    if args.split_mode == 'random':
        if not args.vcf_matrix or not args.bam_matrix or not args.metadata:
            raise ValueError('--vcf-matrix, --bam-matrix, and --metadata are required in split-mode=random')

        X_VCF, X_BAM, donors, barcodes, assignments = load_data(args.vcf_matrix, args.bam_matrix, args.metadata)
        train_loader, val_loader, test_loader = create_random_dataloaders(
            X_VCF,
            X_BAM,
            donors,
            barcodes,
            assignments,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            negative_ratio=args.negative_ratio,
            pair_mode=args.pair_mode,
            hard_negatives_k=args.hard_negatives_k,
            pin_memory=pin_memory,
        )
    elif args.split_mode == 'leave-one-batch-out':
        if args.holdout_batch is None:
            raise ValueError('--holdout-batch is required when split-mode=leave-one-batch-out')

        paths = resolve_loo_paths(args.demuxlet_root, args.holdout_batch)
        print(f"Leave-one-batch-out: holdout={paths['test_batch']} | train={paths['train_combo']}")

        train_data = load_prepared_case(paths['train_dir'], paths['train_combo'])
        test_data = load_prepared_case(paths['test_dir'], paths['test_batch'])

        train_loader, val_loader, test_loader = create_loo_dataloaders(
            train_data=train_data,
            test_data=test_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            negative_ratio=args.negative_ratio,
            pair_mode=args.pair_mode,
            hard_negatives_k=args.hard_negatives_k,
            pin_memory=pin_memory,
        )

        X_VCF = train_data[0]
    else:
        if args.single_batch is None:
            raise ValueError('--single-batch is required when split-mode=single-batch')

        paths = resolve_single_batch_paths(args.demuxlet_root, args.single_batch)
        print(f"Single-batch smoke test: batch={paths['batch_name']}")

        batch_data = load_prepared_case(paths['batch_dir'], paths['batch_prefix'])
        X_VCF, X_BAM, donors, barcodes, assignments = batch_data

        train_loader, val_loader, test_loader = create_random_dataloaders(
            X_VCF,
            X_BAM,
            donors,
            barcodes,
            assignments,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            negative_ratio=args.negative_ratio,
            pair_mode=args.pair_mode,
            hard_negatives_k=args.hard_negatives_k,
            pin_memory=pin_memory,
        )

    n_snps = X_VCF.shape[1]

    if args.network == 'shallow':
        model = ShallowSiameseNetwork(n_snps=n_snps, embedding_dim=min(256, args.embedding_dim), similarity='cosine').to(device)
    else:
        model = SiameseNetwork(n_snps=n_snps, embedding_dim=args.embedding_dim, similarity='cosine').to(device)

    print('\nModel created:')
    print(f'  Type: {args.network}')
    print(f'  SNPs: {n_snps}')
    print(f'  Embedding dim: {args.embedding_dim}')
    print(f'  Total parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.SmoothL1Loss()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=1e-4,
            nesterov=True,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
        )

    print(f'  Optimizer: {args.optimizer}')
    if args.optimizer == 'sgd':
        print(f'  Momentum: {args.momentum}')

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        output_dir=args.output_dir,
        patience=args.patience,
    )

    print('\n✓ Training completed!')
    print(f'  Models saved in: {args.output_dir}/')


if __name__ == '__main__':
    main()

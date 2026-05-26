#!/usr/bin/env python3
"""
Analiza los resultados del smoke test de Siamese sobre `score_pairs.csv`.

Este script resume qué tan bien separa el modelo el donante verdadero frente a
los donantes incorrectos por célula, y guarda una tabla con métricas por célula.

Uso:
    python -m Proyecto.analyze_smoke_a \
        --score-csv models/smoke_A/score_pairs.csv \
        --output-dir models/smoke_A
"""

import argparse
import json
from pathlib import Path
import importlib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


def compute_threshold_metrics(df, threshold):
    """
    Calcula métricas binarias usando un umbral sobre el score.
    """

    y_true = df['label'].astype(int).to_numpy()
    y_pred = (df['score'].astype(float).to_numpy() >= threshold).astype(int)

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    positive_accuracy = float((y_pred[pos_mask] == 1).mean()) if pos_mask.any() else float('nan')
    negative_accuracy = float((y_pred[neg_mask] == 0).mean()) if neg_mask.any() else float('nan')
    global_accuracy = float((y_pred == y_true).mean())
    balanced_accuracy = float(np.nanmean([positive_accuracy, negative_accuracy]))

    return {
        'threshold': float(threshold),
        'global_accuracy': global_accuracy,
        'positive_accuracy': positive_accuracy,
        'negative_accuracy': negative_accuracy,
        'balanced_accuracy': balanced_accuracy,
    }


def sweep_thresholds(df, start=0.0, end=1.0, num=101):
    """
    Barre umbrales y devuelve el mejor según balanced accuracy.
    """

    thresholds = np.linspace(start, end, num)
    rows = [compute_threshold_metrics(df, thr) for thr in thresholds]
    sweep_df = pd.DataFrame(rows)

    best_idx = sweep_df['balanced_accuracy'].idxmax()
    best_row = sweep_df.loc[best_idx].to_dict()

    return best_row, sweep_df


def compute_confusion_matrix(y_true, y_pred):
    """
    Devuelve una matriz de confusión 2x2 en formato lista para JSON.
    """

    cm = confusion_matrix(
        np.asarray(y_true, dtype=int),
        np.asarray(y_pred, dtype=int),
        labels=[0, 1],
    )
    return cm.tolist()


def fit_unsupervised_score_classifier(scores, method='kmeans', random_state=42):
    """
    Clasifica scores binarios sin usar labels reales.

    El método aprende dos grupos sobre el score 1D y asigna la clase 1 al
    grupo con centroide más alto.
    """

    score_values = np.asarray(scores, dtype=float).reshape(-1, 1)

    if score_values.shape[0] == 0:
        raise ValueError('No hay scores para clasificar')

    if method not in {'kmeans', 'nearest_centroid'}:
        raise ValueError(f'Método no supervisado no soportado: {method}')

    model = KMeans(n_clusters=2, n_init='auto', random_state=random_state)
    cluster_ids = model.fit_predict(score_values)
    centers = model.cluster_centers_.reshape(-1)
    positive_cluster = int(np.argmax(centers))

    if method == 'kmeans':
        y_pred = (cluster_ids == positive_cluster).astype(int)
    else:
        distances = np.abs(score_values - centers.reshape(1, -1))
        nearest_center = np.argmin(distances, axis=1)
        y_pred = (nearest_center == positive_cluster).astype(int)

    return y_pred, {
        'method': method,
        'centers': centers.tolist(),
        'positive_cluster': positive_cluster,
    }


def fit_kmeans_1d(scores, random_state=42):
    """
    Ajusta k-means 1D sobre scores y devuelve ids de cluster, centros y umbral.
    """

    score_values = np.asarray(scores, dtype=float).reshape(-1, 1)

    if score_values.shape[0] == 0:
        raise ValueError('No hay scores para clasificar')

    model = KMeans(n_clusters=2, n_init='auto', random_state=random_state)
    cluster_ids = model.fit_predict(score_values)
    centers = model.cluster_centers_.reshape(-1)
    ordered_clusters = np.argsort(centers)
    low_cluster, high_cluster = int(ordered_clusters[0]), int(ordered_clusters[1])
    threshold = float(np.mean(centers))

    return cluster_ids, centers, low_cluster, high_cluster, threshold


def analyze_score_pairs(score_csv_path: Path, unsupervised_method='kmeans', random_state=42):
    """
    Lee la tabla de scores por par y calcula métricas por célula.
    """

    df = pd.read_csv(score_csv_path)

    required_columns = {
        'sample_idx',
        'cell_idx',
        'cell_barcode',
        'donor_idx',
        'donor_name',
        'true_donor_idx',
        'true_donor_name',
        'label',
        'score',
        'prediction',
        'is_correct_pair',
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f'Faltan columnas en {score_csv_path}: {sorted(missing)}')

    # Métricas globales del clasificador/par.
    y_true = df['label'].astype(float).to_numpy()
    y_score = df['score'].astype(float).to_numpy()
    y_pred = df['prediction'].astype(float).to_numpy()
    y_unsup_pred, unsup_info = fit_unsupervised_score_classifier(
        y_score,
        method=unsupervised_method,
        random_state=random_state,
    )

    global_accuracy = float((y_true == y_pred).mean())
    unsupervised_accuracy = float((y_true == y_unsup_pred).mean())
    positive_mask = df['label'] == 1
    negative_mask = df['label'] == 0

    positive_accuracy = float((df.loc[positive_mask, 'prediction'] == 1).mean())
    negative_accuracy = float((df.loc[negative_mask, 'prediction'] == 0).mean())
    unsupervised_positive_accuracy = float((y_unsup_pred[positive_mask.to_numpy()] == 1).mean())
    unsupervised_negative_accuracy = float((y_unsup_pred[negative_mask.to_numpy()] == 0).mean())
    global_positive_mean = float(df.loc[df['label'] == 1, 'score'].mean())
    global_negative_mean = float(df.loc[df['label'] == 0, 'score'].mean())

    threshold_050 = compute_threshold_metrics(df, 0.5)
    best_threshold, threshold_sweep_df = sweep_thresholds(df, start=0.0, end=1.0, num=201)
    best_threshold_pred = (y_score >= best_threshold['threshold']).astype(int)

    # Resumen por célula.
    cell_groups = []
    for cell_idx, group in df.groupby('cell_idx', sort=False):
        group = group.sort_values('score', ascending=False)
        true_row = group.loc[group['label'] == 1].iloc[0]
        top_row = group.iloc[0]
        second_row = group.iloc[1] if len(group) > 1 else None

        cell_groups.append({
            'cell_idx': int(cell_idx),
            'cell_barcode': true_row['cell_barcode'],
            'true_donor_idx': int(true_row['true_donor_idx']),
            'true_donor_name': true_row['true_donor_name'],
            'true_score': float(true_row['score']),
            'true_prediction': float(true_row['prediction']),
            'top_donor_idx': int(top_row['donor_idx']),
            'top_donor_name': top_row['donor_name'],
            'top_score': float(top_row['score']),
            'second_donor_idx': int(second_row['donor_idx']) if second_row is not None else -1,
            'second_donor_name': second_row['donor_name'] if second_row is not None else '',
            'second_score': float(second_row['score']) if second_row is not None else np.nan,
            'top_is_true_donor': int(top_row['donor_idx'] == true_row['true_donor_idx']),
            'margin_top_minus_true': float(top_row['score'] - true_row['score']),
            'margin_true_minus_second': float(true_row['score'] - (second_row['score'] if second_row is not None else np.nan)),
            'n_candidates': int(len(group)),
        })

    cell_df = pd.DataFrame(cell_groups)

    top1_accuracy = float(cell_df['top_is_true_donor'].mean())
    avg_true_score = float(cell_df['true_score'].mean())
    avg_top_score = float(cell_df['top_score'].mean())
    avg_margin = float(cell_df['true_score'].sub(cell_df['top_score']).mean())
    avg_true_minus_second = float(cell_df['margin_true_minus_second'].mean())

    summary = {
        'n_pairs': int(len(df)),
        'n_cells': int(cell_df.shape[0]),
        'global_accuracy': global_accuracy,
        'model_confusion_matrix': compute_confusion_matrix(y_true, y_pred),
        'unsupervised_method': unsup_info['method'],
        'unsupervised_accuracy': unsupervised_accuracy,
        'unsupervised_confusion_matrix': compute_confusion_matrix(y_true, y_unsup_pred),
        'unsupervised_positive_accuracy': unsupervised_positive_accuracy,
        'unsupervised_negative_accuracy': unsupervised_negative_accuracy,
        'unsupervised_centers': unsup_info['centers'],
        'unsupervised_positive_cluster': unsup_info['positive_cluster'],
        'positive_accuracy': positive_accuracy,
        'negative_accuracy': negative_accuracy,
        'threshold_0_5_global_accuracy': threshold_050['global_accuracy'],
        'threshold_0_5_positive_accuracy': threshold_050['positive_accuracy'],
        'threshold_0_5_negative_accuracy': threshold_050['negative_accuracy'],
        'threshold_0_5_balanced_accuracy': threshold_050['balanced_accuracy'],
        'best_threshold': best_threshold['threshold'],
        'best_threshold_global_accuracy': best_threshold['global_accuracy'],
        'best_threshold_positive_accuracy': best_threshold['positive_accuracy'],
        'best_threshold_negative_accuracy': best_threshold['negative_accuracy'],
        'best_threshold_balanced_accuracy': best_threshold['balanced_accuracy'],
        'best_threshold_confusion_matrix': compute_confusion_matrix(y_true, best_threshold_pred),
        'global_positive_mean_score': global_positive_mean,
        'global_negative_mean_score': global_negative_mean,
        'cell_top1_accuracy': top1_accuracy,
        'cell_avg_true_score': avg_true_score,
        'cell_avg_top_score': avg_top_score,
        'cell_avg_true_minus_top_margin': avg_margin,
        'cell_avg_true_minus_second_margin': avg_true_minus_second,
    }

    return summary, cell_df, df, threshold_sweep_df


def save_top1_vs_top2_plot(cell_df: pd.DataFrame, output_path: Path):
    """
    Guarda un scatter por célula con top-1 vs top-2 score.
    """

    try:
        plt = importlib.import_module('matplotlib.pyplot')
        Line2D = importlib.import_module('matplotlib.lines').Line2D
    except ImportError as exc:
        raise RuntimeError('matplotlib es necesario para generar este plot') from exc

    if cell_df.empty:
        raise ValueError('No hay células para graficar')

    plot_df = cell_df.dropna(subset=['top_score', 'second_score']).copy()
    if plot_df.empty:
        raise ValueError('No hay segundos mejores scores para graficar')

    colors = plot_df['top_is_true_donor'].map({1: '#2ca02c', 0: '#d62728'}).to_numpy()
    labels = plot_df['top_is_true_donor'].map({1: 'top-1 correcto', 0: 'top-1 incorrecto'})

    x_vals = plot_df['top_score'].astype(float).to_numpy()
    y_vals = plot_df['second_score'].astype(float).to_numpy()

    finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    plot_df = plot_df.loc[finite_mask].copy()
    x_vals = x_vals[finite_mask]
    y_vals = y_vals[finite_mask]
    colors = colors[finite_mask]
    labels = labels.loc[finite_mask]

    if plot_df.empty:
        raise ValueError('No hay valores finitos para graficar')

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        x_vals,
        y_vals,
        c=colors,
        s=24,
        alpha=0.75,
        linewidths=0,
    )

    min_val = float(min(np.min(x_vals), np.min(y_vals)))
    max_val = float(max(np.max(x_vals), np.max(y_vals)))
    pad = max(0.03, 0.05 * (max_val - min_val if max_val > min_val else 1.0))
    lower = max(0.0, min_val - pad)
    upper = min(1.0, max_val + pad)

    ax.plot([lower, upper], [lower, upper], linestyle='--', color='0.35', linewidth=1.2, label='y = x')
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel('Score del mejor donador (top-1)')
    ax.set_ylabel('Score del segundo mejor donador (top-2)')
    ax.set_title('Top-1 vs Top-2 score por célula')
    ax.grid(True, alpha=0.2)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=8, label='top-1 correcto'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=8, label='top-1 incorrecto'),
        Line2D([0], [0], linestyle='--', color='0.35', label='y = x'),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc='best')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_confusion_style_plot(cell_df: pd.DataFrame, output_path: Path, threshold=0.5):
    """
    Guarda un scatter tipo matriz de confusión.

    Eje X: score del donador verdadero.
    Eje Y: score del mejor donador incorrecto.
    Las líneas del umbral dividen el plano en 4 cuadrantes.
    """

    try:
        plt = importlib.import_module('matplotlib.pyplot')
        Line2D = importlib.import_module('matplotlib.lines').Line2D
    except ImportError as exc:
        raise RuntimeError('matplotlib es necesario para generar este plot') from exc

    if cell_df.empty:
        raise ValueError('No hay células para graficar')

    plot_df = cell_df.dropna(subset=['true_score', 'top_score']).copy()
    if plot_df.empty:
        raise ValueError('No hay scores válidos para graficar')

    x_vals = plot_df['true_score'].astype(float).to_numpy()
    y_vals = plot_df['top_score'].astype(float).to_numpy()
    correct = plot_df['top_is_true_donor'].astype(int).to_numpy()

    colors = np.where(correct == 1, '#2ca02c', '#d62728')
    finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[finite_mask]
    y_vals = y_vals[finite_mask]
    colors = colors[finite_mask]
    correct = correct[finite_mask]

    if x_vals.size == 0:
        raise ValueError('No hay valores finitos para graficar')

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        x_vals,
        y_vals,
        c=colors,
        s=24,
        alpha=0.75,
        linewidths=0,
    )

    min_val = float(min(np.min(x_vals), np.min(y_vals), threshold))
    max_val = float(max(np.max(x_vals), np.max(y_vals), threshold))
    pad = max(0.03, 0.05 * (max_val - min_val if max_val > min_val else 1.0))
    lower = max(0.0, min_val - pad)
    upper = min(1.0, max_val + pad)

    ax.axvline(threshold, linestyle='--', color='0.35', linewidth=1.2)
    ax.axhline(threshold, linestyle='--', color='0.35', linewidth=1.2)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel('Score del donador verdadero')
    ax.set_ylabel('Score del mejor donador incorrecto')
    ax.set_title(f'Diagrama tipo confusión por célula (umbral = {threshold:.3f})')
    ax.grid(True, alpha=0.2)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=8, label='top-1 correcto'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=8, label='top-1 incorrecto'),
        Line2D([0], [0], linestyle='--', color='0.35', label=f'umbral = {threshold:.3f}'),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc='best')

    ax.text(0.03, 0.97, 'TN\nambos bajos', transform=ax.transAxes, va='top', ha='left', fontsize=9, alpha=0.75)
    ax.text(0.03, 0.03, 'FN\ntrue bajo / rival alto', transform=ax.transAxes, va='bottom', ha='left', fontsize=9, alpha=0.75)
    ax.text(0.72, 0.97, 'TP\ntrue alto / rival bajo', transform=ax.transAxes, va='top', ha='left', fontsize=9, alpha=0.75)
    ax.text(0.72, 0.03, 'FP\nambos altos', transform=ax.transAxes, va='bottom', ha='left', fontsize=9, alpha=0.75)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_kmeans_score_plot(score_df: pd.DataFrame, output_path: Path, random_state=42):
    """
    Guarda una visualización 1D fiel a k-means: scores, clusters y centros.
    """

    try:
        plt = importlib.import_module('matplotlib.pyplot')
        Line2D = importlib.import_module('matplotlib.lines').Line2D
    except ImportError as exc:
        raise RuntimeError('matplotlib es necesario para generar este plot') from exc

    if score_df.empty:
        raise ValueError('No hay scores para graficar')

    scores = score_df['score'].astype(float).to_numpy()
    cluster_ids, centers, low_cluster, high_cluster, threshold = fit_kmeans_1d(scores, random_state=random_state)

    plot_df = score_df.copy()
    plot_df['cluster_id'] = cluster_ids
    plot_df['cluster_rank'] = np.where(plot_df['cluster_id'].to_numpy() == high_cluster, 1, 0)

    labels = plot_df['label'].astype(int).to_numpy()
    colors = np.where(labels == 1, '#2ca02c', '#d62728')
    alphas = np.full(len(plot_df), 0.7, dtype=float)
    y_vals = np.random.default_rng(random_state).normal(loc=0.0, scale=0.045, size=len(plot_df))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(
        plot_df['score'].to_numpy(),
        y_vals,
        c=colors,
        s=18,
        alpha=alphas,
        linewidths=0,
    )

    ax.axvline(centers[low_cluster], linestyle='--', color='#d62728', linewidth=1.4, label=f'centro cluster bajo = {centers[low_cluster]:.3f}')
    ax.axvline(centers[high_cluster], linestyle='--', color='#2ca02c', linewidth=1.4, label=f'centro cluster alto = {centers[high_cluster]:.3f}')
    ax.axvline(threshold, linestyle=':', color='0.35', linewidth=1.6, label=f'frontera k-means = {threshold:.3f}')

    ax.set_yticks([])
    ax.set_xlabel('Score del par célula-donante')
    ax.set_title('K-means 1D sobre scores de pares')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.18, 0.18)
    ax.grid(True, axis='x', alpha=0.2)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=7, label='negativo real'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=7, label='positivo real'),
        Line2D([0], [0], linestyle='--', color='0.35', label='frontera k-means'),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc='best')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Analiza score_pairs.csv del smoke test')
    parser.add_argument(
        '--score-csv',
        default='models/smoke_A/score_pairs.csv',
        help='Ruta al CSV con scores por par célula-donante',
    )
    parser.add_argument(
        '--output-dir',
        default='models/smoke_A',
        help='Directorio donde guardar el resumen y la tabla por célula',
    )
    parser.add_argument(
        '--prefix',
        default='smoke_A_analysis',
        help='Prefijo para archivos de salida',
    )
    parser.add_argument(
        '--save-plot',
        action='store_true',
        help='Guardar scatter top-1 vs top-2 por célula',
    )
    parser.add_argument(
        '--plot-name',
        default='kmeans_score_plot.png',
        help='Nombre del archivo de la figura k-means',
    )
    parser.add_argument(
        '--unsupervised-method',
        choices=['kmeans', 'nearest_centroid'],
        default='kmeans',
        help='Método no supervisado para binarizar score',
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Semilla para el método no supervisado',
    )

    args = parser.parse_args()

    score_csv_path = Path(args.score_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, cell_df, raw_df, threshold_sweep_df = analyze_score_pairs(
        score_csv_path,
        unsupervised_method=args.unsupervised_method,
        random_state=args.random_state,
    )

    summary_path = output_dir / f'{args.prefix}.json'
    cell_path = output_dir / f'{args.prefix}_by_cell.csv'
    pair_path = output_dir / f'{args.prefix}_pairs.csv'
    sweep_path = output_dir / f'{args.prefix}_threshold_sweep.csv'

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    cell_df.to_csv(cell_path, index=False)
    raw_df.to_csv(pair_path, index=False)
    threshold_sweep_df.to_csv(sweep_path, index=False)

    plot_path = output_dir / args.plot_name
    if args.save_plot:
        save_kmeans_score_plot(raw_df, plot_path, random_state=args.random_state)

    print('Resumen de smoke_A:')
    print(f"  Pairs:              {summary['n_pairs']}")
    print(f"  Células:            {summary['n_cells']}")
    print(f"  Model accuracy:      {summary['global_accuracy']:.4f}")
    print(f"  Model CM:            {summary['model_confusion_matrix']}")
    print(f"  Unsupervised acc:    {summary['unsupervised_accuracy']:.4f}")
    print(f"  Unsupervised CM:     {summary['unsupervised_confusion_matrix']}")
    print(f"  Positive accuracy:   {summary['positive_accuracy']:.4f}")
    print(f"  Negative accuracy:   {summary['negative_accuracy']:.4f}")
    print(f"  Unsup pos acc:       {summary['unsupervised_positive_accuracy']:.4f}")
    print(f"  Unsup neg acc:       {summary['unsupervised_negative_accuracy']:.4f}")
    print(f"  Mean positive score: {summary['global_positive_mean_score']:.4f}")
    print(f"  Mean negative score: {summary['global_negative_mean_score']:.4f}")
    print(f"  Thr@0.5 bal acc:     {summary['threshold_0_5_balanced_accuracy']:.4f}")
    print(f"  Best threshold:      {summary['best_threshold']:.3f}")
    print(f"  Best bal acc:        {summary['best_threshold_balanced_accuracy']:.4f}")
    print(f"  Best pos acc:        {summary['best_threshold_positive_accuracy']:.4f}")
    print(f"  Best neg acc:        {summary['best_threshold_negative_accuracy']:.4f}")
    print(f"  Best thr CM:         {summary['best_threshold_confusion_matrix']}")
    print(f"  Unsup method:        {summary['unsupervised_method']}")
    print(f"  Unsup centers:       {summary['unsupervised_centers']}")
    print(f"  Top-1 cell accuracy: {summary['cell_top1_accuracy']:.4f}")
    print(f"  Avg true score:      {summary['cell_avg_true_score']:.4f}")
    print(f"  Avg top score:       {summary['cell_avg_top_score']:.4f}")
    print(f"  Avg true-top margin: {summary['cell_avg_true_minus_top_margin']:.4f}")
    print(f"  Avg true-2nd margin: {summary['cell_avg_true_minus_second_margin']:.4f}")
    if args.save_plot:
        print(f"  Plot guardado:       {plot_path}")
    print(f"\nGuardado en:\n  {summary_path}\n  {cell_path}\n  {pair_path}\n  {sweep_path}")
    if args.save_plot:
        print(f"  {plot_path}")


if __name__ == '__main__':
    main()

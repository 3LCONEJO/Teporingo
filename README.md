# Teporingo-Demultiplexing

Teporingo-Demultiplexing is a research-oriented framework for genotype-aware single-cell demultiplexing using paired representations of VCF and BAM-derived information.

The project is designed to support:

- multi-batch experiments
- genotype-guided donor assignment
- siamese or embedding-based neural architectures
- downstream clustering and similarity analysis

Because apparently the universe decided that single-cell pipelines should involve fifteen file formats, three languages, and emotional damage.

---

# Project Structure

```text
Teporingo-Demultiplexing/
├── configs/        # YAML configuration files
├── examples/       # Example workflows
├── models/         # Trained models and outputs
├── src/
│   └── teporingo_demultiplexing/
├── tests/
└── README.md
```

---

# Configuration

The pipeline is controlled through a single YAML file.

Example:

```yaml
pipeline:
  min_maf: 0.01
  max_maf: 0.10
  use_gt: True
  output_prefix: "fastdemux-testing"

  vcf: "/path/to/combined.vcf.gz"

  assignments: "/path/to/assignment.tsv"

  bams:
    batch_a: "/path/to/sample.bam"

training:
  output_dir: "models/fd_smoke_A_siamese"
  batch_label: "batch_a"

  overwrite: false

  epochs: 5
  batch_size: 64
  learning_rate: 0.001

  optimizer: "adamw"

  embedding_dim: 512
  network: "siamese"

  patience: 10
  num_workers: 4
  seed: 30131211

  negative_ratio: 3.0
  pair_mode: "sampled"

  skip_dataloader_smoke_test: true
  hard_negatives_k: 10

evaluate:
  model: "models/fd_smoke_A_siamese/best_model.pt"

  vcf_matrix: "fastdemux-testing_vcf_batch_a_matrix.pt"
  bam_matrix: "fastdemux-testing_bam_batch_a_matrix.npz"
  metadata: "fastdemux-testing_metadata_batch_a.pt"

  output_dir: "models/fd_smoke_A_siamese"

  batch_size: 64
  num_workers: 4

  analysis:
    score_csv: "models/fd_smoke_A_siamese/score_pairs.csv"
    output_dir: "models/fd_smoke_A_siamese"
    prefix: "analyze_output"

    unsupervised_method: "kmeans"
```

---

# YAML Sections

## `pipeline`

Defines the preprocessing and matrix generation stage.

| Parameter | Description |
|---|---|
| `min_maf` | Minimum minor allele frequency threshold |
| `max_maf` | Maximum minor allele frequency threshold |
| `use_gt` | Whether to use GT fields from the VCF |
| `output_prefix` | Prefix used for generated matrices and metadata |
| `vcf` | Path to merged VCF file |
| `assignments` | TSV file mapping barcodes to donor identities |
| `bams` | Dictionary of BAM files keyed by batch name |

### Notes

- A merged VCF shared across batches is strongly recommended.
- BAM files may come from independent runs or pseudo-multiplexed datasets.
- Barcode collisions can occur when merging BAMs from separate experiments. Prefixing barcodes by batch is recommended unless Cell Ranger already handled this.

Because naturally every sequencing platform believes it alone deserves to define barcode conventions.

---

## `training`

Defines neural network training parameters.

| Parameter | Description |
|---|---|
| `output_dir` | Directory where checkpoints and outputs are stored |
| `batch_label` | Which BAM batch to train on |
| `epochs` | Maximum training epochs |
| `batch_size` | Training batch size |
| `learning_rate` | Optimizer learning rate |
| `optimizer` | Optimizer type (`sgd`, `adamw`, etc.) |
| `embedding_dim` | Latent embedding size |
| `network` | Model architecture |
| `patience` | Early stopping patience |
| `num_workers` | DataLoader worker count |
| `seed` | Random seed |
| `negative_ratio` | Ratio of negative training pairs |
| `pair_mode` | Pair sampling strategy |
| `hard_negatives_k` | Number of hard negatives mined per iteration |

### Pair Sampling

Current supported modes:

| Mode | Description |
|---|---|
| `sampled` | Randomly sampled positive/negative pairs |
| `full` | Uses all available pairs (may be expensive) |

---

## `evaluate`

Defines evaluation and downstream analysis settings.

| Parameter | Description |
|---|---|
| `model` | Path to trained checkpoint |
| `vcf_matrix` | Precomputed VCF embedding matrix |
| `bam_matrix` | Precomputed BAM feature matrix |
| `metadata` | Metadata generated during preprocessing |
| `output_dir` | Evaluation output directory |
| `batch_size` | Evaluation batch size |
| `num_workers` | Number of DataLoader workers |

---

# Downstream Analysis

The `analysis` block controls similarity scoring and unsupervised clustering.

| Parameter | Description |
|---|---|
| `score_csv` | Output similarity score table |
| `output_dir` | Analysis output directory |
| `prefix` | Prefix for generated analysis files |
| `unsupervised_method` | Clustering method (`kmeans`, etc.) |

---

# Recommended Workflow

```text
VCF + BAMs
     ↓
Matrix generation
     ↓
Pair construction
     ↓
Neural embedding training
     ↓
Similarity scoring
     ↓
Clustering / donor recovery
```

Tiny little arrows guiding fragile biological truth through a wasteland of sparse matrices and GPU memory errors. Science.

---

# Current Status

This repository is under active development.

Implemented:

- YAML-driven configuration
- Matrix generation scaffold
- Siamese training scaffold
- Evaluation pipeline structure

Planned:

- Contrastive learning improvements
- Better hard negative mining
- Cross-batch generalization
- Probabilistic donor assignment
- Benchmarking against Demuxlet and Vireo



"""Build VCF and BAM matrices for genotype-to-cell matching.

Example inputs relative to the project root:
- Example VCF: Proyecto/data/demuxlet/b1.b2.b3.merged_32.eagle.hrc.imputed.autosomes.dose.mac1.exon.recode.vcf.gz
- Example BAM: Proyecto/data/demuxlet/A.merged.bam.1 (make sure the corresponding .bai index exists)
- Truth CSV: Proyecto/data/demuxlet/jy-a_truth.csv (columns: BARCODE, BEST, ...)
- Assignments TSV: optional input for `pipeline_completo.py` with BARCODE and DONOR columns.

Quick use:
from Proyecto.data.create_X_vcf_matrix import construir_matriz_vcf, construir_matriz_bam
X_vcf, donors, snps = construir_matriz_vcf(
    '<vcf_path>',
    donantes_seleccionados=None,
    usar_ds=True,
    min_maf=0.05,
    max_maf=0.50,
)
X_bam, barcodes, snps_filtrados = construir_matriz_bam('<bam_path>', panel_snps=snps)
"""

import pysam
from scipy.sparse import lil_matrix, csr_matrix
import allel
import numpy as np
import torch
import math


def calcular_maf_desde_genotipos(X_vcf):
    """Compute MAF per SNP from a genotype/dosage matrix.

    Assumes rows = donors and columns = SNPs.
    Negative values are treated as missing.
    """

    X = X_vcf.detach().cpu().numpy() if isinstance(X_vcf, torch.Tensor) else np.asarray(X_vcf)
    X = X.astype(np.float32, copy=False)

    mafs = np.zeros(X.shape[1], dtype=np.float32)

    for snp_idx in range(X.shape[1]):
        genos = X[:, snp_idx]
        validos = genos >= 0
        if not np.any(validos):
            mafs[snp_idx] = 0.0
            continue

        af = genos[validos].mean() / 2.0
        mafs[snp_idx] = min(af, 1.0 - af)

    return mafs


def _coerce_numeric_array(values):
    """Convert allel outputs to a float array, even when they arrive as object dtype."""

    array = np.asarray(values)
    if array.dtype != object:
        return array.astype(np.float32, copy=False)

    try:
        return array.astype(np.float32, copy=False)
    except (TypeError, ValueError):
        return np.asarray(
            [np.nan if value is None else float(value) for value in array.ravel()],
            dtype=np.float32,
        ).reshape(array.shape)

def construir_matriz_vcf(
    ruta_vcf,
    donantes_seleccionados=None,
    usar_ds=True,
    min_maf=None,
    max_maf=None
):

    campos = [
        'samples',
        'variants/CHROM',
        'variants/POS',
        'variants/REF',
        'variants/ALT'
    ]

    if usar_ds:
        campos.append('calldata/DS')
    else:
        campos.append('calldata/GT')

    vcf = allel.read_vcf(
        ruta_vcf,
        fields=campos
    )

    muestras = np.array(vcf['samples'])

    # =========================
    # FILTER DONORS
    # =========================

    if donantes_seleccionados is not None:

        mascara_muestras = np.isin(
            muestras,
            donantes_seleccionados
        )

        muestras = muestras[mascara_muestras]

    chroms = vcf['variants/CHROM']
    poss   = vcf['variants/POS']
    refs   = vcf['variants/REF']
    alts   = vcf['variants/ALT'][:, 0]

    # =========================
    # DOSAGE
    # =========================

    if usar_ds:

        ds = vcf['calldata/DS']  # [F, N]

        # filtrar individuos
        if donantes_seleccionados is not None:
            ds = ds[:, mascara_muestras]

        ds = _coerce_numeric_array(ds)
        ds = np.where(np.isnan(ds), -1.0, ds)

        # [F,N] -> [N,F]
        X = ds.T

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        )

    else:

        gt = vcf['calldata/GT']

        if donantes_seleccionados is not None:
            gt = gt[:, mascara_muestras, :]

        X = gt.sum(axis=2).T

        X = np.where(X < 0, -1, X)

        X_tensor = torch.tensor(
            X,
            dtype=torch.int8
        )

    variantes = list(zip(chroms, poss, refs, alts))

    if min_maf is not None or max_maf is not None:
        mafs = calcular_maf_desde_genotipos(X_tensor)

        if min_maf is None:
            min_maf = 0.0
        if max_maf is None:
            max_maf = 0.5

        mascara_maf = (mafs >= min_maf) & (mafs <= max_maf)

        X_tensor = X_tensor[:, mascara_maf]
        variantes = [var for var, keep in zip(variantes, mascara_maf) if keep]

        print(
            f"VCF MAF filter applied: kept {len(variantes)}/{len(mafs)} SNPs "
            f"(range {min_maf:.3f}-{max_maf:.3f})"
        )

    return X_tensor, muestras.tolist(), variantes



def construir_matriz_bam(
    ruta_bam,
    panel_snps,
    barcode_tag='CB',
    umi_tag='UB',
    min_mapq=20,
    min_baseq=20
):
    """Build X_BAM from a scRNA-seq BAM file.

    Args:
        ruta_bam: Path to the aligned BAM (must have a .bai index)
        panel_snps: List of tuples (chrom, pos, ref, alt)
        barcode_tag: BAM tag used to identify cells
        umi_tag: BAM tag used for UMI deduplication
        min_mapq: Minimum mapping quality
        min_baseq: Minimum base quality

    Returns:
        X_BAM_sparse: scipy.sparse.csr_matrix [M, 2F]
        barcodes: List of unique barcodes
        panel_snps: Filtered SNP list used to build the matrix
    """
    
    F = len(panel_snps)
    
    # Step 1: discover all unique barcodes.
    print("Step 1: Discovering barcodes...")
    barcodes_set = set()

    with pysam.AlignmentFile(ruta_bam, "rb") as bam:
        for chrom, pos, _, _ in panel_snps[:100]:  # Sample for speed.
            for read in bam.fetch(chrom, pos - 1, pos):
                if read.has_tag(barcode_tag):
                    barcodes_set.add(read.get_tag(barcode_tag))

        barcodes = sorted(list(barcodes_set))
        barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
        M = len(barcodes)

        print(f"  Found {M} cells")

        # Step 2: build the sparse matrix.
        print("Step 2: Counting alleles...")

        # Use lil_matrix for efficient incremental construction.
        X_bam = lil_matrix((M, 2 * F), dtype=np.int32)

        # Cache UMIs for deduplication.
        from collections import defaultdict
        umis_vistos = defaultdict(set)  # (barcode, snp_idx, allele) -> set(UMIs)

        for snp_idx, (chrom, pos, ref, alt) in enumerate(panel_snps):
            if snp_idx % 1000 == 0:
                print(f"  Processing SNP {snp_idx}/{F}...")

            for pileupcolumn in bam.pileup(
                chrom,
                pos - 1,
                pos,
                truncate=True,
                stepper='nofilter',
                min_base_quality=min_baseq,
            ):
                if pileupcolumn.pos != pos - 1:
                    continue

                for pileupread in pileupcolumn.pileups:
                    read = pileupread.alignment

                    # Quality filters.
                    if read.mapping_quality < min_mapq:
                        continue
                    if not read.has_tag(barcode_tag):
                        continue
                    if pileupread.is_del or pileupread.is_refskip:
                        continue

                    # Identify the cell.
                    barcode = read.get_tag(barcode_tag)
                    if barcode not in barcode_to_idx:
                        continue
                    celula_idx = barcode_to_idx[barcode]

                    # UMI deduplication.
                    if read.has_tag(umi_tag):
                        umi = read.get_tag(umi_tag)
                        base = pileupread.alignment.query_sequence[
                            pileupread.query_position
                        ]

                        key = (barcode, snp_idx, base)
                        if umi in umis_vistos[key]:
                            continue  # Duplicate UMI.
                        umis_vistos[key].add(umi)

                    # Classify allele.
                    base = pileupread.alignment.query_sequence[
                        pileupread.query_position
                    ]

                    if base == ref:
                        col_idx = snp_idx  # REF channel
                        X_bam[celula_idx, col_idx] += 1
                    elif base == alt:
                        col_idx = F + snp_idx  # ALT channel
                        X_bam[celula_idx, col_idx] += 1
    
    # Step 3: convert to CSR and apply log1p.
    print("Step 3: Applying log transformation...")
    X_bam_csr = X_bam.tocsr()
    X_bam_csr.data = np.log1p(X_bam_csr.data)
    
    return X_bam_csr, barcodes, panel_snps


def scipy_sparse_to_torch_sparse(scipy_csr):
    """
    Convierte scipy.sparse.csr_matrix a torch.sparse_coo_tensor.
    """
    scipy_coo = scipy_csr.tocoo()
    
    indices = torch.LongTensor(
        np.vstack((scipy_coo.row, scipy_coo.col))
    )
    values = torch.FloatTensor(scipy_coo.data)
    shape = torch.Size(scipy_coo.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape)

from torch.utils.data import Dataset
import random

class GenotypeMatchingDataset(Dataset):
    """Dataset for training a genotype-to-cell matching model.

    Returns triples (cell, genotype, label):
    - cell: [2F] vector of log-transformed counts
    - genotype: [F] vector of genotypes {0, 1, 2}
    - label: 1.0 for a correct match, 0.0 otherwise
    """
    
    def __init__(
        self,
        X_vcf,           # torch.Tensor [N, F]
        X_bam_sparse,    # scipy.sparse.csr_matrix [M, 2F]
        cell_to_donor,   # dict: barcode -> donor_name
        barcodes,        # list: orden de filas en X_bam
        donors,          # list: orden de filas en X_vcf
        negative_ratio=1.0,  # Negative pair ratio
        pair_mode='sampled',  # 'sampled' or 'exhaustive'
        return_pair_info=False,
        hard_negatives_k=0,  # If >0, precompute top-k hard negative donors per cell
    ):
        self.X_vcf = X_vcf
        self.X_bam = X_bam_sparse
        self.barcodes = barcodes
        self.donors = donors
        self.negative_ratio = negative_ratio
        self.pair_mode = pair_mode
        self.return_pair_info = return_pair_info
        self.hard_negatives_k = hard_negatives_k
        self.hard_negative_candidates = []
        
        # Map names to indices.
        self.donor_to_idx = {d: i for i, d in enumerate(donors)}
        
        # Build cell-to-donor assignments.
        self.cell_assignments = []
        for i, barcode in enumerate(barcodes):
            if barcode in cell_to_donor:
                donor_name = cell_to_donor[barcode]
                if donor_name in self.donor_to_idx:
                    donor_idx = self.donor_to_idx[donor_name]
                    self.cell_assignments.append((i, donor_idx))
        
        self.N_cells_assigned = len(self.cell_assignments)

        if self.pair_mode not in {'sampled', 'exhaustive'}:
            raise ValueError("pair_mode must be 'sampled' or 'exhaustive'")

        self.N_donors = len(self.donors)
        
        # Precompute dataset size.
        if self.pair_mode == 'exhaustive':
            self.samples_per_cell = self.N_donors
            self.total_samples = self.N_cells_assigned * self.N_donors
        else:
            # For each cell: 1 positive + at least one negative sample.
            # Treat values below 1 as proportions and round up so we never
            # collapse the dataset into positives only.
            negatives_per_cell = max(1, int(math.ceil(float(negative_ratio))))
            self.samples_per_cell = 1 + negatives_per_cell
            self.total_samples = self.N_cells_assigned * self.samples_per_cell

        # Precompute hard negatives per assigned cell if requested
        if self.hard_negatives_k and self.N_cells_assigned > 0:
            try:
                # Prepare donor-expanded vectors matching the BAM channel layout [REF... | ALT...]
                # X_vcf: torch tensor [N_donors, F]
                Xv = self.X_vcf.detach().cpu().numpy() if hasattr(self.X_vcf, 'detach') else np.asarray(self.X_vcf)
            except Exception:
                Xv = np.asarray(self.X_vcf)

            # Handle missing values (<0) by treating them as 0 alt fraction.
            Xv = Xv.astype(np.float32, copy=False)
            Xv[Xv < 0] = 0.0

            alt_frac = Xv / 2.0  # in [0,1]
            ref_frac = 1.0 - alt_frac

            donor_expanded = np.concatenate([ref_frac, alt_frac], axis=1)  # [N_donors, 2F]
            donor_norms = np.linalg.norm(donor_expanded, axis=1) + 1e-8

            # For each assigned cell, compute cosine similarity to all donors and keep top-k
            self.hard_negative_candidates = []
            for (cell_row, true_donor_idx) in self.cell_assignments:
                cell_vec = np.asarray(self.X_bam[cell_row].toarray().squeeze(), dtype=np.float32)
                cell_norm = np.linalg.norm(cell_vec) + 1e-8

                if cell_norm == 0.0:
                    # No signal — fallback to random negatives
                    self.hard_negative_candidates.append([])
                    continue

                sims = donor_expanded.dot(cell_vec) / (donor_norms * cell_norm)

                # Exclude the true donor.
                sims[true_donor_idx] = -np.inf

                # Get the top-k indices.
                k = min(int(self.hard_negatives_k), sims.shape[0] - 1)
                if k <= 0:
                    self.hard_negative_candidates.append([])
                    continue

                topk_idx = np.argpartition(-sims, k - 1)[:k]
                # Sort the top-k indices by descending similarity.
                topk_idx = topk_idx[np.argsort(-sims[topk_idx])]
                self.hard_negative_candidates.append(topk_idx.tolist())
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """Return a (cell, genotype, label) pair."""
        if self.pair_mode == 'exhaustive':
            cell_idx_in_assigned = idx // self.N_donors
            donor_idx = idx % self.N_donors
            is_positive = False
        else:
            # Determine which cell this is and whether it is positive or negative.
            cell_idx_in_assigned = idx // self.samples_per_cell
            is_positive = (idx % self.samples_per_cell) == 0
        
        # Retrieve the real cell and its true donor.
        cell_idx, true_donor_idx = self.cell_assignments[cell_idx_in_assigned]
        
        # Extract the cell vector (convert sparse row to dense).
        cell_vector = torch.FloatTensor(
            self.X_bam[cell_idx].toarray().squeeze()
        )  # [2F]
        
        if self.pair_mode == 'exhaustive':
            label = float(donor_idx == true_donor_idx)
        elif is_positive:
            # Positive pair: use the correct donor.
            donor_idx = true_donor_idx
            label = 1.0
        else:
            # Negative pair: sample from hard negatives when available.
            label = 0.0
            donor_idx = None
            if self.hard_negative_candidates and len(self.hard_negative_candidates) > cell_idx_in_assigned:
                candidates = self.hard_negative_candidates[cell_idx_in_assigned]
                if candidates:
                    donor_idx = random.choice(candidates)

            if donor_idx is None:
                # Fallback a muestreo aleatorio uniforme
                N_donors = len(self.donors)
                donor_idx = random.randint(0, N_donors - 1)
                # Asegurar que es diferente
                while donor_idx == true_donor_idx:
                    donor_idx = random.randint(0, N_donors - 1)
        
        # Extraer vector de genotipo
        genotype_vector = self.X_vcf[donor_idx].float()  # [F]

        if self.return_pair_info:
            return (
                cell_vector,
                genotype_vector,
                torch.tensor(label),
                cell_idx,
                donor_idx,
                true_donor_idx,
            )

        return cell_vector, genotype_vector, torch.tensor(label)
    
    def get_positive_pair(self, cell_idx):
        """Utility to get a specific positive pair for validation."""
        if cell_idx >= self.N_cells_assigned:
            raise IndexError("Cell index out of range")
        
        actual_cell_idx, donor_idx = self.cell_assignments[cell_idx]
        
        cell_vector = torch.FloatTensor(
            self.X_bam[actual_cell_idx].toarray().squeeze()
        )
        genotype_vector = self.X_vcf[donor_idx].float()
        
        return cell_vector, genotype_vector, donor_idx

"""Siamese network for genotype-to-cell matching."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CellEncoder(nn.Module):
    """
    Encoder for cell vectors (X_BAM).

    Input: [batch, 2F] where F = number of SNPs.
    Output: [batch, embedding_dim]
    """
    
    def __init__(
        self,
        input_dim,      # 2F (REF + ALT channels)
        embedding_dim=512,
        hidden_dims=[1024, 512],
        dropout=0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final embedding layer.
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: [batch, 2F] tensor of log-transformed counts

        Returns:
            embeddings: [batch, embedding_dim]
        """
        return self.encoder(x)


class GenotypeEncoder(nn.Module):
    """
    Encoder for genotype vectors (X_VCF).

    Input: [batch, F] dosage or genotype values
    Output: [batch, embedding_dim]
    """
    
    def __init__(
        self,
        input_dim,      # F (number of SNPs)
        embedding_dim=512,
        hidden_dims=[512, 512],
        dropout=0.2
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final embedding layer.
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: [batch, F] tensor of genotypes/dosage values

        Returns:
            embeddings: [batch, embedding_dim]
        """
        return self.encoder(x)


class SiameseNetwork(nn.Module):
    """
    Full Siamese network for matching.
    """
    
    def __init__(
        self,
        n_snps,                 # Number of SNPs in the panel
        embedding_dim=512,
        cell_hidden=[1024, 512],
        geno_hidden=[512, 512],
        dropout_cell=0.3,
        dropout_geno=0.2,
        similarity='cosine'     # 'cosine' or 'euclidean'
    ):
        super().__init__()
        
        self.cell_encoder = CellEncoder(
            input_dim=2 * n_snps,
            embedding_dim=embedding_dim,
            hidden_dims=cell_hidden,
            dropout=dropout_cell
        )
        
        self.genotype_encoder = GenotypeEncoder(
            input_dim=n_snps,
            embedding_dim=embedding_dim,
            hidden_dims=geno_hidden,
            dropout=dropout_geno
        )
        
        self.similarity_metric = similarity
        
    def forward(self, cell_batch, geno_batch):
        """
        Args:
            cell_batch: [batch, 2F] cell vectors
            geno_batch: [batch, F] genotype vectors

        Returns:
            similarity_scores: [batch] scores between 0 and 1
        """
        # Encode both branches.
        cell_emb = self.cell_encoder(cell_batch)      # [batch, 512]
        geno_emb = self.genotype_encoder(geno_batch)  # [batch, 512]
        
        # Normalize embeddings.
        cell_emb = F.normalize(cell_emb, p=2, dim=1)
        geno_emb = F.normalize(geno_emb, p=2, dim=1)
        
        # Compute similarity.
        if self.similarity_metric == 'cosine':
            # Cosine similarity: dot product of normalized vectors.
            similarity = F.cosine_similarity(cell_emb, geno_emb)
        
        elif self.similarity_metric == 'euclidean':
            # Convert Euclidean distance into a similarity score.
            dist = torch.norm(cell_emb - geno_emb, p=2, dim=1)  # [batch]
            # Invert the distance into a similarity score.
            similarity = 1 / (1 + dist)
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return similarity


class ShallowSiameseNetwork(nn.Module):
    """
    Lightweight variant for small datasets or quick baselines.
    """

    def __init__(self, n_snps, embedding_dim=256, similarity='cosine'):
        super().__init__()

        self.cell_encoder = nn.Sequential(
            nn.Linear(2 * n_snps, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
        )

        self.genotype_encoder = nn.Sequential(
            nn.Linear(n_snps, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

        self.similarity_metric = similarity

    def forward(self, cell_batch, geno_batch):
        cell_emb = self.cell_encoder(cell_batch)
        geno_emb = self.genotype_encoder(geno_batch)

        cell_emb = F.normalize(cell_emb, p=2, dim=1)
        geno_emb = F.normalize(geno_emb, p=2, dim=1)

        if self.similarity_metric == 'cosine':
            similarity = (cell_emb * geno_emb).sum(dim=1)
            similarity = (similarity + 1) / 2
        elif self.similarity_metric == 'euclidean':
            dist = torch.norm(cell_emb - geno_emb, p=2, dim=1)
            similarity = 1 / (1 + dist)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        return similarity
    
    def get_embeddings(self, cell_batch=None, geno_batch=None):
        """Get embeddings without computing similarity.

        Useful for visualization and analysis.
        """
        results = {}
        
        if cell_batch is not None:
            cell_emb = self.cell_encoder(cell_batch)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)
            results['cell'] = cell_emb
        
        if geno_batch is not None:
            geno_emb = self.genotype_encoder(geno_batch)
            geno_emb = F.normalize(geno_emb, p=2, dim=1)
            results['geno'] = geno_emb
        
        return results


# ════════════════════════════════════════════════════════════
# VARIANT: Network with a Fusion Layer (Alternative)
# ════════════════════════════════════════════════════════════

class SiameseWithFusion(nn.Module):
    """
    Variant that concatenates embeddings and passes them through an MLP.

    It can be more expressive but less interpretable.
    """
    
    def __init__(
        self,
        n_snps,
        embedding_dim=512,
        cell_hidden=[1024, 512],
        geno_hidden=[512, 512],
        dropout_cell=0.3,
        dropout_geno=0.2
    ):
        super().__init__()
        
        self.cell_encoder = CellEncoder(
            input_dim=2 * n_snps,
            embedding_dim=embedding_dim,
            hidden_dims=cell_hidden,
            dropout=dropout_cell
        )
        
        self.genotype_encoder = GenotypeEncoder(
            input_dim=n_snps,
            embedding_dim=embedding_dim,
            hidden_dims=geno_hidden,
            dropout=dropout_geno
        )
        
        # Fusion MLP.
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, cell_batch, geno_batch):
        cell_emb = self.cell_encoder(cell_batch)
        geno_emb = self.genotype_encoder(geno_batch)
        
        # Concatenate embeddings.
        combined = torch.cat([cell_emb, geno_emb], dim=1)  # [batch, 1024]
        
        # Pass through the MLP.
        similarity = self.fusion(combined).squeeze()  # [batch]

        
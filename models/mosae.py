import torch
from torch import nn
#
from .prot_encoder import ProteomicsAutoencoder
from .phosphoprot_encoder import PhosphoproteomicsAutoencoder

class MOSAE(nn.Module):
    def __init__(self, prot_dim: int, phos_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        
        self.prot_ae = ProteomicsAutoencoder(input_dim=prot_dim, dropout_rate=dropout_rate)
        self.phos_ae = PhosphoproteomicsAutoencoder(input_dim=phos_dim, dropout_rate=dropout_rate)
        
        self.fusion_dropout = nn.Dropout(0.5)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, x_prot, x_phos):
        recon_prot, latent_prot = self.prot_ae(x_prot)
        recon_phos, latent_phos = self.phos_ae(x_phos)
        
        # --- DYNAMIC MASKED FUSION ---
        # Check if the patient has data for each modality (sum of absolute values > 0)
        # We use keepdim=True so the mask shape is [batch_size, 1] allowing it to broadcast
        prot_present = (x_prot.abs().sum(dim=1, keepdim=True) > 0).float()
        phos_present = (x_phos.abs().sum(dim=1, keepdim=True) > 0).float()
        
        # Calculate how many modalities this patient actually has (1 or 2)
        # clamp to a minimum of 1.0 to prevent division by zero
        modality_count = torch.clamp(prot_present + phos_present, min=1.0)
        
        # If a modality is missing, its vector is multiplied by 0 and ignored.
        # If both are present, it averages them (divides by 2).
        fused_latent = (latent_prot * prot_present + latent_phos * phos_present) / modality_count
        # -----------------------------
        
        dropped_latent = self.fusion_dropout(fused_latent)
        logits = self.classifier(dropped_latent)
        
        return recon_prot, recon_phos, fused_latent, logits

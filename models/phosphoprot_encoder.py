import torch.nn as nn


class PhosphoproteomicsAutoencoder(nn.Module):
    def __init__(self, input_dim: int, dropout_rate):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256) # 256-dim bottleneck
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, input_dim) # Reconstructs phospho features
        )

        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     
        #     nn.Linear(512, 256)
        # )
        # 
        # self.decoder = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     
        #     nn.Linear(512, input_dim) # Reconstructs phospho features
        # )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

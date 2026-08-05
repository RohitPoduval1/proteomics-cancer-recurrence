# import cptac
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from data import get_data
from models.mosae import MOSAE  # Updated model import
from train import train
from eval import evaluate_zero_shot

from preprocessing import build_pan_cancer_dataloaders, process_omics


from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

cancer_types = ['brca', 'ccrcc', 'coad', 'gbm', 'hnscc', 'lscc', 'luad', 'ov', 'pdac', 'ucec']
# cancer_types: list[str] = list(cptac.get_cancer_info().keys())
# cancer_types.remove("pda")   # does not exist
cancer_types.remove("pdac")  # testing
print(f"All Cancer Types Used: {cancer_types}")

cancer_data: dict[str, dict] = {}
for cancer_type in cancer_types:
    cancer_data[cancer_type] = get_data(cancer_type)

top_k_prot = 3000
top_k_phos = 2000

(
    train_loader, 
    test_loader, 
    top_prot_features, 
    top_phos_features, 
    pos_weight_val
) = build_pan_cancer_dataloaders(
    cancer_data=cancer_data,
    top_k_prot=top_k_prot,
    top_k_phos=top_k_phos,
    batch_size=32
)

device = torch.device("mps" if torch.mps.is_available() else "cpu")
pos_weight = torch.tensor([pos_weight_val]).to(device)

# 2. Initialize the MOSAE multi-omics model
model = MOSAE(
    prot_dim=len(top_prot_features),
    phos_dim=len(top_phos_features),
    dropout_rate=0.3
)

epochs = 130
model = train(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    pos_weight=pos_weight,
    epochs=epochs,
    early_stopping_patience=12,
    noise_factor=0.15,
    alpha=0.5,
)
print("MOSAE Model training complete!")
print("\nPreparing PDAC for Zero-Shot Evaluation...")
pdac_data = get_data(study_name="pdac")

# Extract both modalities for PDAC
pdac_prot = pdac_data['proteomics']
pdac_phos = pdac_data['phosphoproteomics']
pdac_has_recurrence = pdac_data['has_recurrence']

# 1. Sanitize raw PDAC data against NaNs and Infs
# pdac_prot = pdac_prot.replace([np.inf, -np.inf], np.nan).fillna(0.0)
# pdac_phos = pdac_phos.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Drop zero-variance columns to prevent scaler division errors
# pdac_prot = pdac_prot.loc[:, pdac_prot.var() > 1e-8]
# pdac_phos = pdac_phos.loc[:, pdac_phos.var() > 1e-8]

pdac_prot = process_omics(pdac_prot)
pdac_phos = process_omics(pdac_phos)

labeled_patients = pdac_has_recurrence.dropna().index
labeled_prot = pdac_prot.index.intersection(labeled_patients)
labeled_phos = pdac_phos.index.intersection(labeled_patients)

# Union to keep patients with missing modalities
labeled_patients = labeled_prot.union(labeled_phos)

pdac_prot_valid = pdac_prot.reindex(labeled_patients)
pdac_phos_valid = pdac_phos.reindex(labeled_patients)
pdac_y = pdac_has_recurrence.loc[labeled_patients].astype(float)

# 3. INDEPENDENT SAFE SCALING (Ignores rows of completely missing modalities)
def scale_pdac_ignoring_nans(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    valid_mask = df.notna().any(axis=1)  # Find rows that actually have data
    scaled_df = df.copy()
    
    if valid_mask.sum() > 0:
        scaled_df.loc[valid_mask, :] = scaler.fit_transform(df.loc[valid_mask, :])
    return scaled_df

pdac_prot_scaled = scale_pdac_ignoring_nans(pdac_prot_valid)
pdac_phos_scaled = scale_pdac_ignoring_nans(pdac_phos_valid)

# 4. Align and fill missing with 0.0 (This safely zeros out missing phosphoproteomics)
pdac_prot_aligned = pdac_prot_scaled.reindex(columns=top_prot_features).fillna(0.0)
pdac_phos_aligned = pdac_phos_scaled.reindex(columns=top_phos_features).fillna(0.0)

# 5. Convert to Tensors
pdac_prot_tensor = torch.tensor(pdac_prot_aligned.values, dtype=torch.float32)
pdac_phos_tensor = torch.tensor(pdac_phos_aligned.values, dtype=torch.float32)
pdac_y_tensor = torch.tensor(pdac_y.values, dtype=torch.float32).view(-1, 1)

# Evaluate using multi-omics inputs
evaluate_zero_shot(
    model=model,
    X_prot_tensor=pdac_prot_tensor,
    X_phos_tensor=pdac_phos_tensor,
    y_tensor=pdac_y_tensor,
    device=device,
    study_name="pdac",
)


import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import pandas as pd
import numpy as np

# 1. Wrap the model so it only returns the classification logits
class MOSAEClassifierWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        
    def forward(self, prot, phos):
        outputs = self.model(prot, phos)
        # Grab the logits (index 2 as established in your earlier code)
        logits = outputs[-1] if isinstance(outputs, tuple) else outputs
        return logits

# Initialize the wrapper and put it in eval mode
model.eval()
wrapper = MOSAEClassifierWrapper(model).to(device)


# 2. Initialize Integrated Gradients
ig = IntegratedGradients(wrapper)

# We want to find what drives RECURRENCE. Let's isolate the PDAC patients who actually recurred.
recurrence_mask = (pdac_y_tensor.cpu().numpy() == 1).flatten()
prot_recurrence = pdac_prot_tensor[recurrence_mask].to(device)
phos_recurrence = pdac_phos_tensor[recurrence_mask].to(device)

print("Calculating Integrated Gradients... (This may take a minute)")
# 3. Calculate attributions
attr_prot, attr_phos = ig.attribute(
    inputs=(prot_recurrence, phos_recurrence),
    baselines=(torch.zeros_like(prot_recurrence), torch.zeros_like(phos_recurrence)),
    target=0, # Target node 0 (since it's a binary output layer)
)

# 4. Average the attributions across all recurrence patients
# We use absolute mean because large negative or positive gradients both indicate high importance
mean_attr_prot = torch.mean(torch.abs(attr_prot), dim=0).cpu().detach().numpy()
mean_attr_phos = torch.mean(torch.abs(attr_phos), dim=0).cpu().detach().numpy()


# 5. Create DataFrames to rank the features
# Assuming `top_prot_features` and `top_phos_features` are your column name lists
prot_importance = pd.DataFrame({
    'Feature': top_prot_features,
    'Modality': 'Proteomics',
    'Importance': mean_attr_prot
}).sort_values(by='Importance', ascending=False)

phos_importance = pd.DataFrame({
    'Feature': top_phos_features,
    'Modality': 'Phosphoproteomics',
    'Importance': mean_attr_phos
}).sort_values(by='Importance', ascending=False)

# Combine and get the Top 50 overall drivers
top_50_genes = pd.concat([prot_importance, phos_importance]).sort_values(by='Importance', ascending=False).head(50)

print("\n--- Top 10 Drivers of PDAC Recurrence ---")
print(top_50_genes.head(10))

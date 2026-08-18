import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _get_universal_feature_space(cancer_data: dict[str, dict], modality: str) -> list[str]:
    """Finds the union of all features for a given modality across all provided cancer datasets."""
    universal_features = set()
    for data in cancer_data.values():
        if modality in data:
            universal_features.update(data[modality].columns)
    return sorted(list(universal_features))


def process_omics(X: pd.DataFrame) -> pd.DataFrame:
    X_processed = (
        X
        # replace nan & inf with 0
        .replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # drop columns with 0 variance
        .loc[:, X.var() > 1e-8]
    )
    return X_processed


def _process_single_cohort_multiomics(
    X_prot: pd.DataFrame, 
    X_phos: pd.DataFrame, 
    y: pd.Series, 
    universal_proteins: list[str],
    universal_phosphos: list[str]
) -> tuple:
    """Handles loose patient intersection, synced splitting, safe scaling, and feature padding."""
    
    X_prot = process_omics(X_prot)
    X_phos = process_omics(X_phos)

    # X_prot = X_prot.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # X_phos = X_phos.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Drop columns with zero variance
    # X_prot = X_prot.loc[:, X_prot.var() > 1e-8]
    # X_phos = X_phos.loc[:, X_phos.var() > 1e-8]

    labeled_patients = y.dropna().index
    valid_prot = X_prot.index.intersection(labeled_patients)
    valid_phos = X_phos.index.intersection(labeled_patients)
    valid_patients = valid_prot.intersection(valid_phos)
    
    # Reindex to the master list
    # If a patient is missing a modality, they get a row of NaNs.
    X_prot = X_prot.reindex(valid_patients)
    X_phos = X_phos.reindex(valid_patients)
    y = y.loc[valid_patients].astype(float)
    
    class_counts = y.value_counts()
    
    # 2. SYNCED SPLITTING
    if len(class_counts) < 2 or class_counts.min() < 2:
        is_pos = (y == 1.0)
        is_neg = (y == 0.0)
        
        X_prot_pos, X_phos_pos, y_pos = X_prot[is_pos], X_phos[is_pos], y[is_pos]
        X_prot_neg, X_phos_neg, y_neg = X_prot[is_neg], X_phos[is_neg], y[is_neg]
        
        X_prot_neg_train, X_prot_neg_test, X_phos_neg_train, X_phos_neg_test, y_neg_train, y_neg_test = train_test_split(
            X_prot_neg, X_phos_neg, y_neg, test_size=0.2, random_state=42
        )
        
        X_prot_train = pd.concat([X_prot_neg_train, X_prot_pos])
        X_phos_train = pd.concat([X_phos_neg_train, X_phos_pos])
        y_train = pd.concat([y_neg_train, y_pos])
        
        X_prot_test, X_phos_test, y_test = X_prot_neg_test, X_phos_neg_test, y_neg_test
    else:
        X_prot_train, X_prot_test, X_phos_train, X_phos_test, y_train, y_test = train_test_split(
            X_prot, X_phos, y, test_size=0.2, random_state=42, stratify=y
        )
        
    # 3. INDEPENDENT SAFE Z-SCORE SCALING (Ignores rows of completely missing modalities)
    def scale_ignoring_nans(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        scaler = StandardScaler()
        
        # Find which patients actually have data for this modality
        train_valid = train_df.notna().any(axis=1)
        test_valid = test_df.notna().any(axis=1)
        
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()
        
        # Scale only the valid rows so missing patients don't skew the standard deviation
        if train_valid.sum() > 0:
            train_scaled.loc[train_valid, :] = scaler.fit_transform(train_df.loc[train_valid, :])
        if test_valid.sum() > 0:
            test_scaled.loc[test_valid, :] = scaler.transform(test_df.loc[test_valid, :])
            
        return train_scaled, test_scaled

    X_prot_train_scaled, X_prot_test_scaled = scale_ignoring_nans(X_prot_train, X_prot_test)
    X_phos_train_scaled, X_phos_test_scaled = scale_ignoring_nans(X_phos_train, X_phos_test)
    
    # Convert NaNs to 0 for masking
    X_prot_train_final = X_prot_train_scaled.reindex(columns=universal_proteins).fillna(0.0)
    X_prot_test_final = X_prot_test_scaled.reindex(columns=universal_proteins).fillna(0.0)
    
    X_phos_train_final = X_phos_train_scaled.reindex(columns=universal_phosphos).fillna(0.0)
    X_phos_test_final = X_phos_test_scaled.reindex(columns=universal_phosphos).fillna(0.0)
    
    return (X_prot_train_final, X_prot_test_final, 
            X_phos_train_final, X_phos_test_final, 
            y_train, y_test)

def _apply_variance_filter(X_train: pd.DataFrame, X_test: pd.DataFrame, top_k: int) -> tuple:
    """Filters dataframes to the top_k most variable features based on the training set."""
    variances = X_train.var()
    highly_variable_features = variances.nlargest(top_k).index
    
    return X_train[highly_variable_features], X_test[highly_variable_features], highly_variable_features

def build_pan_cancer_dataloaders(cancer_data: dict[str, dict], top_k_prot: int = 3000, top_k_phos: int = 3000, batch_size: int = 32):
    """
    Orchestrates the multi-omics preprocessing pipeline.
    
    Returns:
        train_loader: Yields (batch_X_prot, batch_X_phos, batch_y)
        test_loader: Yields (batch_X_prot, batch_X_phos, batch_y)
        top_prot_features: Final proteomics feature list
        top_phos_features: Final phosphoproteomics feature list
        pos_weight_val: Weight for BCE Loss
    """
    universal_proteins = _get_universal_feature_space(cancer_data, 'proteomics')
    universal_phosphos = _get_universal_feature_space(cancer_data, 'phosphoproteomics')
    
    print(f"Universal Pan-Cancer Proteomics: {len(universal_proteins)} features")
    print(f"Universal Pan-Cancer Phosphoproteomics: {len(universal_phosphos)} features")
    
    X_prot_train_list, X_prot_test_list = [], []
    X_phos_train_list, X_phos_test_list = [], []
    y_train_list, y_test_list = [], []
    
    for cancer_type, data in cancer_data.items():
        # Handle dict key variations (has_recurrence vs recurrence)
        y_data = data.get('has_recurrence', data.get('recurrence'))
        
        X_prot_tr, X_prot_te, X_phos_tr, X_phos_te, y_tr, y_te = _process_single_cohort_multiomics(
            data['proteomics'], data['phosphoproteomics'], y_data,
            universal_proteins, universal_phosphos
        )
        
        X_prot_train_list.append(X_prot_tr)
        X_prot_test_list.append(X_prot_te)
        X_phos_train_list.append(X_phos_tr)
        X_phos_test_list.append(X_phos_te)
        y_train_list.append(y_tr)
        y_test_list.append(y_te)

    # Combine into unified pan-cancer datasets
    X_prot_train_pan, X_prot_test_pan = pd.concat(X_prot_train_list), pd.concat(X_prot_test_list)
    X_phos_train_pan, X_phos_test_pan = pd.concat(X_phos_train_list), pd.concat(X_phos_test_list)
    y_train_pan, y_test_pan = pd.concat(y_train_list), pd.concat(y_test_list)
    print(f"TOTAL RECURRENCE: {y_train_pan.sum() + y_test_pan.sum()}")

    print(f"\nOriginal Train Shapes -> Prot: {X_prot_train_pan.shape}, Phos: {X_phos_train_pan.shape}")

    # Apply variance filtering to both independently
    X_prot_train_pan, X_prot_test_pan, top_prot_features = _apply_variance_filter(X_prot_train_pan, X_prot_test_pan, top_k_prot)
    X_phos_train_pan, X_phos_test_pan, top_phos_features = _apply_variance_filter(X_phos_train_pan, X_phos_test_pan, top_k_phos)
    
    print(f"Filtered Train Shapes -> Prot: {X_prot_train_pan.shape}, Phos: {X_phos_train_pan.shape}")

    # Calculate global positive weight
    num_negatives = int((y_train_pan.values == 0).sum())
    num_positives = int((y_train_pan.values == 1).sum())
    raw_pos_weight = num_negatives / max(num_positives, 1)
    pos_weight_val = min(raw_pos_weight, 2)

    # Convert to TensorDatasets (Now holding THREE tensors)
    train_dataset = TensorDataset(
        torch.tensor(X_prot_train_pan.values, dtype=torch.float32),
        torch.tensor(X_phos_train_pan.values, dtype=torch.float32),
        torch.tensor(y_train_pan.values, dtype=torch.float32).view(-1, 1)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_prot_test_pan.values, dtype=torch.float32),
        torch.tensor(X_phos_test_pan.values, dtype=torch.float32),
        torch.tensor(y_test_pan.values, dtype=torch.float32).view(-1, 1)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, top_prot_features, top_phos_features, pos_weight_val

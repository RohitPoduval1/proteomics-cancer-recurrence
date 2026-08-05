import pickle
from pathlib import Path
import pandas as pd

# import cptac


CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_study_object(study_name: str):
    # CPTAC class names are capitalized (e.g., 'Brca', 'Coad')
    formatted_name = study_name.lower().capitalize()
    
    try:
        study_class = getattr(cptac, formatted_name)
        return study_class()

    except AttributeError:
        raise ValueError(f"Invalid study name: '{study_name}'. No corresponding dataset found in cptac.")


def get_data(study_name: str, use_cache: bool = True) -> dict[str, pd.DataFrame | pd.Series]:
    cache_path = CACHE_DIR / f"{study_name}_data.pkl"

    if use_cache and cache_path.exists():
        print(f"Loading {study_name.upper()} data from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Fetching and processing {study_name.upper()} from CPTAC...")
    study = _get_study_object(study_name)
    
    # Proteomics
    proteomics_df = study.get_proteomics(source='umich')
    if isinstance(proteomics_df.columns, pd.MultiIndex):
        proteomics_df.columns = proteomics_df.columns.get_level_values(level='Name')
    proteomics_df = proteomics_df.T.groupby(proteomics_df.columns).mean().T
    print(f"Proteomics Shape: {proteomics_df.shape}")

    # Phosphoproteomics
    phosphoprot_df = study.get_phosphoproteomics(source='bcm')
    if isinstance(phosphoprot_df.columns, pd.MultiIndex):
        phosphoprot_df.columns = phosphoprot_df.columns.get_level_values(level='Database_ID')
    phosphoprot_df = phosphoprot_df.T.groupby(phosphoprot_df.columns).mean().T
    print(f"Phospho Shape: {phosphoprot_df.shape}")

    # Clinical (has recurrence only)
    clinical_df = study.get_clinical(source='mssm')
    recurrence_status = pd.to_numeric(clinical_df['Recurrence status (1, yes; 0, no)'], errors='coerce')
    lost_to_followup = clinical_df['is_this_patient_lost_to_follow-up'].astype(str).str.lower().str.strip() == 'yes'
    valid_mask = (recurrence_status == 1) | ((recurrence_status == 0) & ~lost_to_followup)
    has_recurrence = (
        recurrence_status
        .loc[valid_mask]
        .rename("Has_Recurrence")
    )
    print(f"Recurrence Cases: {has_recurrence.shape}")

    data_dict = {
        'proteomics': proteomics_df,
        'phosphoproteomics': phosphoprot_df,
        'has_recurrence': has_recurrence
    }

    if use_cache:
        with open(cache_path, 'wb') as f:
            pickle.dump(data_dict, f)

    return data_dict

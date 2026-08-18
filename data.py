import pickle
import pandas as pd
from pathlib import Path
from typing import Callable

import cptac

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_study_object(study_name: str):
    """Initializes and returns the CPTAC study object"""
    formatted_name = study_name.lower().capitalize()
    try:
        study_class = getattr(cptac, formatted_name)
        return study_class()
    except AttributeError:
        raise ValueError(f"Invalid study name: '{study_name}'. No corresponding dataset found in cptac.")


def _fetch_or_load_raw(study_obj, modality_name: str, fetch_callable: Callable) -> pd.DataFrame:
    """
    Checks cache for the raw modality data. If missing, fetches via the callable,
    caches the result, and returns it.

    Args:
        study_obj: CPTAC study object (e.g., cptac.Luad())
        modality_name (str): Name of the modality to be used when saving
        fetch_callable (Callable): Function that gets the raw modality data

    Returns:
        Omics modality data
    """
    study_name = type(study_obj).__name__.lower()
    cache_path = CACHE_DIR / f"{study_name}_raw_{modality_name}.pkl"

    if cache_path.exists():
        print(f"Loading raw {modality_name} for {study_name.upper()} from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Fetching raw {modality_name} for {study_name.upper()} from CPTAC...")
    raw_data = fetch_callable()
    
    with open(cache_path, 'wb') as f:
        pickle.dump(raw_data, f)

    return raw_data


def get_raw_proteomics(study_obj) -> pd.DataFrame:
    return _fetch_or_load_raw(
        study_obj,
        'proteomics',
        lambda: study_obj.get_proteomics(source='umich')
    )

def get_raw_phosphoproteomics(study_obj) -> pd.DataFrame:
    return _fetch_or_load_raw(
        study_obj,
        'phosphoproteomics',
        lambda: study_obj.get_phosphoproteomics(source='bcm')
    )

def get_raw_transcriptomics(study_obj) -> pd.DataFrame:
    return _fetch_or_load_raw(
        study_obj,
        'transcriptomics',
        lambda: study_obj.get_transcriptomics(source='bcm')
    )

def get_raw_clinical(study_obj) -> pd.DataFrame:
    return _fetch_or_load_raw(
        study_obj,
        'clinical',
        lambda: study_obj.get_clinical(source='mssm')
    )


def process_raw_proteomics(df: pd.DataFrame) -> pd.DataFrame:
    """Process the raw, untouched CPTAC proteomics data"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(level='Name')
    return df.T.groupby(df.columns).mean().T

def process_raw_phosphoproteomics(df: pd.DataFrame) -> pd.DataFrame:
    """Process the raw, untouched CPTAC phosphoproteomics data"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(level='Database_ID')
    return df.T.groupby(df.columns).mean().T

def process_raw_transcriptomics(df: pd.DataFrame) -> pd.DataFrame:
    """Process the raw, untouched CPTAC transcriptomics data"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(level='Database_ID')
    return df.T.groupby(df.columns).mean().T

def extract_has_recurrence(clinical_df: pd.DataFrame) -> pd.Series:
    """
    Process the raw, untouched CPTAC clinical data to extract binary recurrence
    status for Patient_IDs
    """
    recurrence_status = pd.to_numeric(clinical_df['Recurrence status (1, yes; 0, no)'], errors='coerce')
    is_lost_to_followup = clinical_df['is_this_patient_lost_to_follow-up'].astype(str).str.lower().str.strip() == 'yes'
    valid_mask = (recurrence_status == 1) | ((recurrence_status == 0) & ~is_lost_to_followup)
    return recurrence_status.loc[valid_mask].rename("Has_Recurrence")


MODALITY_REGISTRY = {
    'proteomics': {
        'get_raw': get_raw_proteomics,
        'process': process_raw_proteomics
    },
    'phosphoproteomics': {
        'get_raw': get_raw_phosphoproteomics,
        'process': process_raw_phosphoproteomics
    },
    'transcriptomics': {
        'get_raw': get_raw_transcriptomics,
        'process': process_raw_transcriptomics
    },
    'has_recurrence': {
        'get_raw': get_raw_clinical,
        'process': extract_has_recurrence
    }
}


def get_data(study_name: str, modalities: list[str] = None) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Retrieves and processes the specified modalities for a given study.
    Data is pulled from the local cache if available, otherwise fetched from CPTAC.
    
    Args:
        study_name (str): Abbreviated name of the study.
        modalities (list[str]): Optional list of specific modalities to load and process. 
                                Defaults to all in the registry.
    """
    study_obj = _get_study_object(study_name)
    target_modalities = modalities if modalities else MODALITY_REGISTRY.keys()
    
    data_dict = {}

    for mod in target_modalities:
        if mod not in MODALITY_REGISTRY:
            print(f"Warning: Modality '{mod}' is not registered. Skipping.")
            continue

        try:
            # The get_raw function natively handles cache checking and creation
            raw_df = MODALITY_REGISTRY[mod]['get_raw'](study_obj)
            processed_data = MODALITY_REGISTRY[mod]['process'](raw_df)
            
            # Map 'clinical' back to 'has_recurrence' for compatibility with your downstream logic
            key_name = 'has_recurrence' if mod == 'clinical' else mod
            data_dict[key_name] = processed_data
            
        except Exception as e:
            print(f"Failed to process '{mod}': {e}")

    return data_dict

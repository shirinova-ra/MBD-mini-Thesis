import pandas as pd
import os
from pathlib import Path
import tarfile
from huggingface_hub import hf_hub_download
import re
import warnings

# Suppress all warnings for cleaner output during data download
warnings.filterwarnings('ignore')


def download_and_extract():
    """
    Download and extract MBD-mini dataset archives from Hugging Face Hub.
    Creates 'data' directory and extracts three tar.gz files: detail, targets, and client_split.
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # List of dataset archives to download
    archives = ['detail.tar.gz', 'targets.tar.gz', 'client_split.tar.gz']
    
    # Download and extract each archive
    for archive in archives:
        print(f"Downloading {archive}...")
        # Download from Hugging Face Hub dataset repository
        hf_hub_download(
            repo_id="ai-lab/MBD-mini",
            filename=archive,
            repo_type="dataset",
            local_dir='data'
        )
        print(f"Extracting {archive}...")
        # Extract the downloaded archive
        with tarfile.open(f'data/{archive}', 'r:gz') as tar:
            tar.extractall('data/')
    
    print("All archives are ready!")


# Uncomment for initial dataset download:
# download_and_extract()

def save_by_fold_with_clear_names(start_path, dataset_name):
    """
    Finds all fold=0, fold=1... directories and saves each as a single parquet file.
    Adds fold_number and dataset_name columns for easy identification.
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING: {dataset_name} (start: {start_path})")
    print(f"{'='*60}")
    
    fold_dirs = {}
    
    # Find ALL fold=0, fold=1... folders starting from data/
    for root, dirs, _ in os.walk(start_path):
        for dir_name in dirs:
            if dir_name.startswith('fold=') and dir_name[5:].isdigit():
                fold_num = dir_name[5:]
                fold_path = os.path.join(root, dir_name)
                fold_dirs[fold_num] = fold_path
                print(f"  Found: fold={fold_num} → {fold_path}")
    
    print(f"\nFound {len(fold_dirs)} fold folders")
    
    for fold_num in sorted(fold_dirs.keys(), key=int):
        fold_path = fold_dirs[fold_num]
        print(f"\n--> fold={fold_num}: {fold_path}")
        
        # All parquet files from fold=0 folder (and subfolders)
        parquet_files = []
        for r, _, files in os.walk(fold_path):
            for file in files:
                if file.endswith('.parquet') and os.path.getsize(os.path.join(r, file)) > 0:
                    parquet_files.append(os.path.join(r, file))
        
        print(f"  Parquet files: {len(parquet_files)}")
        
        # Combine
        df_list = [pd.read_parquet(f) for f in parquet_files]
        fold_df = pd.concat(df_list, ignore_index=True)
        fold_df['fold_number'] = fold_num
        fold_df['dataset_name'] = dataset_name
        
        # Save with clean name
        safe_name = dataset_name.replace('/', '_')
        filename = f"{safe_name}_fold_{fold_num}.parquet"
        output_dir = Path('data/final_by_fold_named') / safe_name
        output_dir.mkdir(parents=True, exist_ok=True)
        fold_df.to_parquet(output_dir / filename, index=False)
        print(f"{filename}: {fold_df.shape[0]:,} rows")
    
    return fold_dirs

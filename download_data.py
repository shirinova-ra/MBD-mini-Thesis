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

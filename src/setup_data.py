import os
import kaggle
import shutil
from pathlib import Path

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/raw',
        'data/preprocessed',
        'models/saved_models',
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def download_dataset():
    """Download dataset from Kaggle"""
    try:
        # Download dataset
        kaggle.api.dataset_download_files(
            'adinishad/driver-drowsiness-using-keras',
            path='data/raw',
            unzip=True
        )
        print("Dataset downloaded successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("Please make sure you have set up your Kaggle API credentials.")
        print("Visit https://www.kaggle.com/docs/api to learn how to set up your API credentials.")

def organize_data():
    """Organize downloaded data into proper directory structure"""
    raw_dir = Path('data/raw')
    processed_dir = Path('data/preprocessed')
    
    # Create directories for each class
    classes = ['closed', 'open', 'yawn']
    for class_name in classes:
        (processed_dir / class_name).mkdir(exist_ok=True)
        
        # Move files to appropriate directories
        source_dir = raw_dir / class_name
        if source_dir.exists():
            for img_file in source_dir.glob('*.jpg'):
                shutil.copy2(img_file, processed_dir / class_name)

def main():
    print("Setting up project directories...")
    setup_directories()
    
    print("\nDownloading dataset...")
    download_dataset()
    
    print("\nOrganizing data...")
    organize_data()
    
    print("\nSetup complete!")

if __name__ == "__main__":
    main()
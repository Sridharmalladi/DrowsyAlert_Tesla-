import os
from pathlib import Path

class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'preprocessed'
    MODEL_SAVE_DIR = PROJECT_ROOT / 'models' / 'saved_models'

    # Model parameters
    IMG_HEIGHT = 145
    IMG_WIDTH = 145
    CHANNELS = 3
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Classes
    CLASSES = ['closed', 'open', 'yawn']
    NUM_CLASSES = len(CLASSES)

    # Training parameters
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    RANDOM_STATE = 42

    # Detection parameters
    DETECTION_THRESHOLD = 0.5
    
    # Alert parameters
    DROWSINESS_THRESHOLD = 3  # Number of consecutive frames to trigger alert
    ALERT_COOLDOWN = 2.0  # Seconds between alerts
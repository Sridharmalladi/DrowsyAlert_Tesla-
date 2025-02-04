# DrowsAlert: Real-time Driver Drowsiness Detection System
An AI-Powered Solution for Enhanced Road Safety in Autonomous Vehicles

## Overview
DrowsAlert is a real-time drowsiness detection system that uses computer vision and deep learning to monitor driver alertness. The system analyzes facial features to detect signs of drowsiness and provides immediate alerts to prevent potential accidents.

## Features
- Real-time face detection and drowsiness monitoring
- Multi-feature analysis (eye state, facial expressions)
- Audio-visual alerts for drowsiness detection
- Performance visualization and metrics
- Easy-to-use interface

## Prerequisites
- Python 3.8 or higher
- Webcam
- Kaggle account for dataset access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DrowsAlert.git
cd DrowsAlert
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up Kaggle API:
- Go to kaggle.com → Account → Create New API Token
- Download kaggle.json
- Place it in ~/.kaggle/ directory
- Make sure the file has appropriate permissions

## Usage

1. Download and prepare the dataset:
```bash
python -m src.setup_data
```

2. Train the model:
```bash
python -m src.train
```

3. Run real-time detection:
```bash
python -m src.predict
```

## Project Structure
```
DrowsAlert/
│
├── data/                      # Data directory
│   ├── preprocessed/          # Preprocessed images
│   └── raw/                   # Raw downloaded data
│
├── models/                    # Saved models
│   └── saved_models/         # Trained model files
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── config.py            # Configuration settings
│   ├── data_loader.py       # Data loading utilities
│   ├── model.py             # Model architecture
│   ├── train.py             # Training script
│   ├── predict.py           # Real-time prediction
│   └── utils.py             # Utility functions
│
├── requirements.txt         # Project dependencies
└── README.md
```

## Model Performance
The system uses a CNN architecture trained on the Kaggle drowsiness detection dataset. Performance metrics including accuracy, precision, and recall are generated during training and saved in the models directory.

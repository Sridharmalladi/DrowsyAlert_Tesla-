import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from .data_loader import DataLoader
from .model import DrowsinessModel
from .config import Config
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def plot_training_history(history, save_dir):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(save_dir / f'training_history_{timestamp}.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(np.arange(len(classes)) + 0.5, classes, rotation=45)
    plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation=45)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(save_dir / f'confusion_matrix_{timestamp}.png')
    plt.close()

def save_classification_report(y_true, y_pred, classes, save_dir):
    """Save classification report"""
    report = classification_report(
        np.argmax(y_true, axis=1),
        np.argmax(y_pred, axis=1),
        target_names=classes
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = save_dir / f'classification_report_{timestamp}.txt'
    
    with open(report_path, 'w') as f:
        f.write(report)

def main():
    """Main training pipeline"""
    try:
        config = Config()
        logging.info("Starting training pipeline...")

        # Create necessary directories
        config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        logging.info("Loading and preparing data...")
        data_loader = DataLoader()
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data()
        
        logging.info(f"Training data shape: {X_train.shape}")
        logging.info(f"Validation data shape: {X_val.shape}")
        logging.info(f"Test data shape: {X_test.shape}")

        # Initialize and train model
        logging.info("Initializing model...")
        model = DrowsinessModel()
        
        logging.info("Starting model training...")
        history = model.train(X_train, y_train, X_val, y_val)

        # Evaluate model
        logging.info("Evaluating model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        logging.info(f"Test accuracy: {test_accuracy:.4f}")
        logging.info(f"Test loss: {test_loss:.4f}")

        # Generate predictions
        y_pred = model.predict(X_test)

        # Save training visualizations
        logging.info("Generating and saving visualizations...")
        plot_training_history(history, config.MODEL_SAVE_DIR)
        plot_confusion_matrix(y_test, y_pred, config.CLASSES, config.MODEL_SAVE_DIR)
        save_classification_report(y_test, y_pred, config.CLASSES, config.MODEL_SAVE_DIR)

        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model.save_model(f'drowsiness_model_final_{timestamp}.h5')
        
        logging.info("Training pipeline completed successfully!")

    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
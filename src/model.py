import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from .config import Config

class DataLoader:
    def __init__(self):
        self.config = Config()

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        img = cv2.imread(str(image_path))
        img = cv2.resize(img, (self.config.IMG_HEIGHT, self.config.IMG_WIDTH))
        img = img / 255.0  # Normalize
        return img

    def load_data(self):
        """Load all images and labels"""
        images = []
        labels = []
        
        # Load images from each class
        for class_idx, class_name in enumerate(self.config.CLASSES):
            class_path = self.config.PROCESSED_DATA_DIR / class_name
            for img_path in class_path.glob('*.jpg'):
                try:
                    img = self.load_and_preprocess_image(img_path)
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
                    continue

        return np.array(images), np.array(labels)

    def prepare_data(self):
        """Prepare dataset for training"""
        # Load data
        X, y = self.load_data()
        
        # Convert labels to categorical
        y = to_categorical(y, num_classes=self.config.NUM_CLASSES)

        # Split into train and temporary test set
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )

        # Split temporary test set into validation and test set
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            random_state=self.config.RANDOM_STATE
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
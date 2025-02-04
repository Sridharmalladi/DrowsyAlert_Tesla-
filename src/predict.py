import cv2
import numpy as np
import time
from pygame import mixer
import logging
from pathlib import Path
from .model import DrowsinessModel
from .config import Config

class DrowsinessDetector:
    def __init__(self, model_path):
        self.config = Config()
        self.model = DrowsinessModel.load_saved_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize pygame mixer for alerts
        mixer.init()
        self.alert_sound = self._create_alert_sound()
        
        # Initialize state variables
        self.drowsy_frame_count = 0
        self.last_alert_time = time.time()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _create_alert_sound(self):
        """Create a simple alert sound using pygame mixer"""
        frequency = 2000  # Hz
        duration = 1000   # ms
        sample_rate = 44100
        
        # Generate a simple sine wave
        t = np.linspace(0, duration/1000, int(sample_rate * duration/1000))
        signal = np.sin(2 * np.pi * frequency * t)
        sound = np.asarray([32767 * signal, 32767 * signal]).T.astype(np.int16)
        
        return mixer.Sound(sound)

    def preprocess_frame(self, frame):
        """Preprocess a single frame"""
        frame = cv2.resize(frame, (self.config.IMG_HEIGHT, self.config.IMG_WIDTH))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)
        return frame

    def play_alert(self):
        """Play alert sound if cooldown has elapsed"""
        current_time = time.time()
        if current_time - self.last_alert_time >= self.config.ALERT_COOLDOWN:
            self.alert_sound.play()
            self.last_alert_time = current_time

    def detect_drowsiness(self, frame):
        """Detect drowsiness in a single frame"""
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                # Preprocess ROI
                processed_roi = self.preprocess_frame(face_roi)

                # Make prediction
                prediction = self.model.predict(processed_roi)
                class_idx = np.argmax(prediction[0])
                class_name = self.config.CLASSES[class_idx]
                confidence = prediction[0][class_idx]

                # Update drowsiness counter
                if class_name == 'closed' and confidence > self.config.DETECTION_THRESHOLD:
                    self.drowsy_frame_count += 1
                else:
                    self.drowsy_frame_count = 0

                # Check for drowsiness alert
                if self.drowsy_frame_count >= self.config.DROWSINESS_THRESHOLD:
                    self.play_alert()
                    color = (0, 0, 255)  # Red
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    color = (0, 255, 0)  # Green

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            except Exception as e:
                self.logger.error(f"Error processing ROI: {str(e)}")
                continue

        return frame

    def start_webcam_detection(self):
        """Start real-time drowsiness detection using webcam"""
        cap = cv2.VideoCapture(0)
        self.logger.info("Starting webcam detection...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to grab frame")
                    break

                frame = self.detect_drowsiness(frame)
                
                # Display frame
                cv2.imshow('DrowsAlert - Press Q to quit', frame)

                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            self.logger.error(f"Error in webcam detection: {str(e)}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Webcam detection stopped")

def main():
    """Main function to run real-time detection"""
    try:
        # Use the latest model in the saved_models directory
        model_dir = Config.MODEL_SAVE_DIR
        model_files = list(model_dir.glob('*.h5'))
        if not model_files:
            raise FileNotFoundError("No model files found in saved_models directory")
        
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        detector = DrowsinessDetector(latest_model.name)
        detector.start_webcam_detection()

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
import numpy as np
import cv2
from PIL import Image
import os

class NumberDetector:
    """
    SUB ai - Number Detection Model
    Detects and recognizes numbers from images.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the NumberDetector.
        
        Args:
            model_path (str): Path to pre-trained model (optional)
        """
        self.model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No pre-trained model loaded. Using basic detection.")
    
    def preprocess_image(self, image_path):
        """
        Preprocess the image for number detection.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size (28x28 for MNIST-like models)
        resized = cv2.resize(gray, (28, 28))
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        return normalized
    
    def detect(self, image_path):
        """
        Detect if the image contains a number.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Detection result with confidence and classification
        """
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(image_path)
            
            if self.model is None:
                # Basic heuristic detection (placeholder)
                result = self._basic_detection(processed_img)
            else:
                # Use trained model for detection
                result = self._model_detection(processed_img)
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'is_number': False
            }
    
    def _basic_detection(self, image):
        """
        Basic heuristic-based detection (placeholder for initial testing).
        
        Args:
            image (numpy.ndarray): Preprocessed image
            
        Returns:
            dict: Detection result
        """
        # Calculate image statistics
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # Simple heuristic: images with moderate variation might be numbers
        is_number = 0.1 < mean_intensity < 0.9 and std_intensity > 0.1
        
        return {
            'status': 'success',
            'is_number': is_number,
            'message': 'This is a number image!' if is_number else 'This is not a number image.',
            'confidence': 0.5,  # Placeholder confidence
            'method': 'basic_heuristic'
        }
    
    def _model_detection(self, image):
        """
        Model-based detection using trained neural network.
        
        Args:
            image (numpy.ndarray): Preprocessed image
            
        Returns:
            dict: Detection result
        """
        # Reshape for model input
        img_input = image.reshape(1, 28, 28, 1)
        
        # Get prediction
        prediction = self.model.predict(img_input, verbose=0)
        confidence = float(np.max(prediction))
        predicted_digit = int(np.argmax(prediction))
        
        is_number = confidence > 0.7  # Threshold for classification
        
        return {
            'status': 'success',
            'is_number': is_number,
            'predicted_digit': predicted_digit if is_number else None,
            'message': f'This is a number image! Detected: {predicted_digit}' if is_number else 'This is not a number image.',
            'confidence': confidence,
            'method': 'neural_network'
        }
    
    def load_model(self, model_path):
        """
        Load a pre-trained model.
        
        Args:
            model_path (str): Path to the saved model
        """
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def save_model(self, save_path):
        """
        Save the current model.
        
        Args:
            save_path (str): Path to save the model
        """
        if self.model is not None:
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
        else:
            print("No model to save.")


if __name__ == "__main__":
    # Example usage
    detector = NumberDetector()
    
    print("SUB ai - Number Detector")
    print("="*50)
    print("Usage: detector.detect('path/to/image.jpg')")
    print("\nReady for detection!")

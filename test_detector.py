from number_detector import NumberDetector
import cv2
import numpy as np
import os

def create_test_images():
    """
    Create sample test images for testing the detector.
    """
    os.makedirs('test_images', exist_ok=True)
    
    # Create a simple number image (digit 5)
    number_img = np.zeros((28, 28), dtype=np.uint8)
    cv2.putText(number_img, '5', (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 255, 2)
    cv2.imwrite('test_images/number_5.png', number_img)
    
    # Create a non-number image (random pattern)
    non_number_img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    cv2.imwrite('test_images/not_number.png', non_number_img)
    
    # Create another number (digit 3)
    number_img2 = np.zeros((28, 28), dtype=np.uint8)
    cv2.putText(number_img2, '3', (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 255, 2)
    cv2.imwrite('test_images/number_3.png', number_img2)
    
    print("Test images created in 'test_images/' directory\n")

def test_with_model():
    """
    Test the detector with a trained model.
    """
    print("="*60)
    print("SUB ai - Number Detection Test (with trained model)")
    print("="*60)
    print()
    
    # Initialize detector with trained model
    model_path = 'models/sub_ai_model_latest.h5'
    
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        print("Please run 'python train.py' first to train the model.\n")
        return
    
    detector = NumberDetector(model_path=model_path)
    
    # Create test images if they don't exist
    if not os.path.exists('test_images'):
        create_test_images()
    
    # Test images
    test_files = [
        'test_images/number_5.png',
        'test_images/number_3.png',
        'test_images/not_number.png'
    ]
    
    for img_path in test_files:
        if os.path.exists(img_path):
            print(f"Testing: {img_path}")
            result = detector.detect(img_path)
            
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            if result.get('predicted_digit') is not None:
                print(f"Predicted Digit: {result['predicted_digit']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Method: {result['method']}")
            print("-" * 60)
            print()

def test_without_model():
    """
    Test the detector without a trained model (basic detection).
    """
    print("="*60)
    print("SUB ai - Number Detection Test (basic mode)")
    print("="*60)
    print()
    
    # Initialize detector without model
    detector = NumberDetector()
    
    # Create test images if they don't exist
    if not os.path.exists('test_images'):
        create_test_images()
    
    # Test images
    test_files = [
        'test_images/number_5.png',
        'test_images/number_3.png',
        'test_images/not_number.png'
    ]
    
    for img_path in test_files:
        if os.path.exists(img_path):
            print(f"Testing: {img_path}")
            result = detector.detect(img_path)
            
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Method: {result['method']}")
            print("-" * 60)
            print()

if __name__ == "__main__":
    import sys
    
    # Check if model exists
    if os.path.exists('models/sub_ai_model_latest.h5'):
        test_with_model()
    else:
        print("No trained model found. Testing with basic detection mode...\n")
        test_without_model()
        print("\nTo use the trained model, run: python train.py\n")

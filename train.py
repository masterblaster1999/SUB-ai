import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from datetime import datetime

class SUBaiTrainer:
    """
    SUB ai Training Module
    Trains the number detection model using MNIST dataset.
    """
    
    def __init__(self, model_name="sub_ai_model"):
        """
        Initialize the trainer.
        
        Args:
            model_name (str): Name for saving the model
        """
        self.model_name = model_name
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
    def load_data(self):
        """
        Load and preprocess the MNIST dataset.
        """
        print("Loading MNIST dataset...")
        
        # Load MNIST data
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        self.x_train = x_train.astype("float32") / 255.0
        self.x_test = x_test.astype("float32") / 255.0
        
        # Reshape for CNN (add channel dimension)
        self.x_train = np.expand_dims(self.x_train, -1)
        self.x_test = np.expand_dims(self.x_test, -1)
        
        # Convert labels to categorical
        self.y_train = keras.utils.to_categorical(y_train, 10)
        self.y_test = keras.utils.to_categorical(y_test, 10)
        
        print(f"Training samples: {self.x_train.shape[0]}")
        print(f"Testing samples: {self.x_test.shape[0]}")
        print(f"Image shape: {self.x_train.shape[1:]}")
        print("Data loaded successfully!\n")
        
    def build_model(self, model_type="cnn"):
        """
        Build the neural network model.
        
        Args:
            model_type (str): Type of model ('cnn' or 'simple')
        """
        print(f"Building {model_type.upper()} model...")
        
        if model_type == "cnn":
            self.model = self._build_cnn()
        else:
            self.model = self._build_simple()
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model built successfully!\n")
        self.model.summary()
        print()
        
    def _build_cnn(self):
        """
        Build a Convolutional Neural Network.
        
        Returns:
            keras.Model: CNN model
        """
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                         input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')
        ])
        
        return model
    
    def _build_simple(self):
        """
        Build a simple feedforward neural network.
        
        Returns:
            keras.Model: Simple model
        """
        model = keras.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        return model
    
    def train(self, epochs=10, batch_size=128, validation_split=0.1):
        """
        Train the model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of training data for validation
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if self.x_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"Starting training for {epochs} epochs...\n")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!\n")
        
    def evaluate(self):
        """
        Evaluate the model on test data.
        
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        print("Evaluating model on test data...")
        
        test_loss, test_accuracy = self.model.evaluate(
            self.x_test, self.y_test, verbose=0
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
    
    def save_model(self, custom_path=None):
        """
        Save the trained model.
        
        Args:
            custom_path (str): Custom path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        if custom_path:
            save_path = custom_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"models/{self.model_name}_{timestamp}.h5"
        
        self.model.save(save_path)
        print(f"Model saved to: {save_path}")
        
        # Also save as latest
        latest_path = f"models/{self.model_name}_latest.h5"
        self.model.save(latest_path)
        print(f"Model saved to: {latest_path}\n")
        
        return save_path
    
    def plot_training_history(self, save_plot=True):
        """
        Plot training history.
        
        Args:
            save_plot (bool): Whether to save the plot
        """
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = f"models/{self.model_name}_training_history.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {plot_path}")
        
        plt.show()
    
    def test_prediction(self, num_samples=5):
        """
        Test predictions on random samples.
        
        Args:
            num_samples (int): Number of samples to test
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        print(f"Testing predictions on {num_samples} random samples...\n")
        
        # Get random indices
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        
        for idx in indices:
            img = self.x_test[idx:idx+1]
            true_label = np.argmax(self.y_test[idx])
            
            # Predict
            prediction = self.model.predict(img, verbose=0)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            print(f"Sample {idx}:")
            print(f"  True Label: {true_label}")
            print(f"  Predicted: {predicted_label}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Status: {'✓ Correct' if true_label == predicted_label else '✗ Wrong'}\n")


def main():
    """
    Main training pipeline.
    """
    print("="*60)
    print("SUB ai - Number Detection Model Training")
    print("="*60)
    print()
    
    # Initialize trainer
    trainer = SUBaiTrainer(model_name="sub_ai_model")
    
    # Load data
    trainer.load_data()
    
    # Build model (CNN for better accuracy)
    trainer.build_model(model_type="cnn")
    
    # Train the model
    trainer.train(epochs=15, batch_size=128, validation_split=0.1)
    
    # Evaluate
    results = trainer.evaluate()
    
    # Test predictions
    trainer.test_prediction(num_samples=10)
    
    # Save model
    model_path = trainer.save_model()
    
    # Plot training history
    trainer.plot_training_history(save_plot=True)
    
    print("="*60)
    print("Training completed successfully!")
    print(f"Final Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"Model saved to: {model_path}")
    print("="*60)


if __name__ == "__main__":
    main()

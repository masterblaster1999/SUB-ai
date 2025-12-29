import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
import json
from datetime import datetime
import sys

class ChatModelTrainer:
    """
    SUB ai Chat Model Trainer
    Trains a conversational AI model using sequence-to-sequence architecture.
    """
    
    def __init__(self, model_name="sub_ai_chat"):
        """
        Initialize the trainer.
        
        Args:
            model_name (str): Name for saving the model
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.vocab_size = 10000
        self.max_length = 50
        self.embedding_dim = 128
        self.hidden_units = 256
        self.training_data = []
        
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
    
    def create_training_data(self):
        """
        Create sample training data for the chat model.
        This would typically be replaced with a larger dataset.
        """
        print("Creating training data...")
        
        # Sample conversation pairs (question -> answer)
        conversations = [
            # Greetings
            ("hello", "Hello! How can I help you today?"),
            ("hi", "Hi there! I'm SUB ai, your AI assistant."),
            ("hey", "Hey! What can I do for you?"),
            ("good morning", "Good morning! Hope you're having a great day!"),
            ("good evening", "Good evening! How can I assist you?"),
            
            # Identity questions
            ("what is your name", "My name is SUB ai, a small language model."),
            ("who are you", "I'm SUB ai, an AI assistant designed to help you."),
            ("what can you do", "I can detect numbers from images and chat with you!"),
            ("tell me about yourself", "I'm SUB ai, created to assist with number detection and conversations."),
            
            # How are you
            ("how are you", "I'm doing great! Thanks for asking. How about you?"),
            ("how are you doing", "I'm functioning perfectly! How can I help you today?"),
            ("what's up", "Not much! Just here to help you. What's on your mind?"),
            
            # Help
            ("help", "I can help with various tasks! Ask me anything."),
            ("can you help me", "Of course! I'm here to help. What do you need?"),
            ("i need help", "Sure! Tell me what you need help with."),
            
            # Thanks
            ("thank you", "You're welcome! Happy to help!"),
            ("thanks", "You're welcome!"),
            ("appreciate it", "No problem! Glad I could assist."),
            
            # Numbers and math
            ("what is a number", "A number is a mathematical value used for counting and calculations."),
            ("tell me about numbers", "Numbers are fundamental units in mathematics used to quantify things."),
            ("what numbers can you detect", "I can detect digits from 0 to 9 in images!"),
            
            # Capabilities
            ("what is your purpose", "My purpose is to help you with number detection and answer your questions."),
            ("why were you created", "I was created to demonstrate small language model capabilities."),
            ("are you an ai", "Yes, I'm an AI assistant called SUB ai!"),
            
            # General knowledge
            ("what is ai", "AI stands for Artificial Intelligence - computer systems that can learn and reason."),
            ("what is machine learning", "Machine learning is AI that learns from data to make predictions."),
            ("what is deep learning", "Deep learning uses neural networks with multiple layers to learn complex patterns."),
            
            # Conversation
            ("tell me a joke", "Why did the AI go to school? To improve its neural network!"),
            ("tell me something interesting", "Did you know? The MNIST dataset contains 70,000 handwritten digit images!"),
            ("what's new", "I'm always learning! What would you like to know?"),
            
            # Goodbye
            ("bye", "Goodbye! Have a great day!"),
            ("goodbye", "Goodbye! Come back anytime!"),
            ("see you later", "See you later! Take care!"),
            ("gotta go", "Alright! Talk to you soon!"),
            
            # Questions
            ("how old are you", "I was just created, so I'm very new!"),
            ("where are you from", "I exist in the digital realm, created by developers."),
            ("do you have feelings", "I don't have feelings like humans, but I'm designed to be helpful!"),
            
            # More variations
            ("nice to meet you", "Nice to meet you too! How can I help?"),
            ("how does it work", "I use neural networks to understand and generate text responses."),
            ("can you learn", "Yes! Through training on conversation data, I can improve."),
            ("are you smart", "I'm designed to be helpful and knowledgeable within my training!"),
            ("what language do you speak", "I primarily speak English, and I'm always learning!"),
        ]
        
        # Add variations and augmentations
        augmented = []
        for q, a in conversations:
            augmented.append((q, a))
            # Add some case variations
            augmented.append((q.capitalize(), a))
            augmented.append((q.upper(), a))
        
        self.training_data = augmented
        
        print(f"Created {len(self.training_data)} training samples")
        print(f"Sample: {self.training_data[0]}\n")
        
        return self.training_data
    
    def prepare_data(self):
        """
        Prepare and tokenize training data.
        
        Returns:
            tuple: (input_sequences, target_sequences)
        """
        print("Preparing data...")
        
        if not self.training_data:
            self.create_training_data()
        
        # Separate inputs and targets
        inputs = [pair[0] for pair in self.training_data]
        targets = [pair[1] for pair in self.training_data]
        
        # Create tokenizer
        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=self.vocab_size,
            oov_token='<OOV>'
        )
        
        # Fit on all text
        all_text = inputs + targets
        self.tokenizer.fit_on_texts(all_text)
        
        # Convert to sequences
        input_sequences = self.tokenizer.texts_to_sequences(inputs)
        target_sequences = self.tokenizer.texts_to_sequences(targets)
        
        # Pad sequences
        input_sequences = keras.preprocessing.sequence.pad_sequences(
            input_sequences, maxlen=self.max_length, padding='post'
        )
        target_sequences = keras.preprocessing.sequence.pad_sequences(
            target_sequences, maxlen=self.max_length, padding='post'
        )
        
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Input shape: {input_sequences.shape}")
        print(f"Target shape: {target_sequences.shape}\n")
        
        return input_sequences, target_sequences
    
    def build_model(self):
        """
        Build the sequence-to-sequence chat model.
        """
        print("Building chat model...")
        
        vocab_size = min(self.vocab_size, len(self.tokenizer.word_index) + 1)
        
        # Simple sequence model
        model = keras.Sequential([
            layers.Embedding(vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.Bidirectional(layers.LSTM(self.hidden_units, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(self.hidden_units // 2)),
            layers.Dropout(0.3),
            layers.Dense(self.hidden_units, activation='relu'),
            layers.Dense(vocab_size, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("Model built successfully!\n")
        self.model.summary()
        print()
        
    def train(self, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the chat model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation data fraction
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        # Prepare data
        X, y = self.prepare_data()
        
        # Reshape y for sparse_categorical_crossentropy
        y = y[:, 0]  # Take first token as target (simplified)
        
        print(f"Starting training for {epochs} epochs...\n")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!\n")
        return history
    
    def save_model(self):
        """
        Save the trained model and vocabulary.
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/{self.model_name}_{timestamp}.h5"
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Save as latest
        latest_path = f"models/{self.model_name}_latest.h5"
        self.model.save(latest_path)
        print(f"Model saved to: {latest_path}")
        
        # Save vocabulary
        vocab_path = "models/chat_vocab.pkl"
        vocab_data = {
            'tokenizer': self.tokenizer,
            'vocab': self.tokenizer.word_index,
            'max_length': self.max_length
        }
        
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"Vocabulary saved to: {vocab_path}\n")
        
        return latest_path


def main():
    """
    Main training pipeline for chat model.
    """
    print("="*60)
    print("SUB ai - Chat Model Training")
    print("="*60)
    print()
    
    # Get parameters
    epochs = int(os.environ.get('CHAT_EPOCHS', '100'))
    batch_size = int(os.environ.get('CHAT_BATCH_SIZE', '32'))
    
    print(f"Training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print()
    
    try:
        # Initialize trainer
        trainer = ChatModelTrainer()
        
        # Create training data
        trainer.create_training_data()
        
        # Build model
        trainer.build_model()
        
        # Train
        history = trainer.train(epochs=epochs, batch_size=batch_size)
        
        # Save
        model_path = trainer.save_model()
        
        print("="*60)
        print("Chat training completed successfully!")
        print(f"Model saved to: {model_path}")
        print("\nTry it out: python chat_ai.py")
        print("="*60)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

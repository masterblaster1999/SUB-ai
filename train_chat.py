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
    Trains a conversational AI model using datasets from Hugging Face.
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
        self.X_train = None
        self.y_train = None
        
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
    
    def load_huggingface_dataset(self, dataset_name="daily_dialog", max_samples=5000):
        """
        Load conversational dataset from Hugging Face.
        
        Args:
            dataset_name (str): Name of the dataset to load
            max_samples (int): Maximum number of conversation pairs to use
        """
        print(f"Loading dataset from Hugging Face: {dataset_name}...")
        
        try:
            from datasets import load_dataset
            
            conversations = []
            
            if dataset_name == "daily_dialog":
                # Load DailyDialog dataset
                print("Downloading DailyDialog dataset...")
                dataset = load_dataset("daily_dialog", split="train")
                
                for item in dataset:
                    dialog = item['dialog']
                    # Create pairs from consecutive turns
                    for i in range(0, len(dialog) - 1, 2):
                        if i + 1 < len(dialog):
                            question = dialog[i]
                            answer = dialog[i + 1]
                            if len(question) > 5 and len(answer) > 5:
                                conversations.append((question, answer))
                    
                    if len(conversations) >= max_samples:
                        break
            
            elif dataset_name == "empathetic_dialogues":
                # Load Empathetic Dialogues dataset
                print("Downloading Empathetic Dialogues dataset...")
                dataset = load_dataset("empathetic_dialogues", split="train")
                
                for item in dataset:
                    prompt = item['prompt']
                    utterance = item['utterance']
                    if len(prompt) > 5 and len(utterance) > 5:
                        conversations.append((prompt, utterance))
                    
                    if len(conversations) >= max_samples:
                        break
            
            else:
                print(f"Unknown dataset: {dataset_name}. Using fallback data.")
                return self.create_training_data()
            
            self.training_data = conversations[:max_samples]
            
            print(f"Loaded {len(self.training_data)} conversation pairs from Hugging Face")
            print(f"Sample: {self.training_data[0]}\n")
            
            return self.training_data
            
        except ImportError:
            print("⚠️  'datasets' library not found. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
            print("Please run the script again.")
            sys.exit(1)
        except Exception as e:
            print(f"⚠️  Error loading Hugging Face dataset: {e}")
            print("Falling back to local training data...")
            return self.create_training_data()
    
    def create_training_data(self):
        """
        Create sample training data for the chat model (fallback).
        This is used if Hugging Face datasets are unavailable.
        """
        print("Creating local training data...")
        
        # Comprehensive conversation pairs
        conversations = [
            # Greetings
            ("hello", "Hello! How can I help you today?"),
            ("hi", "Hi there! I'm SUB ai, your AI assistant."),
            ("hey", "Hey! What can I do for you?"),
            ("good morning", "Good morning! Hope you're having a great day!"),
            ("good evening", "Good evening! How can I assist you?"),
            ("good afternoon", "Good afternoon! What can I help you with?"),
            ("howdy", "Howdy! Nice to meet you!"),
            
            # Identity questions
            ("what is your name", "My name is SUB ai, a small language model."),
            ("who are you", "I'm SUB ai, an AI assistant designed to help you."),
            ("what can you do", "I can detect numbers from images and chat with you!"),
            ("tell me about yourself", "I'm SUB ai, created to assist with number detection and conversations."),
            ("what are your capabilities", "I can recognize digits in images and have natural conversations!"),
            
            # How are you
            ("how are you", "I'm doing great! Thanks for asking. How about you?"),
            ("how are you doing", "I'm functioning perfectly! How can I help you today?"),
            ("what's up", "Not much! Just here to help you. What's on your mind?"),
            ("how's it going", "It's going well! What can I do for you?"),
            
            # Help
            ("help", "I can help with various tasks! Ask me anything."),
            ("can you help me", "Of course! I'm here to help. What do you need?"),
            ("i need help", "Sure! Tell me what you need help with."),
            ("i need assistance", "I'm here to assist! What can I do for you?"),
            
            # Thanks
            ("thank you", "You're welcome! Happy to help!"),
            ("thanks", "You're welcome!"),
            ("appreciate it", "No problem! Glad I could assist."),
            ("thanks a lot", "You're very welcome! Anytime!"),
            
            # Numbers and math
            ("what is a number", "A number is a mathematical value used for counting and calculations."),
            ("tell me about numbers", "Numbers are fundamental units in mathematics used to quantify things."),
            ("what numbers can you detect", "I can detect digits from 0 to 9 in images!"),
            ("how do you detect numbers", "I use a convolutional neural network trained on the MNIST dataset!"),
            
            # Capabilities
            ("what is your purpose", "My purpose is to help you with number detection and answer your questions."),
            ("why were you created", "I was created to demonstrate small language model capabilities."),
            ("are you an ai", "Yes, I'm an AI assistant called SUB ai!"),
            ("are you a robot", "I'm an AI program, not a physical robot, but I can help you!"),
            
            # General knowledge
            ("what is ai", "AI stands for Artificial Intelligence - computer systems that can learn and reason."),
            ("what is machine learning", "Machine learning is AI that learns from data to make predictions."),
            ("what is deep learning", "Deep learning uses neural networks with multiple layers to learn complex patterns."),
            ("what is a neural network", "A neural network is a computing system inspired by biological brains."),
            ("explain neural networks", "Neural networks process information through layers of connected nodes, learning patterns from data."),
            
            # Conversation
            ("tell me a joke", "Why did the AI go to school? To improve its neural network!"),
            ("tell me something interesting", "Did you know? The MNIST dataset contains 70,000 handwritten digit images!"),
            ("what's new", "I'm always learning! What would you like to know?"),
            ("tell me a fun fact", "The first artificial neural network was created in 1943!"),
            
            # Goodbye
            ("bye", "Goodbye! Have a great day!"),
            ("goodbye", "Goodbye! Come back anytime!"),
            ("see you later", "See you later! Take care!"),
            ("gotta go", "Alright! Talk to you soon!"),
            ("see you", "See you! Have a wonderful day!"),
            
            # Questions about AI
            ("how old are you", "I was just created, so I'm very new!"),
            ("where are you from", "I exist in the digital realm, created by developers."),
            ("do you have feelings", "I don't have feelings like humans, but I'm designed to be helpful!"),
            ("can you think", "I process information and generate responses, which is different from human thinking."),
            
            # More variations
            ("nice to meet you", "Nice to meet you too! How can I help?"),
            ("how does it work", "I use neural networks to understand and generate text responses."),
            ("can you learn", "Yes! Through training on conversation data, I can improve."),
            ("are you smart", "I'm designed to be helpful and knowledgeable within my training!"),
            ("what language do you speak", "I primarily speak English, and I'm always learning!"),
            
            # Technical questions
            ("what model are you", "I'm SUB ai, a small language model with vision and language capabilities."),
            ("what framework do you use", "I'm built with TensorFlow and Keras for deep learning."),
            ("how were you trained", "I was trained on conversational data using neural networks."),
            
            # More natural conversations
            ("that's cool", "Thanks! Is there anything else you'd like to know?"),
            ("interesting", "Glad you think so! What else can I help with?"),
            ("okay", "Alright! Let me know if you need anything."),
            ("i see", "Great! Do you have any questions?"),
        ]
        
        # Add variations
        augmented = []
        for q, a in conversations:
            augmented.append((q, a))
            # Add capitalized version
            augmented.append((q.capitalize(), a))
        
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
            self.load_huggingface_dataset()
        
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
        
        # Store for training
        self.X_train = input_sequences
        self.y_train = target_sequences[:, 0]  # Take first token as target (simplified)
        
        return input_sequences, target_sequences
    
    def build_model(self):
        """
        Build the sequence-to-sequence chat model.
        """
        print("Building chat model...")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call prepare_data() first.")
        
        vocab_size = min(self.vocab_size, len(self.tokenizer.word_index) + 1)
        
        # Enhanced sequence model
        model = keras.Sequential([
            layers.Embedding(vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.Bidirectional(layers.LSTM(self.hidden_units, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(self.hidden_units // 2)),
            layers.Dropout(0.3),
            layers.Dense(self.hidden_units, activation='relu'),
            layers.Dropout(0.2),
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
            raise ValueError("Model not built. Call build_model() first.")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
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
            self.X_train, self.y_train,
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
    print("SUB ai - Chat Model Training (with Hugging Face)")
    print("="*60)
    print()
    
    # Get parameters
    epochs = int(os.environ.get('CHAT_EPOCHS', '100'))
    batch_size = int(os.environ.get('CHAT_BATCH_SIZE', '32'))
    use_hf_dataset = os.environ.get('USE_HF_DATASET', 'true').lower() == 'true'
    dataset_name = os.environ.get('HF_DATASET', 'daily_dialog')
    max_samples = int(os.environ.get('MAX_SAMPLES', '5000'))
    
    print(f"Training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Use Hugging Face: {use_hf_dataset}")
    if use_hf_dataset:
        print(f"  Dataset: {dataset_name}")
        print(f"  Max Samples: {max_samples}")
    print()
    
    try:
        # Initialize trainer
        trainer = ChatModelTrainer()
        
        # Load training data
        if use_hf_dataset:
            trainer.load_huggingface_dataset(dataset_name=dataset_name, max_samples=max_samples)
        else:
            trainer.create_training_data()
        
        # Prepare data (creates tokenizer)
        trainer.prepare_data()
        
        # Build model (now tokenizer exists)
        trainer.build_model()
        
        # Train
        history = trainer.train(epochs=epochs, batch_size=batch_size)
        
        # Save
        model_path = trainer.save_model()
        
        print("="*60)
        print("✅ Chat training completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Training samples used: {len(trainer.training_data)}")
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

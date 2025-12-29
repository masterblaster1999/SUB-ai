import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
import json
from datetime import datetime

class SUBChatAI:
    """
    SUB ai - Conversational AI Module
    Handles text-based conversations using neural language model.
    """
    
    def __init__(self, model_path=None, vocab_path=None):
        """
        Initialize the chat AI.
        
        Args:
            model_path (str): Path to pre-trained chat model
            vocab_path (str): Path to vocabulary file
        """
        self.model = None
        self.tokenizer = None
        self.vocab = None
        self.max_length = 50
        self.embedding_dim = 128
        self.hidden_units = 256
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            print("No vocabulary loaded. Training or loading vocab required.")
    
    def preprocess_text(self, text):
        """
        Preprocess text for the model.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def chat(self, user_input, temperature=0.7, max_response_length=100):
        """
        Generate a response to user input.
        
        Args:
            user_input (str): User's message
            temperature (float): Sampling temperature (higher = more random)
            max_response_length (int): Maximum length of response
            
        Returns:
            dict: Response with message and metadata
        """
        try:
            # Preprocess input
            processed_input = self.preprocess_text(user_input)
            
            if self.model is None or self.tokenizer is None:
                # Fallback responses without model
                return self._fallback_response(processed_input)
            
            # Generate response using model
            response = self._generate_response(
                processed_input, 
                temperature=temperature,
                max_length=max_response_length
            )
            
            return {
                'status': 'success',
                'user_input': user_input,
                'response': response,
                'model': 'neural_language_model',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response': "I'm having trouble understanding. Could you rephrase that?",
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_response(self, input_text, temperature=0.7, max_length=100):
        """
        Generate response using the trained model.
        
        Args:
            input_text (str): Preprocessed input text
            temperature (float): Sampling temperature
            max_length (int): Maximum response length
            
        Returns:
            str: Generated response
        """
        # Tokenize input
        input_seq = self.tokenizer.texts_to_sequences([input_text])
        input_seq = keras.preprocessing.sequence.pad_sequences(
            input_seq, maxlen=self.max_length, padding='post'
        )
        
        # Generate response tokens
        response_tokens = []
        current_input = input_seq
        
        for _ in range(max_length):
            # Predict next token
            predictions = self.model.predict(current_input, verbose=0)[0]
            
            # Apply temperature
            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))
            
            # Sample next token
            next_token = np.random.choice(len(predictions), p=predictions)
            
            # Stop if end token
            if next_token == 0 or next_token == self.tokenizer.word_index.get('<end>', 0):
                break
            
            response_tokens.append(next_token)
            
            # Update input for next prediction
            current_input = np.append(current_input[:, 1:], [[next_token]], axis=1)
        
        # Convert tokens to text
        response = self.tokenizer.sequences_to_texts([response_tokens])[0]
        
        return response if response else "I understand. Tell me more!"
    
    def _fallback_response(self, input_text):
        """
        Generate rule-based responses when model is not available.
        
        Args:
            input_text (str): User input
            
        Returns:
            dict: Response dictionary
        """
        # Simple rule-based responses
        responses = {
            'hello': 'Hello! I\'m SUB ai. How can I help you today?',
            'hi': 'Hi there! I\'m SUB ai, your AI assistant.',
            'how are you': 'I\'m doing great! Thanks for asking. How can I assist you?',
            'what is your name': 'My name is SUB ai, a small language model designed to help you!',
            'who are you': 'I\'m SUB ai, an AI assistant created to detect numbers and chat with you!',
            'help': 'I can help you with various tasks! I can detect numbers from images and have conversations with you.',
            'thank you': 'You\'re welcome! Happy to help!',
            'thanks': 'You\'re welcome!',
            'bye': 'Goodbye! Have a great day!',
            'goodbye': 'Goodbye! Come back anytime!',
        }
        
        # Check for keyword matches
        for key, value in responses.items():
            if key in input_text:
                return {
                    'status': 'success',
                    'user_input': input_text,
                    'response': value,
                    'model': 'rule_based',
                    'timestamp': datetime.now().isoformat()
                }
        
        # Default responses
        default_responses = [
            "That's interesting! Tell me more.",
            "I understand. What else would you like to know?",
            "That's a great question! I'm still learning.",
            "I'm here to help! Could you provide more details?",
            "Interesting! How can I assist you further?"
        ]
        
        response = np.random.choice(default_responses)
        
        return {
            'status': 'success',
            'user_input': input_text,
            'response': response,
            'model': 'rule_based',
            'timestamp': datetime.now().isoformat()
        }
    
    def load_model(self, model_path):
        """
        Load a pre-trained chat model.
        
        Args:
            model_path (str): Path to the saved model
        """
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Chat model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def load_vocab(self, vocab_path):
        """
        Load vocabulary and tokenizer.
        
        Args:
            vocab_path (str): Path to vocabulary file
        """
        try:
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
            
            self.tokenizer = vocab_data['tokenizer']
            self.vocab = vocab_data['vocab']
            self.max_length = vocab_data.get('max_length', 50)
            
            print(f"Vocabulary loaded from {vocab_path}")
            print(f"Vocabulary size: {len(self.vocab)}")
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
            self.tokenizer = None
    
    def save_vocab(self, vocab_path):
        """
        Save vocabulary and tokenizer.
        
        Args:
            vocab_path (str): Path to save vocabulary
        """
        if self.tokenizer is None:
            print("No tokenizer to save.")
            return
        
        vocab_data = {
            'tokenizer': self.tokenizer,
            'vocab': self.vocab,
            'max_length': self.max_length
        }
        
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"Vocabulary saved to {vocab_path}")


def interactive_chat():
    """
    Start an interactive chat session.
    """
    print("="*60)
    print("SUB ai - Interactive Chat")
    print("="*60)
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    # Initialize chat AI
    model_path = 'models/sub_ai_chat_latest.h5'
    vocab_path = 'models/chat_vocab.pkl'
    
    chat_ai = SUBChatAI(
        model_path=model_path if os.path.exists(model_path) else None,
        vocab_path=vocab_path if os.path.exists(vocab_path) else None
    )
    
    if not os.path.exists(model_path):
        print("Note: Using rule-based responses. Train the model for better conversations.")
        print("Run 'python train_chat.py' to train the chat model.\n")
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nSUB ai: Goodbye! Have a great day!")
                break
            
            # Get response
            result = chat_ai.chat(user_input)
            
            print(f"SUB ai: {result['response']}")
            print()
            
        except KeyboardInterrupt:
            print("\n\nSUB ai: Goodbye! Have a great day!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    interactive_chat()

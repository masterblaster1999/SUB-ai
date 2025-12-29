#!/usr/bin/env python3
"""
SUB ai - Transformer Chat Model Training

Trains a transformer-based chat model (GPT-2/DistilGPT-2) that can be converted to GGUF format.
This replaces the LSTM model with a modern transformer architecture.
"""

import os
import json
import argparse
from datetime import datetime

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import load_dataset, Dataset
except ImportError as e:
    print(f"Error: Missing dependencies. Please install them:")
    print("pip install -r requirements-gguf.txt")
    print(f"\nSpecific error: {e}")
    exit(1)


class TransformerChatTrainer:
    """
    Trainer for transformer-based chat models compatible with GGUF conversion.
    """
    
    def __init__(self, 
                 model_name="distilgpt2",
                 output_dir="models/transformer_chat",
                 max_length=128):
        """
        Initialize the trainer.
        
        Args:
            model_name: Base model to fine-tune (distilgpt2, gpt2, etc.)
            output_dir: Directory to save the trained model
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        
        print(f"\n{'='*60}")
        print(f"SUB ai - Transformer Chat Training")
        print(f"{'='*60}")
        print(f"Base Model: {model_name}")
        print(f"Output: {output_dir}")
        print(f"Max Length: {max_length}")
        print()
        
        # Load tokenizer and model
        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"\u2713 Loaded {model_name}")
        print(f"  Parameters: {self.model.num_parameters():,}")
        print()
    
    def prepare_dataset(self, dataset_name="daily_dialog", max_samples=5000):
        """
        Prepare chat dataset for training.
        
        Args:
            dataset_name: HuggingFace dataset name or 'local'
            max_samples: Maximum number of samples to use
        """
        print(f"Loading dataset: {dataset_name}...")
        
        if dataset_name == "daily_dialog":
            try:
                # Try loading with trust_remote_code
                print("Attempting to load daily_dialog dataset...")
                dataset = load_dataset("li2017dailydialog/daily_dialog", split="train", trust_remote_code=True)
                
                # Convert to chat format
                conversations = []
                for example in dataset:
                    dialog = example['dialog']
                    # Create conversation pairs
                    for i in range(len(dialog) - 1):
                        conversations.append({
                            'text': f"User: {dialog[i]}\nAssistant: {dialog[i+1]}"
                        })
                
                # Limit samples
                if max_samples and len(conversations) > max_samples:
                    conversations = conversations[:max_samples]
                
                dataset = Dataset.from_list(conversations)
                print(f"\u2713 Loaded {len(dataset)} conversation pairs from daily_dialog")
                
            except Exception as e:
                print(f"Failed to load daily_dialog: {e}")
                print("Falling back to built-in conversation data...")
                conversations = self._get_builtin_conversations()
                if max_samples:
                    conversations = conversations[:max_samples]
                dataset = Dataset.from_list(conversations)
                print(f"\u2713 Loaded {len(dataset)} built-in conversation pairs")
        
        elif dataset_name == "local":
            print("Using built-in conversation data...")
            conversations = self._get_builtin_conversations()
            if max_samples:
                conversations = conversations[:max_samples]
            dataset = Dataset.from_list(conversations)
            print(f"\u2713 Loaded {len(dataset)} built-in conversation pairs")
        
        else:
            print(f"Unknown dataset: {dataset_name}")
            exit(1)
        
        # Tokenize
        print("Tokenizing dataset...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.max_length,
                padding='max_length'
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        print("\u2713 Dataset tokenized")
        print()
        
        return tokenized_dataset
    
    def _get_builtin_conversations(self):
        """
        Get built-in conversation data as fallback.
        """
        base_conversations = [
            {'text': 'User: Hello!\nAssistant: Hi there! How can I help you today?'},
            {'text': 'User: What is AI?\nAssistant: AI stands for Artificial Intelligence. It refers to computer systems that can perform tasks that typically require human intelligence.'},
            {'text': 'User: How are you?\nAssistant: I\'m doing well, thank you for asking! How are you doing today?'},
            {'text': 'User: What can you do?\nAssistant: I can chat with you, answer questions, and help with various tasks. What would you like to know?'},
            {'text': 'User: Tell me a joke\nAssistant: Why did the programmer quit his job? Because he didn\'t get arrays! Would you like to hear another one?'},
            {'text': 'User: What is Python?\nAssistant: Python is a high-level, interpreted programming language known for its simplicity and versatility.'},
            {'text': 'User: Good morning\nAssistant: Good morning! I hope you have a wonderful day ahead. How can I assist you?'},
            {'text': 'User: Thank you\nAssistant: You\'re very welcome! Feel free to ask if you need anything else.'},
            {'text': 'User: Goodbye\nAssistant: Goodbye! Have a great day and come back anytime you need help!'},
            {'text': 'User: What is machine learning?\nAssistant: Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.'},
            {'text': 'User: Tell me about yourself\nAssistant: I\'m SUB ai, an AI assistant designed to help with conversations and answer questions. I\'m here to assist you!'},
            {'text': 'User: What is deep learning?\nAssistant: Deep learning is a type of machine learning that uses neural networks with multiple layers to learn complex patterns.'},
            {'text': 'User: Can you help me?\nAssistant: Of course! I\'d be happy to help. What do you need assistance with?'},
            {'text': 'User: What time is it?\nAssistant: I don\'t have access to real-time information, but you can check your device\'s clock for the current time.'},
            {'text': 'User: How do I learn programming?\nAssistant: Start with the basics, practice regularly, work on projects, and don\'t be afraid to make mistakes. Python is a great language for beginners!'},
        ]
        # Repeat to get enough samples
        return base_conversations * 350  # This gives ~5250 samples
    
    def train(self, dataset, epochs=3, batch_size=8, learning_rate=5e-5):
        """
        Fine-tune the model on the dataset.
        
        Args:
            dataset: Tokenized dataset
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        print(f"Starting training...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Learning Rate: {learning_rate}")
        print()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            report_to="none",  # Disable wandb/tensorboard
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal language modeling (not masked)
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("Training...")
        trainer.train()
        print("\n\u2713 Training complete!")
        print()
        
        return trainer
    
    def save_model(self):
        """
        Save the trained model and tokenizer.
        """
        print(f"Saving model to {self.output_dir}...")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'trained_at': datetime.now().isoformat(),
            'parameters': self.model.num_parameters()
        }
        
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\u2713 Model saved to {self.output_dir}")
        print()
        print("Next steps:")
        print("1. Test the model: python chat_transformer.py")
        print("2. Convert to GGUF: python convert_to_gguf.py")
        print()


def main():
    parser = argparse.ArgumentParser(description='Train transformer chat model')
    parser.add_argument('--model', type=str, default='distilgpt2',
                      help='Base model (distilgpt2, gpt2, etc.)')
    parser.add_argument('--dataset', type=str, default='daily_dialog',
                      help='Dataset name')
    parser.add_argument('--max-samples', type=int, default=5000,
                      help='Max training samples')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--max-length', type=int, default=128,
                      help='Max sequence length')
    parser.add_argument('--output', type=str, default='models/transformer_chat',
                      help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = TransformerChatTrainer(
        model_name=args.model,
        output_dir=args.output,
        max_length=args.max_length
    )
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(
        dataset_name=args.dataset,
        max_samples=args.max_samples
    )
    
    # Train
    trainer.train(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save
    trainer.save_model()
    
    print("="*60)
    print("Training complete! \U0001f389")
    print("="*60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SUB ai - Chat with GGUF Model

Interactive chat interface using GGUF format models via llama-cpp-python.
"""

import os
import sys
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python not installed")
    print("\nInstall with:")
    print("  pip install llama-cpp-python")
    print("\nOr with GPU support:")
    print("  CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python")
    sys.exit(1)


class GGUFChat:
    """
    Chat interface for GGUF models.
    """
    
    def __init__(self, model_path, n_ctx=512, n_threads=4):
        """
        Initialize GGUF chat.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_threads: Number of CPU threads to use
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            print(f"Error: Model not found at {model_path}")
            print("\nPlease download a GGUF model first.")
            sys.exit(1)
        
        print(f"Loading model: {self.model_path.name}...")
        print(f"Context: {n_ctx} tokens, Threads: {n_threads}")
        print()
        
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False
        )
        
        print("✓ Model loaded successfully!")
        print()
    
    def chat(self, prompt, max_tokens=100, temperature=0.7):
        """
        Generate a chat response.
        
        Args:
            prompt: User input
            max_tokens: Maximum response length
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Generated response text
        """
        # Format prompt for chat
        formatted_prompt = f"User: {prompt}\nAssistant:"
        
        # Generate
        output = self.llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["User:", "\n\n"],
            echo=False
        )
        
        response = output['choices'][0]['text'].strip()
        return response


def interactive_chat():
    """
    Start interactive chat session.
    """
    print("="*60)
    print("SUB ai - GGUF Chat Interface")
    print("="*60)
    print()
    
    # Find GGUF model
    gguf_dir = Path("models/gguf")
    
    if not gguf_dir.exists():
        print("Error: No GGUF models found.")
        print("\nPlease:")
        print("1. Train and convert a model:")
        print("   python train_transformer_chat.py")
        print("   python convert_to_gguf.py")
        print("\n2. Or download from releases:")
        print("   https://github.com/subhobhai943/SUB-ai/releases")
        sys.exit(1)
    
    # List available models
    models = list(gguf_dir.glob("*.gguf"))
    
    if not models:
        print("Error: No GGUF models found in models/gguf/")
        sys.exit(1)
    
    if len(models) == 1:
        model_path = models[0]
    else:
        print("Available models:")
        for i, model in enumerate(models, 1):
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"  {i}. {model.name} ({size_mb:.1f} MB)")
        
        choice = input("\nSelect model (1): ").strip() or "1"
        model_path = models[int(choice) - 1]
    
    # Initialize chat
    chat = GGUFChat(model_path)
    
    print("Type 'quit' or 'exit' to end the conversation.")
    print("Type 'clear' to clear chat history.")
    print()
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nSUB ai: Goodbye! Have a great day!")
                break
            
            if user_input.lower() == 'clear':
                print("\n" * 50)  # Clear screen
                print("Chat history cleared.\n")
                continue
            
            # Get response
            print("SUB ai: ", end="", flush=True)
            response = chat.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nSUB ai: Goodbye! Have a great day!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    interactive_chat()

#!/usr/bin/env python3
"""
SUB ai - Unified AI Interface
Combines number detection and chat capabilities.
"""

import sys
import os
from number_detector import NumberDetector
from chat_ai import SUBChatAI

class SUBai:
    """
    Main SUB ai class - unified interface for all capabilities.
    """
    
    def __init__(self):
        """
        Initialize SUB ai with all modules.
        """
        print("Initializing SUB ai...")
        
        # Load number detector
        number_model = 'models/sub_ai_model_latest.h5'
        self.number_detector = NumberDetector(
            model_path=number_model if os.path.exists(number_model) else None
        )
        
        # Load chat AI
        chat_model = 'models/sub_ai_chat_latest.h5'
        chat_vocab = 'models/chat_vocab.pkl'
        self.chat_ai = SUBChatAI(
            model_path=chat_model if os.path.exists(chat_model) else None,
            vocab_path=chat_vocab if os.path.exists(chat_vocab) else None
        )
        
        print("SUB ai initialized!\n")
    
    def detect_number(self, image_path):
        """
        Detect numbers from an image.
        
        Args:
            image_path (str): Path to image
            
        Returns:
            dict: Detection result
        """
        return self.number_detector.detect(image_path)
    
    def chat(self, message, temperature=0.7):
        """
        Chat with SUB ai.
        
        Args:
            message (str): User message
            temperature (float): Response randomness
            
        Returns:
            dict: Chat response
        """
        return self.chat_ai.chat(message, temperature=temperature)
    
    def interactive_mode(self):
        """
        Start interactive mode with both capabilities.
        """
        print("="*60)
        print("SUB ai - Interactive Mode")
        print("="*60)
        print("\nCapabilities:")
        print("  1. Chat: Just type your message")
        print("  2. Detect numbers: Type 'detect <image_path>'")
        print("\nCommands:")
        print("  - 'quit' or 'exit': End session")
        print("  - 'help': Show this help")
        print("="*60)
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\nSUB ai: Goodbye! Have a great day! 👋")
                    break
                
                # Check for help
                if user_input.lower() == 'help':
                    print("\nSUB ai: I can chat with you and detect numbers from images!")
                    print("  - To chat: Just type normally")
                    print("  - To detect: Type 'detect <image_path>'\n")
                    continue
                
                # Check for detection command
                if user_input.lower().startswith('detect '):
                    image_path = user_input[7:].strip()
                    if os.path.exists(image_path):
                        result = self.detect_number(image_path)
                        print(f"\nSUB ai: {result['message']}")
                        if result.get('predicted_digit') is not None:
                            print(f"  Digit: {result['predicted_digit']}")
                        print(f"  Confidence: {result['confidence']:.2%}\n")
                    else:
                        print(f"\nSUB ai: Sorry, I can't find the image at '{image_path}'\n")
                    continue
                
                # Regular chat
                response = self.chat(user_input)
                print(f"\nSUB ai: {response['response']}\n")
                
            except KeyboardInterrupt:
                print("\n\nSUB ai: Goodbye! Have a great day! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
                continue


def main():
    """
    Main entry point for SUB ai.
    """
    # Check if models exist
    number_model_exists = os.path.exists('models/sub_ai_model_latest.h5')
    chat_model_exists = os.path.exists('models/sub_ai_chat_latest.h5')
    
    if not number_model_exists:
        print("⚠️  Number detection model not found.")
        print("   Run: python train.py\n")
    
    if not chat_model_exists:
        print("⚠️  Chat model not found. Using rule-based responses.")
        print("   Run: python train_chat.py\n")
    
    # Initialize SUB ai
    ai = SUBai()
    
    # Start interactive mode
    ai.interactive_mode()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SUB ai - Convert Transformer Model to GGUF Format

Converts a trained transformer model (PyTorch/HuggingFace) to GGUF format
for efficient inference with llama.cpp.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    print("Error: Missing dependencies. Install with:")
    print("pip install torch transformers")
    sys.exit(1)


class GGUFConverter:
    """
    Convert HuggingFace transformer models to GGUF format.
    """
    
    def __init__(self, model_dir, output_dir="models/gguf"):
        """
        Initialize converter.
        
        Args:
            model_dir: Directory containing trained model
            output_dir: Output directory for GGUF files
        """
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("SUB ai - GGUF Converter")
        print(f"{'='*60}")
        print(f"Model: {self.model_dir}")
        print(f"Output: {self.output_dir}")
        print()
    
    def check_llama_cpp(self):
        """
        Check if llama.cpp is available and clone if needed.
        """
        llama_dir = Path("llama.cpp")
        
        if llama_dir.exists():
            print("✓ llama.cpp already available")
            return llama_dir
        
        print("Cloning llama.cpp...")
        import subprocess
        
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
                check=True,
                capture_output=True
            )
            print("✓ llama.cpp cloned")
            return llama_dir
        except Exception as e:
            print(f"Error cloning llama.cpp: {e}")
            print("\nPlease clone manually:")
            print("git clone https://github.com/ggerganov/llama.cpp.git")
            sys.exit(1)
    
    def convert_to_gguf(self, quantization="f16"):
        """
        Convert model to GGUF format.
        
        Args:
            quantization: Quantization type (f16, q8_0, q5_k_m, q4_k_m)
        """
        print(f"Converting to GGUF (quantization: {quantization})...")
        print()
        
        # Check llama.cpp
        llama_dir = self.check_llama_cpp()
        convert_script = llama_dir / "convert_hf_to_gguf.py"
        
        if not convert_script.exists():
            # Try old script name
            convert_script = llama_dir / "convert.py"
            if not convert_script.exists():
                print("Error: Conversion script not found in llama.cpp")
                print("Please update llama.cpp: cd llama.cpp && git pull")
                sys.exit(1)
        
        # Step 1: Convert to GGUF F16 (base)
        import subprocess
        
        base_output = self.output_dir / "sub_ai_chat_f16.gguf"
        
        print("Step 1: Converting to GGUF F16 format...")
        try:
            cmd = [
                sys.executable,
                str(convert_script),
                str(self.model_dir),
                "--outfile", str(base_output),
                "--outtype", "f16"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            print(f"✓ Created: {base_output}")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print(e.stderr)
            sys.exit(1)
        
        # Step 2: Quantize if requested
        if quantization != "f16":
            print(f"\nStep 2: Quantizing to {quantization.upper()}...")
            
            quantize_bin = llama_dir / "llama-quantize"
            if not quantize_bin.exists():
                quantize_bin = llama_dir / "quantize"
            
            if not quantize_bin.exists():
                print("Warning: quantize binary not found")
                print("Please build llama.cpp first:")
                print("  cd llama.cpp && make")
                print(f"\nUsing F16 version: {base_output}")
                return base_output
            
            quant_output = self.output_dir / f"sub_ai_chat_{quantization}.gguf"
            
            try:
                cmd = [
                    str(quantize_bin),
                    str(base_output),
                    str(quant_output),
                    quantization.upper()
                ]
                
                print(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(result.stdout)
                print(f"✓ Created: {quant_output}")
                
                return quant_output
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
                print(e.stderr)
                print(f"\nUsing F16 version: {base_output}")
                return base_output
        
        return base_output
    
    def get_model_info(self):
        """
        Get information about the converted models.
        """
        print("\n" + "="*60)
        print("Conversion Complete! 🎉")
        print("="*60)
        print("\nGenerated GGUF files:")
        
        for gguf_file in self.output_dir.glob("*.gguf"):
            size_mb = gguf_file.stat().st_size / (1024 * 1024)
            print(f"  {gguf_file.name}: {size_mb:.1f} MB")
        
        print("\nUsage:")
        print("\n1. With llama.cpp:")
        print("   cd llama.cpp")
        print("   ./llama-cli -m ../models/gguf/sub_ai_chat_q4_k_m.gguf -p 'Hello!'")
        
        print("\n2. With Python (llama-cpp-python):")
        print("   pip install llama-cpp-python")
        print("   python chat_gguf.py")
        
        print("\n3. With LM Studio:")
        print("   - Open LM Studio")
        print("   - Load models/gguf/sub_ai_chat_q4_k_m.gguf")
        
        print("\n4. With Ollama:")
        print("   - Create Modelfile")
        print("   - ollama create sub-ai -f Modelfile")
        print("   - ollama run sub-ai")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Convert transformer model to GGUF format'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='models/transformer_chat',
        help='Path to trained model directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/gguf',
        help='Output directory for GGUF files'
    )
    parser.add_argument(
        '--quantize',
        type=str,
        default='q4_k_m',
        choices=['f16', 'q8_0', 'q5_k_m', 'q4_k_m', 'q4_0'],
        help='Quantization type'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("\nPlease train the model first:")
        print("  python train_transformer_chat.py")
        sys.exit(1)
    
    # Convert
    converter = GGUFConverter(args.model, args.output)
    converter.convert_to_gguf(quantization=args.quantize)
    converter.get_model_info()


if __name__ == "__main__":
    main()

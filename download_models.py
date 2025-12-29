#!/usr/bin/env python3
"""
SUB ai - Model Downloader
Downloads the latest trained models from GitHub Releases.
"""

import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

REPO_OWNER = "subhobhai943"
REPO_NAME = "SUB-ai"
MODELS_DIR = "models"

def get_latest_release(tag_prefix):
    """
    Get the latest release with a specific tag prefix.
    
    Args:
        tag_prefix (str): Tag prefix to search for (e.g., 'number-model' or 'chat-model')
    
    Returns:
        dict: Release information or None
    """
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases"
    
    try:
        with urllib.request.urlopen(api_url) as response:
            releases = json.loads(response.read())
        
        # Find the latest release with matching tag prefix
        for release in releases:
            if release['tag_name'].startswith(tag_prefix):
                return release
        
        return None
    except urllib.error.URLError as e:
        print(f"Error fetching releases: {e}")
        return None

def download_file(url, filepath):
    """
    Download a file from URL.
    
    Args:
        url (str): URL to download from
        filepath (str): Local path to save file
    
    Returns:
        bool: True if successful
    """
    try:
        print(f"Downloading: {os.path.basename(filepath)}...")
        urllib.request.urlretrieve(url, filepath)
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✅ Downloaded: {os.path.basename(filepath)} ({file_size:.2f} MB)")
        return True
    except Exception as e:
        print(f"❌ Error downloading {os.path.basename(filepath)}: {e}")
        return False

def download_number_model():
    """
    Download the latest number detection model.
    
    Returns:
        bool: True if successful
    """
    print("\n" + "="*60)
    print("Downloading Number Detection Model")
    print("="*60)
    
    release = get_latest_release('number-model')
    
    if not release:
        print("❌ No number detection model release found.")
        print("   Run training: python train.py")
        print("   Or run workflow: GitHub Actions > Train SUB ai Model")
        return False
    
    print(f"\nFound Release: {release['name']}")
    print(f"Published: {release['published_at']}")
    print(f"Tag: {release['tag_name']}\n")
    
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Download model files
    success = True
    for asset in release['assets']:
        if asset['name'].endswith('.h5') or asset['name'].endswith('.png'):
            filepath = os.path.join(MODELS_DIR, asset['name'])
            if not download_file(asset['browser_download_url'], filepath):
                success = False
    
    if success:
        print("\n✅ Number detection model downloaded successfully!")
        print(f"   Model location: {MODELS_DIR}/sub_ai_model_latest.h5")
        print("\nTest it: python test_detector.py")
    
    return success

def download_chat_model():
    """
    Download the latest chat model.
    
    Returns:
        bool: True if successful
    """
    print("\n" + "="*60)
    print("Downloading Chat Model")
    print("="*60)
    
    release = get_latest_release('chat-model')
    
    if not release:
        print("❌ No chat model release found.")
        print("   Run training: python train_chat.py")
        print("   Or run workflow: GitHub Actions > Train SUB ai Chat Model")
        return False
    
    print(f"\nFound Release: {release['name']}")
    print(f"Published: {release['published_at']}")
    print(f"Tag: {release['tag_name']}\n")
    
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Download model files
    success = True
    for asset in release['assets']:
        if asset['name'].endswith('.h5') or asset['name'].endswith('.pkl'):
            filepath = os.path.join(MODELS_DIR, asset['name'])
            if not download_file(asset['browser_download_url'], filepath):
                success = False
    
    if success:
        print("\n✅ Chat model downloaded successfully!")
        print(f"   Model location: {MODELS_DIR}/sub_ai_chat_latest.h5")
        print(f"   Vocab location: {MODELS_DIR}/chat_vocab.pkl")
        print("\nTest it: python chat_ai.py")
    
    return success

def download_all_models():
    """
    Download both number detection and chat models.
    
    Returns:
        bool: True if both successful
    """
    print("\n" + "="*60)
    print("SUB ai - Model Downloader")
    print("="*60)
    print(f"\nRepository: {REPO_OWNER}/{REPO_NAME}")
    print(f"Downloading models from GitHub Releases...\n")
    
    success1 = download_number_model()
    success2 = download_chat_model()
    
    if success1 and success2:
        print("\n" + "="*60)
        print("✅ All models downloaded successfully!")
        print("="*60)
        print("\nRun SUB ai: python sub_ai.py")
        return True
    else:
        print("\n" + "="*60)
        print("⚠️  Some models failed to download")
        print("="*60)
        print("\nYou can train models locally:")
        print("  python train.py        # Number detection")
        print("  python train_chat.py   # Chat model")
        return False

def main():
    """
    Main entry point.
    """
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
        
        if model_type in ['number', 'num', 'detection']:
            download_number_model()
        elif model_type in ['chat', 'conversation']:
            download_chat_model()
        elif model_type in ['all', 'both']:
            download_all_models()
        else:
            print(f"Unknown model type: {model_type}")
            print("\nUsage:")
            print("  python download_models.py          # Download all models")
            print("  python download_models.py number   # Number detection only")
            print("  python download_models.py chat     # Chat model only")
            print("  python download_models.py all      # All models")
    else:
        download_all_models()

if __name__ == "__main__":
    main()

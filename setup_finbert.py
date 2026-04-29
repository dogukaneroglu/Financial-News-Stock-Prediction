"""
Setup script for FinBERT model.
Downloads and tests the FinBERT model for financial sentiment analysis.
"""

import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['transformers', 'torch']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall them with:")
        print("  pip install transformers torch")
        return False
    
    return True


def download_finbert():
    """Download FinBERT model."""
    print("="*60)
    print("FinBERT Model Setup")
    print("="*60)
    
    if not check_dependencies():
        return False
    
    print("\nImporting libraries...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    
    print("\nDownloading FinBERT model from HuggingFace...")
    print("Model: ProsusAI/finbert")
    print("Size: ~440MB (this may take a few minutes)")
    
    try:
        # Download tokenizer
        print("\n1. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        print("   [OK] Tokenizer downloaded")
        
        # Download model
        print("\n2. Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        print("   [OK] Model downloaded")
        
        # Test model
        print("\n3. Testing model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # Test with sample text
        test_text = "Apple reports record quarterly earnings beating expectations"
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        
        print(f"   Test input: '{test_text}'")
        print(f"   Positive: {probs[0].item():.3f}")
        print(f"   Negative: {probs[1].item():.3f}")
        print(f"   Neutral: {probs[2].item():.3f}")
        print("   [OK] Model working correctly")
        
        print(f"\n[SUCCESS] FinBERT setup complete!")
        print(f"  Device: {device}")
        print(f"  Model cached in: ~/.cache/huggingface/transformers/")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error during setup: {str(e)}")
        return False


def main():
    """Main setup function."""
    print("\nThis script will download the FinBERT model for financial sentiment analysis.")
    print("The model will be cached locally for future use.\n")
    
    success = download_finbert()
    
    if success:
        print("\n" + "="*60)
        print("Setup Complete!")
        print("="*60)
        print("\nYou can now use FinBERT in your project:")
        print("\n  from preprocessing.nlp_processor import NLPProcessor")
        print("  processor = NLPProcessor(method='finbert')")
        print("\nOr run the demo:")
        print("  python src/preprocessing/nlp_processor.py")
    else:
        print("\n" + "="*60)
        print("Setup Failed")
        print("="*60)
        print("\nPlease install required packages and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

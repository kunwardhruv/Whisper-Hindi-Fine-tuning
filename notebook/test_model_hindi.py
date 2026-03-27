#!/usr/bin/env python3
"""
Test Fine-tuned Whisper Model for Hindi
======================================

This script tests your fine-tuned Whisper model specifically for Hindi.
"""

import sys
import os
import torch
import librosa

# Set environment variable to avoid TensorFlow issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import Whisper components directly
from transformers.models.whisper import WhisperProcessor, WhisperForConditionalGeneration

def load_model(model_path="../models/whisper-quick-trained"):
    """Load the fine-tuned model and processor."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        print("Please train the model first using quick_train.py")
        return None, None
    
    print(f"[INFO] Loading model from {model_path}...")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # Set to evaluation mode
    model.eval()
    
    print("[SUCCESS] Model loaded successfully!")
    return processor, model

def transcribe_audio_hindi(audio_path, processor, model):
    """Transcribe an audio file using the fine-tuned model with Hindi language setting."""
    if not os.path.exists(audio_path):
        print(f"[ERROR] Audio file not found: {audio_path}")
        return None
    
    print(f"[INFO] Processing audio: {audio_path}")
    
    try:
        # Load and preprocess audio
        audio_array, sr = librosa.load(audio_path, sr=16000)
        
        # Extract features
        input_features = processor.feature_extractor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).input_features
        
        # Generate transcription with Hindi language forced
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="hindi",  # Force Hindi language
                task="transcribe",  # Force transcription (not translation)
                max_length=200,  # Limit output length
                num_beams=1,  # Faster generation
                do_sample=False,  # Deterministic output
            )
        
        # Decode the transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
        
    except Exception as e:
        print(f"[ERROR] Error transcribing audio: {e}")
        return None

def compare_with_original(audio_file):
    """Compare transcription with original text if available."""
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    original_path = f"../data/transcripts_txt/{base_name}.txt"
    
    if os.path.exists(original_path):
        try:
            with open(original_path, 'r', encoding='utf-8') as f:
                original_text = f.read().strip()
            
            # Show first 200 characters of original
            print(f"\n[ORIGINAL] First 200 chars:")
            print(f"{'='*50}")
            print(original_text[:200] + "..." if len(original_text) > 200 else original_text)
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"[WARNING] Could not read original transcript: {e}")

def main():
    """Main function to test the model."""
    if len(sys.argv) != 2:
        print("Usage: python test_model_hindi.py <audio_file_path>")
        print("\nExample:")
        print("  python test_model_hindi.py ../data/audio/238079.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # Load model
    processor, model = load_model()
    if processor is None or model is None:
        sys.exit(1)
    
    # Show original transcript first
    compare_with_original(audio_path)
    
    # Transcribe audio
    transcription = transcribe_audio_hindi(audio_path, processor, model)
    
    if transcription:
        print(f"\n[RESULT] Model Transcription:")
        print(f"{'='*50}")
        print(transcription)
        print(f"{'='*50}")
        
        # Basic quality check
        if len(transcription.strip()) > 10:
            print("\n[INFO] Transcription looks reasonable (length > 10 chars)")
        else:
            print("\n[WARNING] Transcription seems too short")
            
    else:
        print("[ERROR] Failed to transcribe audio")
        sys.exit(1)

if __name__ == "__main__":
    main()
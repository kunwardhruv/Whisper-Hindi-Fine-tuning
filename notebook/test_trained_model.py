#!/usr/bin/env python3
"""
Test Fine-tuned Whisper Model
=============================

This script tests your fine-tuned Whisper model on audio files.

Usage:
    python test_trained_model.py <audio_file_path>
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
        print("Please train the model first using train_whisper_model.py")
        return None, None
    
    print(f"[INFO] Loading model from {model_path}...")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # Set to evaluation mode
    model.eval()
    
    print("[SUCCESS] Model loaded successfully!")
    return processor, model

def transcribe_audio(audio_path, processor, model):
    """Transcribe an audio file using the fine-tuned model."""
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
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        # Decode the transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
        
    except Exception as e:
        print(f"[ERROR] Error transcribing audio: {e}")
        return None

def main():
    """Main function to test the model."""
    if len(sys.argv) != 2:
        print("Usage: python test_trained_model.py <audio_file_path>")
        print("\nExample:")
        print("  python test_trained_model.py ../data/audio/825780.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # Load model
    processor, model = load_model()
    if processor is None or model is None:
        sys.exit(1)
    
    # Transcribe audio
    transcription = transcribe_audio(audio_path, processor, model)
    
    if transcription:
        print(f"\n[RESULT] Transcription Result:")
        print(f"{'='*50}")
        print(transcription)
        print(f"{'='*50}")
    else:
        print("[ERROR] Failed to transcribe audio")
        sys.exit(1)

if __name__ == "__main__":
    main()
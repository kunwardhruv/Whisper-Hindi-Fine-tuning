#!/usr/bin/env python3
"""
Final Evaluation for Assignment Submission
=========================================

This provides the final WER numbers for your Josh Talks assignment.
"""

import os
import sys
import torch
import librosa
import jiwer
from tqdm import tqdm

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers.models.whisper import WhisperProcessor, WhisperForConditionalGeneration

def evaluate_sample_files():
    """Quick evaluation on a few sample files for assignment."""
    
    print("FINAL EVALUATION FOR JOSH TALKS ASSIGNMENT")
    print("=" * 50)
    
    # Load models
    print("\n1. Loading Models...")
    baseline_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    baseline_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    baseline_model.eval()
    
    ft_processor = WhisperProcessor.from_pretrained("models/whisper-quick-trained")
    ft_model = WhisperForConditionalGeneration.from_pretrained("models/whisper-quick-trained")
    ft_model.eval()
    
    print("✓ Models loaded successfully")
    
    # Test on a few sample files
    test_files = [
        "238079.wav", "240907.wav", "269383.wav", "301057.wav", "350297.wav"
    ]
    
    baseline_predictions = []
    ft_predictions = []
    references = []
    
    print("\n2. Running Evaluation...")
    
    for filename in test_files:
        audio_path = f"data/audio/{filename}"
        txt_path = f"data/transcripts_txt/{filename.replace('.wav', '.txt')}"
        
        if not os.path.exists(audio_path) or not os.path.exists(txt_path):
            continue
            
        # Load audio
        audio_array, sr = librosa.load(audio_path, sr=16000)
        
        # Load reference
        with open(txt_path, 'r', encoding='utf-8') as f:
            reference = f.read().strip()
        
        # Limit reference length for fair comparison
        reference = " ".join(reference.split()[:50])  # First 50 words
        
        # Get baseline prediction
        input_features = baseline_processor.feature_extractor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).input_features
        
        with torch.no_grad():
            baseline_ids = baseline_model.generate(
                input_features, language="hindi", task="transcribe", max_length=150
            )
        baseline_pred = baseline_processor.batch_decode(baseline_ids, skip_special_tokens=True)[0]

        # Get fine-tuned prediction (original behavior)
        input_features = ft_processor.feature_extractor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).input_features
        
        with torch.no_grad():
            ft_ids = ft_model.generate(
                input_features, language="hindi", task="transcribe", max_length=150
            )
        ft_pred = ft_processor.batch_decode(ft_ids, skip_special_tokens=True)[0]

        # Get fine-tuned prediction with decoding fixes (no repeat ngrams, beam search)
        with torch.no_grad():
            ft_ids_fix = ft_model.generate(
                input_features,
                language="hindi",
                task="transcribe",
                max_length=150,
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True,
            )
        ft_pred_fix = ft_processor.batch_decode(ft_ids_fix, skip_special_tokens=True)[0]

        def dedupe_repeated_tokens(text):
            tokens = text.split()
            out = []
            for t in tokens:
                if len(out) >= 3 and out[-1] == t and out[-2] == t and out[-3] == t:
                    continue
                out.append(t)
            return " ".join(out)

        ft_pred_fix = dedupe_repeated_tokens(ft_pred_fix)

        baseline_predictions.append(baseline_pred)
        ft_predictions.append(ft_pred)
        references.append(reference)

        # Only store fix predictions for comparison later
        if not hasattr(evaluate_sample_files, "ft_predictions_fix"):
            evaluate_sample_files.ft_predictions_fix = []
        evaluate_sample_files.ft_predictions_fix.append(ft_pred_fix)
        
        print(f"✓ Processed {filename}")
    
    # Calculate WER
    if len(references) > 0:
        baseline_wer = jiwer.wer(references, baseline_predictions)
        ft_wer = jiwer.wer(references, ft_predictions)
        ft_wer_fix = jiwer.wer(references, evaluate_sample_files.ft_predictions_fix)

        print("\n3. FINAL RESULTS")
        print("=" * 50)
        print(f"Baseline Whisper-small WER: {baseline_wer:.3f}")
        print(f"Fine-tuned Whisper-small WER: {ft_wer:.3f}")
        print(f"Fine-tuned with decoding fix WER: {ft_wer_fix:.3f}")
        print(f"Baseline -> FT improvement: {((baseline_wer - ft_wer) / baseline_wer * 100):.1f}%")
        print(f"FT -> FT+fix improvement: {((ft_wer - ft_wer_fix) / ft_wer * 100):.1f}%")

        # For assignment table
        print("\n4. FOR YOUR ASSIGNMENT TABLE:")
        print("=" * 50)
        print("| Model | Hindi WER |")
        print("|-------|-----------|")
        print(f"| Whisper Small (Pretrained) | {baseline_wer:.3f} |")
        print(f"| FT Whisper Small (yours) | {ft_wer:.3f} |")
        print(f"| FT (decoding fix) | {ft_wer_fix:.3f} |")

        return baseline_wer, ft_wer, ft_wer_fix

    return None, None, None

if __name__ == "__main__":
    evaluate_sample_files()
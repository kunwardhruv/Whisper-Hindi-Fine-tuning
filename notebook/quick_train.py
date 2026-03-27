#!/usr/bin/env python3
"""
Quick Whisper Training - No Preprocessing Required
=================================================

This script trains Whisper directly from your audio files without 
complex preprocessing steps that can hang.
"""

import os
import sys
import torch
import librosa
import numpy as np
from pathlib import Path
import json

# Disable TensorFlow
os.environ['USE_TF'] = 'NO'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def check_data():
    """Quick check if we have the basic data."""
    audio_dir = "../data/audio"
    transcript_dir = "../data/transcripts_txt"
    
    if not os.path.exists(audio_dir):
        print(f"[ERROR] Audio directory not found: {audio_dir}")
        return False
        
    if not os.path.exists(transcript_dir):
        print(f"[ERROR] Transcript directory not found: {transcript_dir}")
        return False
    
    audio_files = list(Path(audio_dir).glob("*.wav"))
    txt_files = list(Path(transcript_dir).glob("*.txt"))
    
    print(f"[INFO] Found {len(audio_files)} audio files and {len(txt_files)} transcript files")
    
    if len(audio_files) == 0 or len(txt_files) == 0:
        print("[ERROR] No audio or transcript files found")
        return False
        
    return True

def create_simple_dataset():
    """Create a simple dataset from audio and text files."""
    print("[INFO] Creating simple dataset...")
    
    audio_dir = Path("../data/audio")
    transcript_dir = Path("../data/transcripts_txt")
    
    # Find matching audio/text pairs
    pairs = []
    for audio_file in audio_dir.glob("*.wav"):
        txt_file = transcript_dir / f"{audio_file.stem}.txt"
        if txt_file.exists():
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                # Filter out empty transcripts, but keep most others
                if text and len(text.split()) < 1000:  # Limit to ~1000 words
                    # Truncate very long texts to first 100 words for training
                    if len(text.split()) > 100:
                        text = " ".join(text.split()[:100])
                    pairs.append({
                        "audio": str(audio_file),
                        "sentence": text
                    })
                elif len(text.split()) >= 1000:
                    print(f"[WARNING] Skipping extremely long transcript: {txt_file.name} ({len(text.split())} words)")
            except Exception as e:
                print(f"[WARNING] Skipping {txt_file}: {e}")
    
    print(f"[INFO] Found {len(pairs)} valid audio-text pairs")
    
    if len(pairs) < 5:
        print("[ERROR] Not enough valid pairs for training")
        return None
    
    # Split into train/test (90/10)
    split_idx = int(len(pairs) * 0.9)
    train_data = pairs[:split_idx]
    test_data = pairs[split_idx:]
    
    print(f"[INFO] Split: {len(train_data)} train, {len(test_data)} test")
    
    return {"train": train_data, "test": test_data}

def quick_train():
    """Quick training function."""
    print("[INFO] Starting quick Whisper training...")
    
    # Check data
    if not check_data():
        return False
    
    # Create dataset
    dataset = create_simple_dataset()
    if not dataset:
        return False
    
    try:
        # Import transformers
        from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
        
        print("[INFO] Loading Whisper model...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        
        # Simple data collator
        def collate_fn(batch):
            # Process batch
            input_features = []
            labels = []
            
            for item in batch:
                try:
                    # Load audio
                    audio_array, sr = librosa.load(item["audio"], sr=16000)
                    
                    # Get features
                    features = processor.feature_extractor(audio_array, sampling_rate=16000).input_features[0]
                    input_features.append(features)
                    
                    # Tokenize text (truncate if too long)
                    label = processor.tokenizer(
                        item["sentence"], 
                        max_length=400,  # Leave some room for special tokens
                        truncation=True
                    ).input_ids
                    labels.append(label)
                    
                except Exception as e:
                    print(f"[WARNING] Skipping item due to error: {e}")
                    continue
            
            if not input_features or not labels:
                # Return empty batch that trainer can handle
                return {
                    "input_features": torch.empty(0, 80, 3000),
                    "labels": torch.empty(0, 1, dtype=torch.long)
                }
            
            # Convert to tensors
            max_label_len = max(len(label) for label in labels) if labels else 1
            padded_labels = []
            for label in labels:
                padded = label + [-100] * (max_label_len - len(label))
                padded_labels.append(padded)
            
            return {
                "input_features": torch.tensor(input_features),
                "labels": torch.tensor(padded_labels)
            }
        
        # Create simple dataset class
        class SimpleDataset:
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        train_dataset = SimpleDataset(dataset["train"])
        eval_dataset = SimpleDataset(dataset["test"])
        
        print(f"[INFO] Training dataset size: {len(train_dataset)}")
        print(f"[INFO] Eval dataset size: {len(eval_dataset)}")
        
        # Training arguments - very conservative
        training_args = TrainingArguments(
            output_dir="../models/whisper-quick-trained",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=5e-6,
            logging_steps=5,
            save_steps=20,
            eval_steps=20,
            eval_strategy="steps",  # Fixed parameter name
            fp16=False,
            dataloader_drop_last=False,
            remove_unused_columns=False,
            report_to=[],  # Disable wandb/tensorboard
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
        )
        
        print("[INFO] Starting training (this will take a while)...")
        trainer.train()
        
        # Save model
        output_dir = "../models/whisper-quick-trained"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        
        print(f"[SUCCESS] Training completed! Model saved to {output_dir}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Quick Whisper Training")
    print("=" * 30)
    
    success = quick_train()
    
    if success:
        print("\n[SUCCESS] Quick training completed!")
        print("Test your model with:")
        print("python test_trained_model.py ../data/audio/sample.wav")
    else:
        print("\n[ERROR] Training failed. Check the error messages above.")
        sys.exit(1)
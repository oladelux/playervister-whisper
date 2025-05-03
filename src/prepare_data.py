import os
import json
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm
import augment
import argparse

def load_transcript(transcript_path):
    """Load transcript from a text file"""
    with open(transcript_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def process_audio(audio_path, output_path=None, sample_rate=16000):
    """Process audio file to the format expected by Whisper"""
    # If output path not specified, create one
    if output_path is None:
        dir_name = os.path.dirname(audio_path)
        base_name = os.path.basename(audio_path)
        name, _ = os.path.splitext(base_name)
        output_path = os.path.join(dir_name, f"{name}_processed.wav")
    
    # For m4a files, use pydub
    _, ext = os.path.splitext(audio_path)
    if ext.lower() == '.m4a':
        try:
            audio = AudioSegment.from_file(audio_path, format="m4a")
            audio = audio.set_channels(1).set_frame_rate(sample_rate)
            audio.export(output_path, format="wav")
            return output_path
        except Exception as e:
            print(f"Error processing m4a: {e}")
            print("Falling back to librosa...")
    
    # For other formats or as fallback, use librosa
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        sf.write(output_path, y, sample_rate, 'PCM_16')
        return output_path
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def augment_audio(audio_path, output_dir, config, num_augmentations=5):
    """Create augmented versions of audio for better training"""
    augmented_paths = []
    
    # Get base filename without extension
    base_name = os.path.basename(audio_path)
    name, ext = os.path.splitext(base_name)
    
    # Load audio
    try:
        y, sr = librosa.load(audio_path, sr=config["sample_rate"], mono=True)
    except Exception as e:
        print(f"Error loading audio for augmentation: {e}")
        return augmented_paths
    
    for i in range(num_augmentations):
        aug_audio = y.copy()
        
        # Apply pitch shifting (randomly)
        if random.random() < 0.7:
            shift_steps = np.random.uniform(-config["pitch_shift_range"], config["pitch_shift_range"])
            aug_audio = librosa.effects.pitch_shift(aug_audio, sr=sr, n_steps=shift_steps)
        
        # Apply time stretching (randomly)
        if random.random() < 0.7:
            stretch_factor = np.random.uniform(1 - config["speed_range"], 1 + config["speed_range"])
            aug_audio = librosa.effects.time_stretch(aug_audio, rate=stretch_factor)
        
        # Add background noise (randomly)
        if random.random() < 0.7 and config["noise_level"] > 0:
            noise = np.random.randn(len(aug_audio)) * config["noise_level"]
            aug_audio = aug_audio + noise
            
        # Save augmented audio
        aug_path = os.path.join(output_dir, f"{name}_aug_{i}.wav")
        sf.write(aug_path, aug_audio, sr, 'PCM_16')
        augmented_paths.append(aug_path)
        
    return augmented_paths

def create_dataset(audio_dir, transcript_dir, output_dir, config):
    """Create dataset from audio and transcript files"""
    os.makedirs(output_dir, exist_ok=True)
    processed_dir = os.path.join(output_dir, "processed_audio")
    os.makedirs(processed_dir, exist_ok=True)
    
    if config["augmentation"]["enabled"]:
        augmented_dir = os.path.join(output_dir, "augmented_audio")
        os.makedirs(augmented_dir, exist_ok=True)
    
    # Find matching audio and transcript files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.m4a', '.mp3', '.wav', '.flac'))]
    
    dataset = []
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        # Determine transcript file name
        audio_name = os.path.splitext(audio_file)[0]
        transcript_file = f"{audio_name}.txt"
        transcript_path = os.path.join(transcript_dir, transcript_file)
        
        if not os.path.exists(transcript_path):
            print(f"Warning: No transcript found for {audio_file}")
            continue
        
        # Process audio
        audio_path = os.path.join(audio_dir, audio_file)
        processed_path = process_audio(
            audio_path, 
            os.path.join(processed_dir, f"{audio_name}.wav"), 
            config["sample_rate"]
        )
        
        if not processed_path:
            print(f"Warning: Failed to process {audio_file}")
            continue
        
        # Load transcript
        transcript = load_transcript(transcript_path)
        
        # Add original example to dataset
        dataset.append({
            "audio_path": processed_path,
            "text": transcript
        })
        
        # Create augmented versions if enabled
        if config["augmentation"]["enabled"]:
            augmented_paths = augment_audio(
                processed_path, 
                augmented_dir, 
                config, 
                config["augmentation"]["num_augmentations"]
            )
            
            for aug_path in augmented_paths:
                dataset.append({
                    "audio_path": aug_path,
                    "text": transcript
                })
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train, validation and test sets
    total_samples = len(df)
    train_size = int(total_samples * config["train_split"])
    val_size = int(total_samples * config["val_split"])
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Created dataset with {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test examples")
    
    return {
        "train_csv": os.path.join(output_dir, "train.csv"),
        "val_csv": os.path.join(output_dir, "validation.csv"),
        "test_csv": os.path.join(output_dir, "test.csv")
    }

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Whisper fine-tuning")
    parser.add_argument("--audio_dir", default="data/audio", help="Directory containing audio files")
    parser.add_argument("--transcript_dir", default="data/transcripts", help="Directory containing transcript files")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory for processed data")
    parser.add_argument("--config_path", default="configs/training_config.json", help="Path to config file")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # Create dataset
    dataset_paths = create_dataset(args.audio_dir, args.transcript_dir, args.output_dir, config["data"])
    
    print(f"Dataset preparation complete!")
    print(f"Train CSV: {dataset_paths['train_csv']}")
    print(f"Validation CSV: {dataset_paths['val_csv']}")
    print(f"Test CSV: {dataset_paths['test_csv']}")

if __name__ == "__main__":
    main() 
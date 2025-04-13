import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pydub import AudioSegment
import torch
import torchaudio
import librosa
import soundfile as sf
from difflib import SequenceMatcher
import string
from typing import List, Dict, Tuple, Union

def create_directory(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def normalize_text(text):
    """Normalize text for comparison"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except for periods
    text = re.sub(r'[^\w\s.]', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Replace multiple periods with a single period
    text = re.sub(r'\.+', '.', text)
    
    # Ensure there's a space after each period
    text = re.sub(r'\.([^\s])', '. \\1', text)
    
    # Trim
    text = text.strip()
    
    return text

def sequence_similarity(text1, text2):
    """Calculate sequence similarity between two texts"""
    # Normalize texts
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    
    # Calculate similarity
    matcher = SequenceMatcher(None, text1, text2)
    
    return matcher.ratio()

def extract_soccer_actions(text):
    """
    Extract soccer actions from text
    Format: "<player_number> <action> <result>"
    """
    # Normalize text
    text = normalize_text(text)
    
    # Split into actions
    actions = [a.strip() for a in text.split(".") if a.strip()]
    
    # Parse each action
    parsed_actions = []
    for action in actions:
        # Try to extract player number, action, and result
        match = re.match(r'(\d+)\s+(.+)', action)
        
        if match:
            player = match.group(1)
            action_text = match.group(2)
            
            parsed_actions.append({
                "player": player,
                "action": action_text
            })
        else:
            # If no match, keep the raw action
            parsed_actions.append({
                "raw": action
            })
    
    return parsed_actions

def compare_actions(predicted_actions, reference_actions, similarity_threshold=0.8):
    """Compare predicted and reference actions"""
    # Track matches
    matches = []
    unmatched_pred = []
    unmatched_ref = []
    
    # For each reference action, find the best match in predictions
    for ref_idx, ref_action in enumerate(reference_actions):
        best_match_idx = None
        best_score = 0
        
        ref_text = ref_action.get("raw", f"{ref_action.get('player', '')} {ref_action.get('action', '')}")
        
        for pred_idx, pred_action in enumerate(predicted_actions):
            pred_text = pred_action.get("raw", f"{pred_action.get('player', '')} {pred_action.get('action', '')}")
            
            score = sequence_similarity(pred_text, ref_text)
            
            if score > best_score:
                best_score = score
                best_match_idx = pred_idx
        
        # If a good match is found
        if best_score > similarity_threshold and best_match_idx is not None:
            matches.append({
                "ref_idx": ref_idx,
                "pred_idx": best_match_idx,
                "similarity": best_score,
                "ref_text": ref_text,
                "pred_text": predicted_actions[best_match_idx].get("raw", 
                                f"{predicted_actions[best_match_idx].get('player', '')} {predicted_actions[best_match_idx].get('action', '')}")
            })
            
            # Mark as matched
            predicted_actions[best_match_idx]["matched"] = True
        else:
            unmatched_ref.append(ref_idx)
    
    # Find unmatched predictions
    for pred_idx, pred_action in enumerate(predicted_actions):
        if not pred_action.get("matched", False):
            unmatched_pred.append(pred_idx)
    
    return matches, unmatched_pred, unmatched_ref

def convert_audio_format(input_path, output_path=None, target_format="wav", sample_rate=16000, channels=1):
    """Convert audio to a specific format and sample rate"""
    # If output path not specified, create one based on input
    if output_path is None:
        dirname = os.path.dirname(input_path)
        basename = os.path.basename(input_path)
        filename, _ = os.path.splitext(basename)
        output_path = os.path.join(dirname, f"{filename}.{target_format}")
    
    # Load audio
    audio = AudioSegment.from_file(input_path)
    
    # Convert to mono if needed
    if channels == 1 and audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Set sample rate
    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)
    
    # Export
    audio.export(output_path, format=target_format)
    
    return output_path

def extract_audio_features(audio_path, sample_rate=16000):
    """Extract audio features for analysis"""
    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Extract features
    features = {
        "duration": librosa.get_duration(y=y, sr=sr),
        "rms": np.mean(librosa.feature.rms(y=y)[0]),
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y=y)[0]),
        "mfcc": np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]),
        "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]),
        "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]),
    }
    
    return features

def visualize_audio(audio_path, output_path=None, sample_rate=16000):
    """Create visualizations of audio for analysis"""
    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    # If output path not specified, create one based on input
    if output_path is None:
        dirname = os.path.dirname(audio_path)
        basename = os.path.basename(audio_path)
        filename, _ = os.path.splitext(basename)
        output_path = os.path.join(dirname, f"{filename}_visualization.png")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    
    # Plot spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # Plot MFCC
    plt.subplot(3, 1, 3)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def split_audio_file(audio_path, output_dir, segment_length=5.0, overlap=0.0):
    """Split an audio file into segments"""
    # Create output directory
    create_directory(output_dir)
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Calculate segment parameters
    segment_samples = int(segment_length * sr)
    hop_samples = int((segment_length - overlap) * sr)
    
    # Split audio
    segments = []
    for i, start_sample in enumerate(range(0, len(y), hop_samples)):
        end_sample = min(start_sample + segment_samples, len(y))
        
        # Get segment
        segment = y[start_sample:end_sample]
        
        # If the segment is too short, pad it or skip it
        if len(segment) < segment_samples * 0.5:
            continue
        
        # Save segment
        basename = os.path.basename(audio_path)
        filename, ext = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{filename}_segment_{i:03d}.wav")
        sf.write(output_path, segment, sr)
        
        # Calculate segment time
        start_time = start_sample / sr
        end_time = end_sample / sr
        
        # Store segment info
        segments.append({
            "path": output_path,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
        })
    
    return segments

def augment_audio(audio_path, output_dir, num_augmentations=5, noise_level=0.005, pitch_shift_range=2, speed_range=0.2):
    """Create augmented versions of an audio file"""
    # Create output directory
    create_directory(output_dir)
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Get basename
    basename = os.path.basename(audio_path)
    filename, ext = os.path.splitext(basename)
    
    augmentations = []
    
    # Original audio
    original_path = os.path.join(output_dir, f"{filename}_original.wav")
    sf.write(original_path, y, sr)
    augmentations.append({
        "path": original_path,
        "type": "original"
    })
    
    # Add noise
    for i in range(num_augmentations):
        noise = np.random.normal(0, noise_level, len(y))
        noisy_audio = y + noise
        output_path = os.path.join(output_dir, f"{filename}_noise_{i:02d}.wav")
        sf.write(output_path, noisy_audio, sr)
        augmentations.append({
            "path": output_path,
            "type": "noise",
            "level": noise_level
        })
    
    # Pitch shift
    for i in range(num_augmentations):
        n_steps = np.random.uniform(-pitch_shift_range, pitch_shift_range)
        shifted_audio = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        output_path = os.path.join(output_dir, f"{filename}_pitch_{i:02d}.wav")
        sf.write(output_path, shifted_audio, sr)
        augmentations.append({
            "path": output_path,
            "type": "pitch_shift",
            "n_steps": n_steps
        })
    
    # Time stretch
    for i in range(num_augmentations):
        rate = np.random.uniform(1 - speed_range, 1 + speed_range)
        stretched_audio = librosa.effects.time_stretch(y, rate=rate)
        output_path = os.path.join(output_dir, f"{filename}_speed_{i:02d}.wav")
        sf.write(output_path, stretched_audio, sr)
        augmentations.append({
            "path": output_path,
            "type": "time_stretch",
            "rate": rate
        })
    
    return augmentations
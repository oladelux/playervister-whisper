import numpy as np
import librosa
import soundfile as sf
import random
import os
from scipy import signal

def add_noise(audio, noise_level=0.005):
    """Add random noise to the audio"""
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

def stretch_audio(audio, rate=1.0):
    """Time-stretch the audio without changing the pitch"""
    return librosa.effects.time_stretch(audio, rate=rate)

def shift_pitch(audio, sr, n_steps=0):
    """Shift the pitch of the audio"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def change_volume(audio, factor=1.0):
    """Change the volume of the audio"""
    return audio * factor

def apply_filter(audio, sr, filter_type='lowpass', cutoff_freq=1000):
    """Apply a filter to the audio"""
    nyq = 0.5 * sr
    normal_cutoff = cutoff_freq / nyq
    b, a = signal.butter(5, normal_cutoff, btype=filter_type, analog=False)
    return signal.filtfilt(b, a, audio)

def change_speed(audio, speed_factor=1.0):
    """Change the speed of the audio without time-stretching"""
    indices = np.round(np.arange(0, len(audio), speed_factor)).astype(int)
    indices = indices[indices < len(audio)]
    return audio[indices]

def apply_room_simulation(audio, sr, room_scale=0.5, reverb_time=0.1):
    """Apply simple room simulation (reverb)"""
    # Create a simple reverb impulse response
    reverb_length = int(sr * reverb_time)
    impulse_response = np.exp(-np.arange(reverb_length) / (sr * room_scale))
    impulse_response = impulse_response / np.sum(impulse_response)
    
    # Apply convolution for reverb effect
    return signal.convolve(audio, impulse_response, mode='full')[:len(audio)]

def augment_for_training(audio, sr, config):
    """Apply a random selection of augmentations for training"""
    # Make a copy to avoid modifying the original
    augmented = np.copy(audio)
    
    # Randomly apply augmentations
    # Each one has a probability of being applied
    
    # 1. Pitch shift (randomly)
    if random.random() < 0.5 and "pitch_shift_range" in config:
        shift_steps = np.random.uniform(-config["pitch_shift_range"], config["pitch_shift_range"])
        augmented = shift_pitch(augmented, sr, shift_steps)
    
    # 2. Time stretching (randomly)
    if random.random() < 0.5 and "speed_range" in config:
        stretch_factor = np.random.uniform(1 - config["speed_range"], 1 + config["speed_range"])
        augmented = stretch_audio(augmented, stretch_factor)
    
    # 3. Add background noise (randomly)
    if random.random() < 0.5 and "noise_level" in config:
        augmented = add_noise(augmented, config["noise_level"])
    
    # 4. Change volume (randomly)
    if random.random() < 0.3:
        volume_factor = np.random.uniform(0.8, 1.2)
        augmented = change_volume(augmented, volume_factor)
    
    # 5. Apply filter (randomly)
    if random.random() < 0.3:
        filter_type = random.choice(['lowpass', 'highpass'])
        cutoff = np.random.uniform(1000, 7000) if filter_type == 'lowpass' else np.random.uniform(100, 1000)
        augmented = apply_filter(augmented, sr, filter_type, cutoff)
    
    # 6. Apply reverb (randomly)
    if random.random() < 0.3:
        room_scale = np.random.uniform(0.1, 0.5)
        reverb_time = np.random.uniform(0.05, 0.2)
        augmented = apply_room_simulation(augmented, sr, room_scale, reverb_time)
    
    return augmented

def save_augmented(audio, sr, output_path):
    """Save augmented audio to file"""
    sf.write(output_path, audio, sr, 'PCM_16')
    return output_path

def batch_augment(audio_path, output_dir, config, num_augmentations=5):
    """Create multiple augmented versions of a single audio file"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base name for output files
    base_name = os.path.basename(audio_path)
    name, _ = os.path.splitext(base_name)
    
    # Load audio
    try:
        y, sr = librosa.load(audio_path, sr=config.get("sample_rate", 16000), mono=True)
    except Exception as e:
        print(f"Error loading audio for augmentation: {e}")
        return []
    
    augmented_paths = []
    
    # Create multiple augmented versions
    for i in range(num_augmentations):
        aug_audio = augment_for_training(y, sr, config)
        aug_path = os.path.join(output_dir, f"{name}_aug_{i}.wav")
        save_augmented(aug_audio, sr, aug_path)
        augmented_paths.append(aug_path)
    
    return augmented_paths 
import os
import json
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment
import random
from tqdm import tqdm
import argparse
import shutil

def create_directory(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def validate_and_normalize_audio(audio_path, output_path=None, sample_rate=16000):
    """
    Validate audio file and normalize it to 16kHz mono WAV format
    Returns the path to the normalized audio
    """
    try:
        # If output path not specified, use the original path
        if output_path is None:
            base_name = os.path.basename(audio_path)
            file_name, _ = os.path.splitext(base_name)
            output_path = os.path.join(os.path.dirname(audio_path), f"{file_name}_normalized.wav")

        print(f"  Processing audio: {audio_path}")
        print(f"  Output path: {output_path}")
        
        # For m4a files, try using pydub first
        _, ext = os.path.splitext(audio_path)
        if ext.lower() == '.m4a':
            try:
                print(f"  File is m4a, trying pydub conversion...")
                audio = AudioSegment.from_file(audio_path, format="m4a")
                audio = audio.set_channels(1).set_frame_rate(sample_rate)
                audio.export(output_path, format="wav")
                print(f"  ✓ Pydub conversion successful")
                return output_path
            except Exception as e:
                print(f"  ✗ Pydub conversion failed: {e}")
                print(f"  Falling back to librosa...")

        # Load audio file with librosa
        print(f"  Loading with librosa...")
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        
        # Normalize audio
        print(f"  Normalizing audio...")
        y = librosa.util.normalize(y)
        
        # Save as WAV
        print(f"  Saving as WAV...")
        sf.write(output_path, y, sample_rate, 'PCM_16')
        
        print(f"  ✓ Processing complete")
        return output_path
    except Exception as e:
        print(f"  ✗ Error processing {audio_path}: {e}")
        return None

def create_dataset_json(audio_files, transcript_files, output_json_path, train_split=0.8, val_split=0.1):
    """
    Create JSON files for training, validation, and testing
    Each JSON file contains a list of {"audio_filepath": "...", "text": "..."}
    """
    data = []
    
    print(f"Creating dataset with {len(audio_files)} audio files and {len(transcript_files)} transcript files")
    for audio_path, transcript_path in tqdm(zip(audio_files, transcript_files), total=len(audio_files)):
        # Read transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read().strip()
        
        # Debug info
        print(f"Audio: {audio_path}")
        print(f"Transcript: {transcript_path}")
        print(f"Text: {transcript}")
        
        # Create data entry
        data.append({
            "audio_filepath": audio_path,
            "text": transcript
        })
    
    # Shuffle data
    random.shuffle(data)
    
    # Always ensure at least one sample in each split if possible
    if len(data) < 3:
        if len(data) == 1:
            train_data = data
            val_data = []
            test_data = []
        elif len(data) == 2:
            train_data = data[:1]
            val_data = data[1:]
            test_data = []
    else:
        # Regular split for larger datasets
        train_end = max(1, int(len(data) * train_split))
        val_end = train_end + max(1, int(len(data) * val_split))
        val_end = min(val_end, len(data) - 1)  # Ensure we have at least one test sample
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
    
    # Create directories
    output_dir = os.path.dirname(output_json_path)
    create_directory(output_dir)
    
    # Save JSON files
    base_name = os.path.basename(output_json_path)
    file_name, ext = os.path.splitext(base_name)
    
    train_json_path = os.path.join(output_dir, f"{file_name}_train{ext}")
    val_json_path = os.path.join(output_dir, f"{file_name}_val{ext}")
    test_json_path = os.path.join(output_dir, f"{file_name}_test{ext}")
    
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)
    
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    
    print(f"Created {len(train_data)} training samples, {len(val_data)} validation samples, {len(test_data)} test samples")
    
    # Create a metadata file
    metadata = {
        "dataset_size": len(data),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "train_path": train_json_path,
        "val_path": val_json_path,
        "test_path": test_json_path
    }
    
    metadata_path = os.path.join(output_dir, f"{file_name}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    return metadata

def prepare_soccer_data(audio_dir, transcript_dir, output_dir, normalize_audio=True):
    """
    Prepare soccer data for fine-tuning
    1. Match audio files with transcript files
    2. Normalize audio files (optional)
    3. Create JSON files for training, validation, and testing
    """
    # Create output directories
    processed_audio_dir = os.path.join(output_dir, "processed_audio")
    create_directory(processed_audio_dir)
    
    # Get all audio files
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files:")
    for audio_file in audio_files:
        print(f"  - {audio_file}")
    
    # Get all transcript files
    transcript_files = []
    for root, _, files in os.walk(transcript_dir):
        for file in files:
            if file.endswith(('.txt', '.json')):
                transcript_files.append(os.path.join(root, file))
    
    print(f"Found {len(transcript_files)} transcript files:")
    for transcript_file in transcript_files:
        print(f"  - {transcript_file}")
    
    # Match audio files with transcript files
    # Assuming the filenames match except for the extension
    matched_pairs = []
    normalized_audio_files = []
    matched_transcript_files = []
    
    print("\nMatching files...")
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        audio_basename = os.path.basename(audio_path)
        audio_name, _ = os.path.splitext(audio_basename)
        print(f"\nAudio file: {audio_basename}, Base name: {audio_name}")
        
        # Find matching transcript file
        transcript_path = None
        for transcript_file in transcript_files:
            transcript_basename = os.path.basename(transcript_file)
            transcript_name, _ = os.path.splitext(transcript_basename)
            print(f"  Checking against transcript: {transcript_basename}, Base name: {transcript_name}")
            
            if audio_name == transcript_name:
                transcript_path = transcript_file
                print(f"  ✓ MATCH FOUND!")
                break
        
        if transcript_path is not None:
            # Normalize audio if requested
            if normalize_audio:
                output_audio_path = os.path.join(processed_audio_dir, f"{audio_name}.wav")
                normalized_path = validate_and_normalize_audio(audio_path, output_audio_path)
                
                if normalized_path is not None:
                    normalized_audio_files.append(normalized_path)
                    matched_transcript_files.append(transcript_path)
                    print(f"  Added to dataset")
                else:
                    print(f"  ✗ Audio processing failed, skipping")
            else:
                normalized_audio_files.append(audio_path)
                matched_transcript_files.append(transcript_path)
                print(f"  Added to dataset (no normalization)")
        else:
            print(f"  ✗ No matching transcript found")
    
    print(f"Found {len(normalized_audio_files)} matched and processed audio-transcript pairs")
    
    # Create dataset JSON
    output_json_path = os.path.join(output_dir, "soccer_dataset.json")
    metadata = create_dataset_json(normalized_audio_files, matched_transcript_files, output_json_path)
    
    return metadata

def create_huggingface_dataset(json_file, output_dir):
    """Convert JSON dataset to Hugging Face format"""
    print(f"Creating Hugging Face dataset from {json_file}")
    
    # Check if file exists
    if not os.path.exists(json_file):
        print(f"Error: JSON file not found: {json_file}")
        return None
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Debug JSON content
    print(f"JSON contains {len(data)} entries")
    
    # Create CSV file
    df = pd.DataFrame(data)
    df = df.rename(columns={"audio_filepath": "audio", "text": "sentence"})
    
    # Create output directory
    create_directory(output_dir)
    
    # Save CSV file
    output_csv = os.path.join(output_dir, "dataset.csv")
    df.to_csv(output_csv, index=False)
    
    print(f"Saved CSV file with {len(df)} rows to {output_csv}")
    
    return output_csv

def main():
    parser = argparse.ArgumentParser(description="Prepare soccer data for fine-tuning")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--transcript_dir", required=True, help="Directory containing transcript files")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed data")
    parser.add_argument("--no_normalize", action="store_true", help="Skip audio normalization")
    
    args = parser.parse_args()
    
    print("\n=== Starting Data Preparation ===")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Transcript directory: {args.transcript_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Normalize audio: {not args.no_normalize}")
    
    # Prepare data
    metadata = prepare_soccer_data(
        args.audio_dir, 
        args.transcript_dir, 
        args.output_dir, 
        normalize_audio=not args.no_normalize
    )
    
    if metadata["train_size"] == 0:
        print("\n⚠️ WARNING: No training samples found!")
        print("This will prevent training from working properly.")
        print("Please check that your audio and transcript files have matching names.")
        return
    
    # Create Hugging Face datasets
    train_csv = create_huggingface_dataset(
        metadata["train_path"], 
        os.path.join(args.output_dir, "hf_train")
    )
    
    val_csv = create_huggingface_dataset(
        metadata["val_path"], 
        os.path.join(args.output_dir, "hf_val")
    )
    
    test_csv = create_huggingface_dataset(
        metadata["test_path"], 
        os.path.join(args.output_dir, "hf_test")
    )
    
    print(f"Created Hugging Face datasets: {train_csv}, {val_csv}, {test_csv}")
    print("\n=== Data Preparation Complete ===")

if __name__ == "__main__":
    main()

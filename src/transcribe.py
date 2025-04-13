import argparse
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import soundfile as sf
from pydub import AudioSegment
import os
import json

def process_audio(audio_path, output_path=None, sample_rate=16000):
    """Process audio file to the format expected by Whisper"""
    print(f"Processing audio: {audio_path}")
    
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
            print(f"Converting m4a to wav...")
            audio = AudioSegment.from_file(audio_path, format="m4a")
            audio = audio.set_channels(1).set_frame_rate(sample_rate)
            audio.export(output_path, format="wav")
            print(f"Saved processed audio to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error processing m4a: {e}")
            print("Falling back to librosa...")
    
    # For other formats or as fallback, use librosa
    try:
        print(f"Loading audio with librosa...")
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        print(f"Saving as wav...")
        sf.write(output_path, y, sample_rate, 'PCM_16')
        print(f"Saved processed audio to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def transcribe_audio(model_path, audio_path, device="auto"):
    """Transcribe audio using the specified model"""
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load model and processor
    print(f"Loading model from {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    
    # Process audio if needed
    _, ext = os.path.splitext(audio_path)
    if ext.lower() in ['.m4a', '.mp3', '.ogg']:
        processed_path = process_audio(audio_path)
        if processed_path:
            audio_path = processed_path
    
    # Load audio
    print(f"Loading audio file: {audio_path}")
    audio_input, sample_rate = librosa.load(audio_path, sr=16000)
    
    # Process audio for model input
    print("Processing audio for model input")
    features = processor.feature_extractor(
        audio_input, 
        sampling_rate=sample_rate, 
        return_tensors="pt"
    )
    
    # Explicitly create attention mask
    input_features = features.input_features.to(device)
    attention_mask = torch.ones(input_features.shape[:2], device=device)
    
    # Generate transcription with proper handling of attention mask
    print("Generating transcription")
    with torch.no_grad():
        generated_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            language="en",
            task="transcribe",
            return_dict_in_generate=False,
        )
    
    # Decode the prediction
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Transcription: {transcription}")
    
    return transcription

def batch_process(model_path, audio_dir, output_file, device="auto"):
    """Process all audio files in a directory"""
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load model and processor
    print(f"Loading model from {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    
    # Find all audio files
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each file
    results = []
    for audio_path in audio_files:
        try:
            # Process audio if needed
            _, ext = os.path.splitext(audio_path)
            if ext.lower() in ['.m4a', '.mp3', '.ogg']:
                processed_path = process_audio(audio_path)
                if processed_path:
                    audio_path_to_use = processed_path
                else:
                    continue
            else:
                audio_path_to_use = audio_path
            
            # Load audio
            audio_input, sample_rate = librosa.load(audio_path_to_use, sr=16000)
            
            # Process audio for model input with attention mask
            features = processor.feature_extractor(
                audio_input, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            )
            
            input_features = features.input_features.to(device)
            attention_mask = torch.ones(input_features.shape[:2], device=device)
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    language="en",
                    task="transcribe",
                    return_dict_in_generate=False,
                )
            
            # Decode the prediction
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Store result
            result = {
                "file": audio_path,
                "transcription": transcription,
                "status": "success"
            }
            
            # Print result
            print(f"\nFile: {os.path.basename(audio_path)}")
            print(f"Transcription: {transcription}")
            
            results.append(result)
            
        except Exception as e:
            # Handle errors
            result = {
                "file": audio_path,
                "error": str(e),
                "status": "error"
            }
            
            # Print error
            print(f"\nError processing {os.path.basename(audio_path)}: {str(e)}")
            
            results.append(result)
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with fine-tuned Whisper model")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--mode", choices=["file", "batch"], default="file", help="Transcribe single file or batch process a directory")
    parser.add_argument("--audio_path", required=True, help="Path to audio file or directory")
    parser.add_argument("--output_file", default="transcriptions.json", help="Path to save batch results (only used in batch mode)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use for inference")
    
    args = parser.parse_args()
    
    if args.mode == "file":
        # Transcribe single file
        transcription = transcribe_audio(args.model_path, args.audio_path, args.device)
        print("\n===== Transcription =====")
        print(transcription)
        print("========================\n")
    else:
        # Batch process directory
        batch_process(args.model_path, args.audio_path, args.output_file, args.device)

if __name__ == "__main__":
    main()
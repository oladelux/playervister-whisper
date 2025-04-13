import argparse
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import soundfile as sf
from pydub import AudioSegment
import os

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
    input_features = processor.feature_extractor(
        audio_input, 
        sampling_rate=sample_rate, 
        return_tensors="pt"
    ).input_features.to(device)
    
    # Clear any forced decoder IDs from generation config
    # This is the key fix for the compatibility issue
    model.generation_config.forced_decoder_ids = None
    
    # Generate transcription
    print("Generating transcription")
    predicted_ids = model.generate(input_features)
    
    # Decode the prediction
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Whisper model")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--audio_path", required=True, help="Path to audio file")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Run transcription
    transcription = transcribe_audio(args.model_path, args.audio_path, args.device)
    
    # Print result
    print("\n===== Transcription =====")
    print(transcription)
    print("========================\n")

if __name__ == "__main__":
    main()
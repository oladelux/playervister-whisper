import argparse
import torch
import os
import json
import logging
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import soundfile as sf
from pydub import AudioSegment
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_audio(audio_path, output_path=None, sample_rate=16000):
    """Process audio file to the format expected by Whisper"""
    logger.info(f"Processing audio: {audio_path}")
    
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
            logger.info(f"Converting m4a to wav...")
            audio = AudioSegment.from_file(audio_path, format="m4a")
            audio = audio.set_channels(1).set_frame_rate(sample_rate)
            audio.export(output_path, format="wav")
            logger.info(f"Saved processed audio to {output_path}")
            return output_path
        except Exception as e:
            logger.warning(f"Error processing m4a: {e}")
            logger.info("Falling back to librosa...")
    
    # For other formats or as fallback, use librosa
    try:
        logger.info(f"Loading audio with librosa...")
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        
        # Normalize audio if it's too quiet
        if np.abs(y).max() < 0.1:
            logger.info("Audio volume is low, normalizing...")
            y = y / (np.abs(y).max() + 1e-10) * 0.95
        
        logger.info(f"Saving as wav...")
        sf.write(output_path, y, sample_rate, 'PCM_16')
        logger.info(f"Saved processed audio to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return None

def load_model(model_path, device="auto"):
    """Load the Whisper model and processor"""
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load model and processor
    logger.info(f"Loading model from {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    
    return model, processor, device

def transcribe_audio(model, processor, audio_path, device="cuda", generation_config=None):
    """Transcribe audio using the specified model"""
    # Process audio if needed
    _, ext = os.path.splitext(audio_path)
    if ext.lower() in ['.m4a', '.mp3', '.ogg']:
        processed_path = process_audio(audio_path)
        if processed_path:
            audio_path = processed_path
    
    # Load audio
    logger.info(f"Loading audio file: {audio_path}")
    audio_input, sample_rate = librosa.load(audio_path, sr=16000)
    
    # Process audio for model input
    logger.info("Processing audio for model input")
    input_features = processor.feature_extractor(
        audio_input, 
        sampling_rate=sample_rate, 
        return_tensors="pt"
    ).input_features.to(device)
    
    # Set up generation config
    if generation_config is None:
        generation_config = {
            "max_length": 225,
            "num_beams": 5,
            "temperature": 0.0,
            "language": "en",
            "task": "transcribe"
        }
    
    # Clear any forced decoder IDs from generation config and set them properly
    model.generation_config.forced_decoder_ids = None
    
    # Set proper decoder IDs based on specified language and task
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=generation_config.get("language", "en"),
        task=generation_config.get("task", "transcribe")
    )
    
    # Generate transcription
    logger.info("Generating transcription")
    start_time = time.time()
    
    predicted_ids = model.generate(
        input_features, 
        max_length=generation_config.get("max_length", 225),
        num_beams=generation_config.get("num_beams", 5),
        temperature=generation_config.get("temperature", 0.0),
        forced_decoder_ids=forced_decoder_ids
    )
    
    end_time = time.time()
    
    # Decode the prediction
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")
    
    return transcription

def batch_transcribe(model_path, audio_dir, output_file, device="auto", generation_config=None):
    """Transcribe all audio files in a directory"""
    # Load model
    model, processor, device = load_model(model_path, device)
    
    # Find all audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg'))]
    
    results = []
    
    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)
        logger.info(f"Transcribing {audio_file}")
        
        try:
            transcription = transcribe_audio(model, processor, audio_path, device, generation_config)
            results.append({
                "file": audio_file,
                "transcription": transcription
            })
            logger.info(f"Transcription: {transcription}")
        except Exception as e:
            logger.error(f"Error transcribing {audio_file}: {e}")
            results.append({
                "file": audio_file,
                "transcription": "ERROR",
                "error": str(e)
            })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Transcriptions saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Whisper model")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--audio_path", help="Path to audio file or directory of files")
    parser.add_argument("--output_file", help="Path to output file for batch transcription")
    parser.add_argument("--batch_mode", action="store_true", help="Transcribe all audio files in directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use for inference")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    
    args = parser.parse_args()
    
    # Set up generation config
    generation_config = {
        "num_beams": args.num_beams,
        "temperature": args.temperature,
        "language": "en",
        "task": "transcribe"
    }
    
    # Run transcription
    if args.batch_mode:
        if not args.audio_path or not os.path.isdir(args.audio_path):
            logger.error("In batch mode, audio_path must be a directory")
            return
        
        if not args.output_file:
            args.output_file = os.path.join(args.audio_path, "transcriptions.json")
        
        batch_transcribe(args.model_path, args.audio_path, args.output_file, args.device, generation_config)
    else:
        if not args.audio_path:
            logger.error("audio_path is required in single file mode")
            return
        
        # Load model
        model, processor, device = load_model(args.model_path, args.device)
        
        # Transcribe
        transcription = transcribe_audio(model, processor, args.audio_path, device, generation_config)
        
        # Print result
        print("\n===== Transcription =====")
        print(transcription)
        print("========================\n")

if __name__ == "__main__":
    main()
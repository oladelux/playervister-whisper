# Football Command Speech Recognition

A specialized speech recognition system for football/soccer commands, built using OpenAI's Whisper model. This system is fine-tuned to achieve high accuracy (close to 100%) for transcribing specific football action commands.

## Overview

This project fine-tunes the Whisper model specifically for football action commands like:

- "7 pass successful"
- "10 shot on target"
- "5 clearance successful"
- "9 penalty scored"
- etc.

The model is optimized to recognize player numbers and action types with extremely high accuracy, making it suitable for automated football analysis systems.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/football-whisper.git
cd football-whisper
```

2. Create and activate a virtual environment:

```bash
python -m venv soccer_whisper_env
source soccer_whisper_env/bin/activate  # On Windows: soccer_whisper_env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── configs/                # Configuration files
│   └── training_config.json # Training configuration
├── data/                   # Data directory
│   ├── audio/              # Football command audio files
│   └── transcripts/        # Corresponding transcript files
├── models/                 # Saved models directory
├── src/                    # Source code
│   ├── augment.py          # Audio augmentation utilities
│   ├── evaluate.py         # Model evaluation script
│   ├── inference.py        # Inference script
│   ├── prepare_data.py     # Data preparation script
│   └── train.py            # Model training script
├── simple_inference.py     # Simplified inference script
├── simple_train.py         # Simplified training script
└── requirements.txt        # Python dependencies
```

## Usage

### Data Preparation

1. Place your audio files in `data/audio/` and corresponding transcripts in `data/transcripts/`.
2. Run the data preparation script:

```bash
python src/prepare_data.py --audio_dir data/audio --transcript_dir data/transcripts --output_dir data/processed
```

This will:

- Process audio files to the format expected by Whisper
- Create augmented versions for better training
- Split data into training, validation, and test sets

### Training

Train the model using:

```bash
python src/train.py --config configs/training_config.json --train_csv data/processed/train.csv --val_csv data/processed/validation.csv --output_dir models/football-whisper
```

Configuration options can be adjusted in `configs/training_config.json`. The default configuration uses:

- `whisper-large-v2` model for maximum accuracy
- 30 training epochs
- Frozen encoder for faster training
- Audio augmentation for better robustness

### Inference

For transcribing single audio files:

```bash
python src/inference.py --model_path models/football-whisper/final_model --audio_path path/to/audio.m4a
```

For batch transcription:

```bash
python src/inference.py --model_path models/football-whisper/final_model --audio_path data/test_audio --batch_mode --output_file transcriptions.json
```

### Evaluation

Evaluate model performance:

```bash
python src/evaluate.py --model_path models/football-whisper/final_model --test_csv data/processed/test.csv --output_dir evaluation_results
```

This will:

- Calculate WER, CER, and other accuracy metrics
- Analyze errors by command type
- Generate visualizations and reports
- Identify the most challenging examples

## Model Selection

This project uses OpenAI's Whisper model, with the following options available:

| Model Size | Parameters | Required VRAM | Accuracy  | Training Speed |
| ---------- | ---------- | ------------- | --------- | -------------- |
| tiny       | 39M        | ~2GB          | Low       | Very Fast      |
| base       | 74M        | ~3GB          | Medium    | Fast           |
| small      | 244M       | ~4GB          | Good      | Medium         |
| medium     | 769M       | ~8GB          | Very Good | Slow           |
| large-v2   | 1550M      | ~14GB         | Excellent | Very Slow      |

The default configuration uses `whisper-large-v2` for maximum accuracy, but you can change it in the config file if you have limited GPU resources.

## Tuning for Accuracy

To achieve near 100% accuracy:

1. Use the largest model your hardware can support
2. Increase data augmentation to add more training examples
3. Train for more epochs (30-50)
4. Use a lower learning rate (5e-6 or lower)
5. Focus on the specific vocabulary in your domain
6. Analyze errors and add more examples of problematic command types

## License

[MIT License](LICENSE)

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

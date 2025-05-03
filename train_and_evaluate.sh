#!/bin/bash

# Football Whisper Training and Evaluation Pipeline
# This script runs the entire pipeline from data preparation to evaluation

# Set variables
CONFIG_FILE="configs/training_config.json"
AUDIO_DIR="data/audio"
TRANSCRIPT_DIR="data/transcripts"
PROCESSED_DIR="data/processed"
MODEL_DIR="models/football-whisper"
EVAL_DIR="evaluation_results"

# Set environment variables
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=true

# Create directories
mkdir -p $PROCESSED_DIR
mkdir -p $MODEL_DIR
mkdir -p $EVAL_DIR

echo "========================================"
echo "🎤 FOOTBALL WHISPER TRAINING PIPELINE 🎤"
echo "========================================"

# Check if GPU is available
if python -c "import torch; print(torch.cuda.is_available());" | grep -q "True"; then
  echo "✅ GPU is available for training"
else
  echo "⚠️  WARNING: GPU is not available. Training will be slow!"
  read -p "Do you want to continue with CPU training? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting..."
    exit 1
  fi
fi

# Step 1: Data preparation
echo ""
echo "📋 Step 1: Preparing data..."
python src/prepare_data.py --audio_dir $AUDIO_DIR --transcript_dir $TRANSCRIPT_DIR --output_dir $PROCESSED_DIR --config_path $CONFIG_FILE
if [ $? -ne 0 ]; then
  echo "❌ Data preparation failed. Exiting."
  exit 1
fi

# Check if data files exist
if [ ! -f "$PROCESSED_DIR/train.csv" ]; then
  echo "❌ Training data not found. Exiting."
  exit 1
fi

# Step 2: Training the model
echo ""
echo "🚀 Step 2: Training model..."
python src/train.py --config $CONFIG_FILE --train_csv $PROCESSED_DIR/train.csv --val_csv $PROCESSED_DIR/validation.csv --output_dir $MODEL_DIR
if [ $? -ne 0 ]; then
  echo "❌ Training failed. Exiting."
  exit 1
fi

# Check if model was generated
if [ ! -d "$MODEL_DIR/final_model" ]; then
  echo "❌ Model was not saved properly. Exiting."
  exit 1
fi

# Step 3: Evaluating model
echo ""
echo "📊 Step 3: Evaluating model..."
python src/evaluate.py --model_path $MODEL_DIR/final_model --test_csv $PROCESSED_DIR/test.csv --output_dir $EVAL_DIR
if [ $? -ne 0 ]; then
  echo "❌ Evaluation failed."
  exit 1
fi

# Step 4: Test with a sample file
echo ""
echo "🔈 Step 4: Testing with a sample file..."
# Get the first audio file from the test set
SAMPLE_FILE=$(head -n 2 $PROCESSED_DIR/test.csv | tail -n 1 | cut -d',' -f1)
if [ -f "$SAMPLE_FILE" ]; then
  python src/inference.py --model_path $MODEL_DIR/final_model --audio_path "$SAMPLE_FILE"
else
  echo "⚠️  No sample file available for testing"
fi

echo ""
echo "✅ Training and evaluation completed!"
echo "   - Model saved to: $MODEL_DIR/final_model"
echo "   - Evaluation results saved to: $EVAL_DIR"
echo ""
echo "To transcribe new audio files, run:"
echo "python src/inference.py --model_path $MODEL_DIR/final_model --audio_path path/to/audio.m4a"
echo "" 
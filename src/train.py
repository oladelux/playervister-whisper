import os
import json
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, Audio, Dataset
import pandas as pd
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    TrainerCallback
)
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
import wandb

# Ignore specific warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*The current process just got forked.*")

class LoggingCallback(TrainerCallback):
    """Custom callback for logging metrics during training"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            step = state.global_step
            for k, v in logs.items():
                if k != "epoch":
                    try:
                        wandb.log({k: v}, step=step)
                    except Exception as e:
                        print(f"Error logging to wandb: {e}")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text models"""
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Pad inputs
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore when computing loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Add labels to batch
        batch["labels"] = labels
        
        return batch

def compute_metrics(pred):
    """Compute WER and CER metrics"""
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    # Compute CER
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    # Print some examples for manual inspection
    if len(pred_str) > 0:
        print("\n===== Example Predictions =====")
        for i in range(min(3, len(pred_str))):
            print(f"Reference: {label_str[i]}")
            print(f"Prediction: {pred_str[i]}")
            print("-------------------")
    
    return {"wer": wer, "cer": cer}

def load_from_csv(csv_path):
    """Load dataset from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_columns = ["audio_path", "text"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")
        
        # Convert to Dataset
        dataset = Dataset.from_pandas(df)
        
        # Add audio column
        dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))
        
        return dataset
    except Exception as e:
        print(f"Error loading dataset from {csv_path}: {e}")
        return None

def prepare_dataset(dataset, processor):
    """Prepare dataset for training"""
    print(f"Processing dataset with {len(dataset)} examples")
    
    # Define processing function
    def prepare_example(example):
        # Process audio
        audio = example["audio_path"]
        
        # Check if audio is a dict (from Audio column)
        if isinstance(audio, dict):
            example["input_features"] = processor.feature_extractor(
                audio["array"], 
                sampling_rate=audio["sampling_rate"]
            ).input_features[0]
        else:
            # Assume it's a path
            try:
                # Load audio file
                import librosa
                audio_array, sr = librosa.load(audio, sr=16000, mono=True)
                example["input_features"] = processor.feature_extractor(
                    audio_array, 
                    sampling_rate=sr
                ).input_features[0]
            except Exception as e:
                print(f"Error loading audio file {audio}: {e}")
                # Create empty features as fallback
                example["input_features"] = torch.zeros(1, 80, 3000)
        
        # Process text
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        
        return example
    
    # Process dataset
    processed_dataset = dataset.map(
        prepare_example,
        remove_columns=dataset.column_names,
    )
    
    return processed_dataset

def train(config, train_csv, val_csv=None, output_dir=None, resume_from=None):
    """Train model with the given configuration"""
    # Set up device
    device = "cuda" if torch.cuda.is_available() and not config.get("no_cuda", False) else "cpu"
    print(f"Using device: {device}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join("models", f"whisper_{config['model_name'].split('/')[-1]}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize wandb if enabled
    if config.get("use_wandb", False):
        try:
            wandb.init(
                project=config.get("wandb_project", "football-whisper"),
                name=config.get("wandb_run_name", f"run-{config['model_name'].split('/')[-1]}"),
                config=config
            )
        except Exception as e:
            print(f"Error initializing wandb: {e}")
            print("Continuing without wandb logging")
    
    # Load model and processor
    print(f"Loading model: {config['model_name']}")
    global processor
    processor = WhisperProcessor.from_pretrained(config['model_name'])
    
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        model = WhisperForConditionalGeneration.from_pretrained(resume_from)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(config['model_name'])
    
    # Freeze encoder if specified
    if config.get("freeze_encoder", False):
        print("Freezing encoder parameters")
        for param in model.get_encoder().parameters():
            param.requires_grad = False
    
    # Enable gradient checkpointing if specified
    if config.get("gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Set language and task for generation
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config.get("language", "en"), 
        task=config.get("task", "transcribe")
    )
    
    # Load and prepare datasets
    print(f"Loading train dataset from {train_csv}")
    train_dataset = load_from_csv(train_csv)
    if train_dataset is None:
        print("Failed to load training dataset")
        return
    
    val_dataset = None
    if val_csv and os.path.exists(val_csv):
        print(f"Loading validation dataset from {val_csv}")
        val_dataset = load_from_csv(val_csv)
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_dataset, processor)
    if val_dataset:
        val_dataset = prepare_dataset(val_dataset, processor)
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("batch_size", 8),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        learning_rate=config.get("learning_rate", 1e-5),
        warmup_steps=config.get("warmup_steps", 500),
        max_steps=config.get("max_steps", -1),
        num_train_epochs=config.get("num_epochs", 10),
        fp16=torch.cuda.is_available() and config.get("fp16", True),
        logging_strategy="steps",
        logging_steps=config.get("logging_steps", 50),
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=config.get("eval_steps", 500) if val_dataset else None,
        save_strategy="steps",
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 3),
        load_best_model_at_end=val_dataset is not None and config.get("load_best_model_at_end", True),
        metric_for_best_model="wer" if val_dataset else None,
        greater_is_better=False,
        push_to_hub=config.get("push_to_hub", False),
        hub_model_id=config.get("hub_model_id", None),
        hub_token=config.get("hub_token", None),
        report_to="none",  # We'll handle wandb ourselves
        remove_unused_columns=True,
        label_smoothing_factor=config.get("label_smoothing", 0.1),
        group_by_length=config.get("group_by_length", True),
        weight_decay=config.get("weight_decay", 0.01),
        dataloader_num_workers=config.get("num_workers", 4),
    )
    
    # Create trainer
    callbacks = []
    if config.get("use_wandb", False):
        callbacks.append(LoggingCallback())
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics if val_dataset else None,
        callbacks=callbacks
    )
    
    print(f"Starting training for {config.get('num_epochs', 10)} epochs")
    
    # Start training
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate on validation set
    if val_dataset:
        print("Evaluating model on validation set")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Close wandb
    if config.get("use_wandb", False):
        wandb.finish()
    
    print("Training complete! Model saved to:", final_model_path)
    
    return final_model_path

def main():
    parser = argparse.ArgumentParser(description="Train Whisper model on football commands")
    parser.add_argument("--config", default="configs/training_config.json", help="Path to training config file")
    parser.add_argument("--train_csv", help="Path to training CSV file")
    parser.add_argument("--val_csv", help="Path to validation CSV file")
    parser.add_argument("--output_dir", help="Directory to save model")
    parser.add_argument("--resume_from", help="Path to checkpoint to resume from")
    parser.add_argument("--model_name", help="Whisper model to fine-tune")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        print("Using default config")
        config = {
            "model_name": "openai/whisper-small",
            "num_epochs": 10,
            "batch_size": 8,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "freeze_encoder": True,
            "gradient_checkpointing": True,
            "fp16": True
        }
    
    # Override config with command line arguments
    if args.model_name:
        config["model_name"] = args.model_name
    if args.no_cuda:
        config["no_cuda"] = True
    
    # Run training
    final_model_path = train(
        config, 
        train_csv=args.train_csv, 
        val_csv=args.val_csv, 
        output_dir=args.output_dir,
        resume_from=args.resume_from
    )
    
    return final_model_path

if __name__ == "__main__":
    main()
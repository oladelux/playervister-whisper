import os
import json
import torch
import argparse
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

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
    
    return {"wer": wer, "cer": cer}

def prepare_dataset(dataset, processor):
    """Prepare dataset for training"""
    # Check if audio column is a path or audio array
    if isinstance(dataset[0]["audio"], str):
        # Convert to audio array
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Define data preprocessing function
    def prepare_example(example):
        # Process audio
        audio = example["audio"]
        
        # Check if audio is a dict (from Audio column)
        if isinstance(audio, dict):
            example["input_features"] = processor.feature_extractor(
                audio["array"], 
                sampling_rate=audio["sampling_rate"]
            ).input_features[0]
        else:
            # Assume it's already a numpy array
            example["input_features"] = processor.feature_extractor(
                audio, 
                sampling_rate=16000
            ).input_features[0]
        
        # Process text
        example["labels"] = processor.tokenizer(example["sentence"]).input_ids
        
        return example
    
    # Process dataset
    processed_dataset = dataset.map(
        prepare_example,
        remove_columns=dataset.column_names,
        num_proc=2
    )
    
    return processed_dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on soccer data")
    parser.add_argument("--train_csv", required=True, help="Path to training CSV file")
    parser.add_argument("--val_csv", required=True, help="Path to validation CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save model")
    parser.add_argument("--model_name", default="openai/whisper-small", help="Whisper model to fine-tune")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    
    args = parser.parse_args()
    
    # Load datasets
    print(f"Loading training dataset from {args.train_csv}")
    train_dataset = load_dataset("csv", data_files=args.train_csv)["train"]
    print(f"Loaded {len(train_dataset)} training samples")
    
    print(f"Loading validation dataset from {args.val_csv}")
    val_dataset = load_dataset("csv", data_files=args.val_csv)["train"]
    print(f"Loaded {len(val_dataset)} validation samples")
    
    # Load model and processor
    print(f"Loading model: {args.model_name}")
    global processor
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Clear the forced decoder IDs to avoid issues
    model.config.forced_decoder_ids = None
    if hasattr(model, 'generation_config'):
        model.generation_config.forced_decoder_ids = None
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Prepare datasets
    print("Preparing training dataset")
    train_dataset = prepare_dataset(train_dataset, processor)
    print("Preparing validation dataset")
    val_dataset = prepare_dataset(val_dataset, processor)
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Create a minimal set of training arguments without problematic parameters
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=float(args.num_epochs),
        save_strategy="epoch",
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics if len(val_dataset) > 0 else None,
        tokenizer=processor,
    )
    
    # Start training
    print(f"Starting training for {args.num_epochs} epochs")
    trainer.train()
    
    # Save best model
    print("Saving final model")
    trainer.save_model(os.path.join(args.output_dir, "best_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "best_model"))
    
    # Evaluate on validation set if available
    if len(val_dataset) > 0:
        eval_results = trainer.evaluate()
        print(f"Final validation results: {eval_results}")
        
        # Save evaluation results
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f)
    
    print(f"Training complete! Model saved to {os.path.join(args.output_dir, 'best_model')}")

if __name__ == "__main__":
    main()
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
    # Cast the audio column to Audio type
    if isinstance(dataset[0]["audio"], str):
        print(f"Converting audio column from path to Audio type")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Define processing function
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
    print(f"Processing dataset with {len(dataset)} examples")
    processed_dataset = dataset.map(
        prepare_example,
        remove_columns=dataset.column_names,
    )
    
    return processed_dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on soccer data")
    parser.add_argument("--train_csv", required=True, help="Path to training CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save model")
    parser.add_argument("--model_name", default="openai/whisper-tiny", help="Whisper model to fine-tune")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # Setup device
    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.train_csv}")
    train_dataset = load_dataset("csv", data_files=args.train_csv)["train"]
    print(f"Loaded {len(train_dataset)} training examples")
    
    # Load model and processor
    print(f"Loading model: {args.model_name}")
    global processor
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Set language and task for generation
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    
    # Prepare dataset
    train_dataset = prepare_dataset(train_dataset, processor)
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Define training arguments
    # For a small dataset, we'll increase the number of epochs and disable evaluation
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,  # Small batch size for small dataset
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=0,
        max_steps=-1,  # Use num_epochs instead
        num_train_epochs=args.num_epochs,
        fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA is available
        logging_strategy="steps",
        logging_steps=1,  # Log after each step (batch)
        save_strategy="epoch",
        save_total_limit=3,  # Keep only the last 3 checkpoints
        load_best_model_at_end=False,  # No evaluation, so can't load best model
        report_to="none",  # Disable reporting to avoid wandb etc.
        remove_unused_columns=True,
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )
    
    print(f"Starting training for {args.num_epochs} epochs")
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    print("Training complete! Model saved to:", os.path.join(args.output_dir, "final_model"))
    
    return os.path.join(args.output_dir, "final_model")

if __name__ == "__main__":
    main()
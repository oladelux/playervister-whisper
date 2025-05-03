import argparse
import torch
import os
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def load_test_data(test_csv):
    """Load test dataset from CSV file"""
    logger.info(f"Loading test data from {test_csv}")
    df = pd.read_csv(test_csv)
    
    # Check required columns
    required_columns = ["audio_path", "text"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")
    
    # Convert to Dataset
    dataset = Dataset.from_pandas(df)
    
    # Add audio column
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))
    
    logger.info(f"Loaded {len(dataset)} test examples")
    return dataset

def preprocess_dataset(dataset, processor):
    """Preprocess dataset for evaluation"""
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
                logger.error(f"Error loading audio file {audio}: {e}")
                # Create empty features as fallback
                example["input_features"] = torch.zeros(1, 80, 3000)
        
        # Keep the reference text
        example["reference"] = example["text"]
        
        return example
    
    # Process dataset
    logger.info("Preprocessing dataset")
    return dataset.map(prepare_example)

def transcribe_batch(model, processor, batch, device, generation_config):
    """Transcribe a batch of audio files"""
    input_features = torch.stack(batch["input_features"]).to(device)
    
    # Set proper decoder IDs
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=generation_config.get("language", "en"),
        task=generation_config.get("task", "transcribe")
    )
    
    # Generate transcriptions
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=generation_config.get("max_length", 225),
            num_beams=generation_config.get("num_beams", 5),
            temperature=generation_config.get("temperature", 0.0)
        )
    
    # Decode predictions
    predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    return predictions

def evaluate_model(model, processor, dataset, device, batch_size=16, generation_config=None):
    """Evaluate model on test dataset"""
    if generation_config is None:
        generation_config = {
            "max_length": 225,
            "num_beams": 5,
            "temperature": 0.0,
            "language": "en",
            "task": "transcribe"
        }
    
    # Create batches
    dataset_len = len(dataset)
    all_predictions = []
    all_references = []
    
    # Process in batches
    logger.info(f"Evaluating model on {dataset_len} examples")
    for i in tqdm(range(0, dataset_len, batch_size)):
        batch = dataset[i:min(i + batch_size, dataset_len)]
        predictions = transcribe_batch(model, processor, batch, device, generation_config)
        
        all_predictions.extend(predictions)
        all_references.extend([example["reference"] for example in batch])
    
    # Calculate metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    wer = wer_metric.compute(predictions=all_predictions, references=all_references)
    cer = cer_metric.compute(predictions=all_predictions, references=all_references)
    
    # Calculate fuzzy match similarity scores
    similarity_scores = [fuzz.ratio(pred, ref) / 100.0 for pred, ref in zip(all_predictions, all_references)]
    avg_similarity = np.mean(similarity_scores)
    
    # Create detailed results
    results = []
    for i, (pred, ref) in enumerate(zip(all_predictions, all_references)):
        results.append({
            "index": i,
            "prediction": pred,
            "reference": ref,
            "similarity": similarity_scores[i],
            "perfect_match": pred.strip() == ref.strip(),
            "wer": wer_metric.compute(predictions=[pred], references=[ref])
        })
    
    # Sort by similarity (worst first)
    results.sort(key=lambda x: x["similarity"])
    
    # Calculate accuracy
    perfect_matches = sum(1 for r in results if r["perfect_match"])
    accuracy = perfect_matches / len(results) if results else 0
    
    # Analyze errors by command type
    command_errors = defaultdict(lambda: {"total": 0, "errors": 0})
    for result in results:
        # Extract the commands from the reference text
        ref_commands = result["reference"].split(". ")
        for cmd in ref_commands:
            if cmd:
                # Extract player number and action type
                parts = cmd.strip().split(" ")
                if len(parts) >= 2:
                    action_type = " ".join(parts[1:])
                    command_errors[action_type]["total"] += 1
                    if not result["perfect_match"]:
                        command_errors[action_type]["errors"] += 1
    
    # Calculate error rates by command type
    command_error_rates = {}
    for cmd, counts in command_errors.items():
        if counts["total"] > 0:
            command_error_rates[cmd] = counts["errors"] / counts["total"]
    
    metrics = {
        "wer": wer,
        "cer": cer,
        "average_similarity": avg_similarity,
        "accuracy": accuracy,
        "perfect_matches": perfect_matches,
        "total_examples": len(results),
        "command_error_rates": command_error_rates
    }
    
    return metrics, results

def generate_report(metrics, results, output_dir):
    """Generate and save evaluation report and visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed results
    with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Create error distribution histogram
    plt.figure(figsize=(10, 6))
    similarity_scores = [r["similarity"] for r in results]
    plt.hist(similarity_scores, bins=20, alpha=0.7)
    plt.title("Distribution of Similarity Scores")
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "similarity_distribution.png"))
    
    # Create command error rate bar chart
    if "command_error_rates" in metrics and metrics["command_error_rates"]:
        cmd_errors = metrics["command_error_rates"]
        plt.figure(figsize=(12, 8))
        commands = list(cmd_errors.keys())
        error_rates = list(cmd_errors.values())
        
        # Sort by error rate
        sorted_data = sorted(zip(commands, error_rates), key=lambda x: x[1], reverse=True)
        commands = [x[0] for x in sorted_data]
        error_rates = [x[1] for x in sorted_data]
        
        plt.barh(commands, error_rates, alpha=0.7)
        plt.title("Error Rates by Command Type")
        plt.xlabel("Error Rate")
        plt.ylabel("Command")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "command_error_rates.png"))
    
    # Create worst examples report
    worst_examples = results[:10]  # 10 worst examples
    with open(os.path.join(output_dir, "worst_examples.txt"), "w") as f:
        f.write("===== 10 WORST EXAMPLES =====\n\n")
        for i, example in enumerate(worst_examples):
            f.write(f"Example {i+1} (Similarity: {example['similarity']:.2f})\n")
            f.write(f"Reference: {example['reference']}\n")
            f.write(f"Prediction: {example['prediction']}\n")
            f.write("-" * 50 + "\n\n")
    
    # Create text report
    with open(os.path.join(output_dir, "evaluation_report.txt"), "w") as f:
        f.write("===== EVALUATION REPORT =====\n\n")
        f.write(f"Word Error Rate (WER): {metrics['wer']:.4f}\n")
        f.write(f"Character Error Rate (CER): {metrics['cer']:.4f}\n")
        f.write(f"Average Similarity: {metrics['average_similarity']:.4f}\n")
        f.write(f"Perfect Match Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Perfect Matches: {metrics['perfect_matches']} / {metrics['total_examples']}\n\n")
        
        f.write("===== COMMAND TYPE ERROR ANALYSIS =====\n\n")
        for cmd, rate in sorted(metrics["command_error_rates"].items(), key=lambda x: x[1], reverse=True):
            f.write(f"{cmd}: {rate:.4f}\n")
    
    logger.info(f"Evaluation report saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper model on football commands")
    parser.add_argument("--model_path", required=True, help="Path to the fine-tuned model")
    parser.add_argument("--test_csv", required=True, help="Path to test CSV file")
    parser.add_argument("--output_dir", default="evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    
    args = parser.parse_args()
    
    # Load model
    model, processor, device = load_model(args.model_path, args.device)
    
    # Load and preprocess test data
    dataset = load_test_data(args.test_csv)
    dataset = preprocess_dataset(dataset, processor)
    
    # Set up generation config
    generation_config = {
        "max_length": 225,
        "num_beams": args.num_beams,
        "temperature": 0.0,
        "language": "en",
        "task": "transcribe"
    }
    
    # Evaluate model
    metrics, results = evaluate_model(
        model, 
        processor,
        dataset, 
        device,
        batch_size=args.batch_size,
        generation_config=generation_config
    )
    
    # Generate and save report
    generate_report(metrics, results, args.output_dir)
    
    # Print summary
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Word Error Rate (WER): {metrics['wer']:.4f}")
    print(f"Character Error Rate (CER): {metrics['cer']:.4f}")
    print(f"Average Similarity: {metrics['average_similarity']:.4f}")
    print(f"Perfect Match Accuracy: {metrics['accuracy']:.4f}")
    print(f"Perfect Matches: {metrics['perfect_matches']} / {metrics['total_examples']}")
    print(f"Detailed results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
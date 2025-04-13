import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from tqdm import tqdm
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
import re

def load_model_and_processor(model_path):
    """Load fine-tuned model and processor"""
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # Set language and task for generation
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    
    return model, processor

def prepare_dataset(dataset_path, processor):
    """Prepare dataset for evaluation"""
    # Load dataset
    dataset = load_dataset("csv", data_files=dataset_path)["train"]
    
    # Cast audio column to Audio type
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    return dataset

def transcribe_audio(model, processor, audio_array, sampling_rate, device):
    """Transcribe audio using the fine-tuned model"""
    # Process audio
    input_features = processor.feature_extractor(
        audio_array, 
        sampling_rate=sampling_rate, 
        return_tensors="pt"
    ).input_features.to(device)
    
    # Generate transcription
    predicted_ids = model.generate(input_features)
    
    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription[0]

def normalize_text(text):
    """Normalize text for comparison"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except for dot (.)
    text = re.sub(r'[^\w\s.]', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Trim
    text = text.strip()
    
    return text

def sequence_similarity(text1, text2):
    """Calculate sequence similarity between two texts"""
    # Normalize texts
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    
    # Calculate similarity
    matcher = SequenceMatcher(None, text1, text2)
    
    return matcher.ratio()

def evaluate_soccer_actions(predicted, reference):
    """
    Evaluate soccer action predictions
    Format: "<player_number> <action> <result>"
    """
    # Normalize texts
    predicted = normalize_text(predicted)
    reference = normalize_text(reference)
    
    # Split into individual actions
    pred_actions = [a.strip() for a in predicted.split(".") if a.strip()]
    ref_actions = [a.strip() for a in reference.split(".") if a.strip()]
    
    # Count correct predictions
    total_actions = len(ref_actions)
    correct_actions = 0
    
    # Match predictions with references
    for ref_action in ref_actions:
        best_match = None
        best_score = 0
        
        for pred_action in pred_actions:
            score = sequence_similarity(pred_action, ref_action)
            
            if score > best_score:
                best_score = score
                best_match = pred_action
        
        # Consider it correct if similarity > 0.8
        if best_score > 0.8:
            correct_actions += 1
            # Remove matched prediction to avoid double-counting
            if best_match in pred_actions:
                pred_actions.remove(best_match)
    
    # Calculate accuracy
    accuracy = correct_actions / total_actions if total_actions > 0 else 0
    
    return {
        "accuracy": accuracy,
        "correct_actions": correct_actions,
        "total_actions": total_actions,
    }

def evaluate_soccer_model(model, processor, dataset, device, n_samples=None):
    """Evaluate model on soccer dataset"""
    # Limit dataset for testing
    if n_samples is not None:
        dataset = dataset.select(range(min(n_samples, len(dataset))))
    
    # Initialize metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    # List to store detailed results
    results = []
    
    # Process each example
    for example in tqdm(dataset, desc="Evaluating"):
        # Get audio
        audio = example["audio"]
        
        # Get reference
        reference = example["sentence"]
        
        # Transcribe
        prediction = transcribe_audio(
            model, 
            processor, 
            audio["array"], 
            audio["sampling_rate"],
            device
        )
        
        # Calculate metrics
        wer = wer_metric.compute(predictions=[prediction], references=[reference])
        cer = cer_metric.compute(predictions=[prediction], references=[reference])
        
        # Calculate soccer-specific metrics
        soccer_metrics = evaluate_soccer_actions(prediction, reference)
        
        # Store results
        results.append({
            "reference": reference,
            "prediction": prediction,
            "wer": wer,
            "cer": cer,
            "accuracy": soccer_metrics["accuracy"],
            "correct_actions": soccer_metrics["correct_actions"],
            "total_actions": soccer_metrics["total_actions"],
        })
    
    return results

def aggregate_results(results):
    """Aggregate evaluation results"""
    # Calculate average metrics
    avg_wer = np.mean([r["wer"] for r in results])
    avg_cer = np.mean([r["cer"] for r in results])
    avg_accuracy = np.mean([r["accuracy"] for r in results])
    
    # Calculate total actions and correct actions
    total_actions = sum([r["total_actions"] for r in results])
    correct_actions = sum([r["correct_actions"] for r in results])
    overall_accuracy = correct_actions / total_actions if total_actions > 0 else 0
    
    return {
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "avg_accuracy": avg_accuracy,
        "overall_accuracy": overall_accuracy,
        "total_actions": total_actions,
        "correct_actions": correct_actions,
    }

def visualize_results(results, output_dir):
    """Visualize evaluation results"""
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Plot WER distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["wer"], bins=20)
    plt.xlabel("Word Error Rate (WER)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Word Error Rate")
    plt.savefig(os.path.join(output_dir, "wer_distribution.png"))
    plt.close()
    
    # Plot CER distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["cer"], bins=20)
    plt.xlabel("Character Error Rate (CER)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Character Error Rate")
    plt.savefig(os.path.join(output_dir, "cer_distribution.png"))
    plt.close()
    
    # Plot accuracy distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["accuracy"], bins=10)
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.title("Distribution of Soccer Action Accuracy")
    plt.savefig(os.path.join(output_dir, "accuracy_distribution.png"))
    plt.close()
    
    # Create scatter plot of WER vs. accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(df["wer"], df["accuracy"])
    plt.xlabel("Word Error Rate (WER)")
    plt.ylabel("Soccer Action Accuracy")
    plt.title("WER vs. Soccer Action Accuracy")
    plt.savefig(os.path.join(output_dir, "wer_vs_accuracy.png"))
    plt.close()
    
    # Save detailed results to CSV
    df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Whisper model on soccer data")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--test_csv", required=True, help="Path to test CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Load model and processor
    model, processor = load_model_and_processor(args.model_path)
    model = model.to(args.device)
    
    # Prepare dataset
    dataset = prepare_dataset(args.test_csv, processor)
    
    # Evaluate model
    results = evaluate_soccer_model(model, processor, dataset, args.device, args.n_samples)
    
    # Aggregate results
    agg_results = aggregate_results(results)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Average WER: {agg_results['avg_wer']:.4f}")
    print(f"Average CER: {agg_results['avg_cer']:.4f}")
    print(f"Average Soccer Action Accuracy: {agg_results['avg_accuracy']:.4f}")
    print(f"Overall Soccer Action Accuracy: {agg_results['overall_accuracy']:.4f}")
    print(f"Correct Actions: {agg_results['correct_actions']} / {agg_results['total_actions']}")
    
    # Check if target accuracy is achieved
    target_accuracy = 0.95
    if agg_results['overall_accuracy'] >= target_accuracy:
        print(f"\n✅ Target accuracy of {target_accuracy:.2f} achieved!")
    else:
        print(f"\n❌ Target accuracy of {target_accuracy:.2f} not achieved. Current accuracy: {agg_results['overall_accuracy']:.4f}")
    
    # Visualize results
    visualize_results(results, args.output_dir)
    
    # Save aggregate results
    with open(os.path.join(args.output_dir, "evaluation_summary.json"), "w") as f:
        json.dump(agg_results, f, indent=4)
    
    return agg_results
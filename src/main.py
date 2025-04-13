import os
import argparse
import json
import subprocess
import shutil
from datetime import datetime
import sys

def run_command(command, description=None):
    """Run a shell command and handle errors"""
    if description:
        print(f"\n{description}")
        print("=" * 80)
    
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {' '.join(command)}")
        print(f"Error message: {result.stderr}")
        sys.exit(1)
    
    return result.stdout

def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline for fine-tuning Whisper on soccer data")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--transcript_dir", required=True, help="Directory containing transcript files")
    parser.add_argument("--output_dir", default="./output", help="Directory to save outputs")
    parser.add_argument("--config", default="./configs/training_config.json", help="Path to training configuration file")
    parser.add_argument("--skip_data_prep", action="store_true", help="Skip data preparation step")
    parser.add_argument("--skip_training", action="store_true", help="Skip training step")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation step")
    parser.add_argument("--device", default="cuda" if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu", help="Device to use for training")
    
    args = parser.parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    
    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    data_dir = os.path.join(run_dir, "data")
    model_dir = os.path.join(run_dir, "model")
    eval_dir = os.path.join(run_dir, "evaluation")
    
    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Save config to run directory
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Step 1: Data Preparation
    if not args.skip_data_prep:
        run_command(
            ["python", "src/data_preparation.py", 
             "--audio_dir", args.audio_dir,
             "--transcript_dir", args.transcript_dir,
             "--output_dir", data_dir,
             "--no_normalize" if not config["data"]["normalize_audio"] else ""
            ], 
            "Step 1: Data Preparation"
        )
    else:
        print("\nSkipping data preparation step...")
    
    # Step 2: Training
    if not args.skip_training:
        train_csv = os.path.join(data_dir, "hf_train", "dataset.csv")
        val_csv = os.path.join(data_dir, "hf_val", "dataset.csv")
        
        run_command(
            ["python", "src/train.py",
             "--train_csv", train_csv,
             "--val_csv", val_csv,
             "--output_dir", model_dir,
             "--model_name", config["model_name"],
             "--num_epochs", str(config["training"]["num_epochs"]),
             "--batch_size", str(config["training"]["batch_size"]),
             "--learning_rate", str(config["training"]["learning_rate"]),
             "--freeze_encoder" if config["training"]["freeze_encoder"] else "",
             "--gradient_checkpointing" if config["training"]["gradient_checkpointing"] else "",
             "--use_wandb"
            ],
            "Step 2: Training"
        )
    else:
        print("\nSkipping training step...")
    
    # Step 3: Evaluation
    if not args.skip_evaluation:
        test_csv = os.path.join(data_dir, "hf_test", "dataset.csv")
        best_model_path = os.path.join(model_dir, "best_model")
        
        run_command(
            ["python", "src/evaluate.py",
             "--model_path", best_model_path,
             "--test_csv", test_csv,
             "--output_dir", eval_dir,
             "--device", args.device
            ],
            "Step 3: Evaluation"
        )
    else:
        print("\nSkipping evaluation step...")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Pipeline completed successfully!")
    print(f"Run directory: {run_dir}")
    print("=" * 80)
    
    # Load evaluation results if available
    eval_results_path = os.path.join(eval_dir, "evaluation_summary.json")
    if os.path.exists(eval_results_path):
        with open(eval_results_path, "r") as f:
            eval_results = json.load(f)
            
        print("\nEvaluation Results:")
        print(f"Word Error Rate (WER): {eval_results['avg_wer']:.4f}")
        print(f"Character Error Rate (CER): {eval_results['avg_cer']:.4f}")
        print(f"Soccer Action Accuracy: {eval_results['overall_accuracy']:.4f}")
        print(f"Target Accuracy (95%): {'✅ Achieved' if eval_results['overall_accuracy'] >= 0.95 else '❌ Not Achieved'}")
    
    # Print instructions for using the model
    print("\nTo use the fine-tuned model for inference:")
    print(f"python src/inference.py --model_path {os.path.join(model_dir, 'best_model')} --mode file --audio_path <path_to_audio>")
    print("\nFor batch processing:")
    print(f"python src/inference.py --model_path {os.path.join(model_dir, 'best_model')} --mode batch --audio_path <directory_with_audio_files>")

if __name__ == "__main__":
    main()
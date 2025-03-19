import subprocess
import os
from pathlib import Path
import argparse

# Checkpoints to evaluate
CHECKPOINTS = [17500, 20000]

def run_inference_for_checkpoint(checkpoint_step, input_file, base_dir):
    """Run batch inference for a specific checkpoint"""
    checkpoint_dir = os.path.join(base_dir, f"checkpoint-{checkpoint_step}")
    output_dir = f"rapid-test_{checkpoint_step}"
    
    print(f"\nProcessing checkpoint {checkpoint_step}")
    print("-" * 50)
    
    cmd = [
        "python", "batch-inference.py",
        "--input-file", input_file,
        "--checkpoint", checkpoint_dir,
        "--output-dir", output_dir
    ]
    
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run inference on multiple checkpoints")
    parser.add_argument("--input-file", type=str, required=True,
                      help="Input TSV file with test sentences")
    parser.add_argument("--base-dir", type=str, 
                      default="/home/twbgmy/play/marma-tts/mms-rmz/finetuned/run_2024-12-25_18-46",
                      help="Base directory containing checkpoints")
    
    args = parser.parse_args()
    
    print(f"Will process {len(CHECKPOINTS)} checkpoints: {CHECKPOINTS}")
    
    for checkpoint in CHECKPOINTS:
        run_inference_for_checkpoint(checkpoint, args.input_file, args.base_dir)
    
    print("\nAll checkpoints processed!")
    print("\nOutput directories created:")
    for checkpoint in CHECKPOINTS:
        print(f"  rapid-test_{checkpoint}/")

if __name__ == "__main__":
    main()
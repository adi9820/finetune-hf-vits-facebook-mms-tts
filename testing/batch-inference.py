import argparse
from transformers import VitsModel, AutoTokenizer
from safetensors.torch import load_file
import os
import torch
from transformers import pipeline
import scipy.io.wavfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def load_model_from_checkpoint(base_model_path, checkpoint_path):
    """Load model and tokenizer with specific checkpoint weights."""
    # Load base model and tokenizer
    model = VitsModel.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load checkpoint weights
    checkpoint = load_file(os.path.join(checkpoint_path, "model.safetensors"))
    model.load_state_dict(checkpoint, strict=False)
    
    return model, tokenizer

def synthesize_texts(test_data, model, tokenizer, output_dir, device="cpu"):
    """Synthesize texts from test data DataFrame."""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device set to use {device}")
    
    # Create TTS pipeline
    synthesiser = pipeline(
        "text-to-speech", 
        model=model, 
        tokenizer=tokenizer,
        device=device
    )
    
    # Process each row in the test data
    print(f"\nGenerating {len(test_data)} audio samples...")
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Synthesizing"):
        test_id = row['test_id']
        text = row['text']  # Using the text column
        
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            print(f"\nSkipping empty or invalid text for test_id {test_id}")
            continue
            
        # Generate audio
        try:
            speech = synthesiser(text.strip())
            
            # Create output filename using test_id
            output_file = output_dir / f"{test_id}.wav"
            
            # Save audio file
            scipy.io.wavfile.write(
                str(output_file),
                rate=speech["sampling_rate"],
                data=speech["audio"][0]
            )
            
        except Exception as e:
            print(f"\nError processing test_id {test_id}: {text}")
            print(f"Error message: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Batch TTS inference from checkpoint")
    parser.add_argument("--base-model", type=str, 
                      default="/home/twbgmy/play/marma-tts/mms-rmz",
                      help="Path to base model")
    parser.add_argument("--checkpoint", type=str, 
                      default="/home/twbgmy/play/marma-tts/mms-rmz/finetuned/run_2024-12-25_18-46/checkpoint-15000",
                      help="Path to checkpoint directory")
    parser.add_argument("--input-file", type=str, required=True,
                      help="TSV file containing test data (columns: test_id, text)")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Directory to save generated audio files")
    parser.add_argument("--device", type=str, default="cpu",
                      help="Device to use for inference (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.input_file}")
    test_data = pd.read_csv(args.input_file, 
                           names=['test_id', 'text'],  # Just two columns
                           sep='\t')  # Tab-separated
    
    # Load model and tokenizer
    print(f"Loading model from checkpoint {args.checkpoint}")
    model, tokenizer = load_model_from_checkpoint(args.base_model, args.checkpoint)
    
    # Generate audio
    synthesize_texts(test_data, model, tokenizer, args.output_dir, args.device)
    
    print(f"\nDone! Generated samples are saved in {args.output_dir}")
    print(f"Number of samples generated: {len(list(Path(args.output_dir).glob('*.wav')))}")

if __name__ == "__main__":
    main()
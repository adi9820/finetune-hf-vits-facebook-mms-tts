import pandas as pd
import argparse
from pathlib import Path
import os 

def create_evaluation_sheet(test_file, no_models):
    """Create evaluation sheet template incorporating test file contents"""
    
    # Read test cases
    test_data = pd.read_csv(test_file, sep='\t', names=['test_id', 'text'])

    test_file_id = os.path.splitext(test_file)[0]

    
    # Define models
    models = [chr(65 + i) for i in range(int(no_models))]
    
    # Create evaluation template
    rows = []
    
    # Add header row for descriptions
    rows.append({
        'Model': 'Description',
        'File_ID': 'Test file identifier',
        'Text': 'Transcription of audio',
        'Pronunciation_Accuracy': 'Score 1-5: How accurately are words pronounced?',
        'Word_Recognition': 'Score 1-5: How many words can be clearly recognized?',
        'Overall_Naturalness': 'Score 1-5: How natural does the speech sound?',
        'Notes': 'Note any specific pronunciation errors or issues'
    })
    
    # Add rows for each model and test case combination
    for _, test_row in test_data.iterrows():
        for model in models:
            rows.append({
                'Model': model,
                'File_ID': test_row['test_id'],
                'Text': test_row['text'],
                'Pronunciation_Accuracy': '',
                'Word_Recognition': '',
                'Overall_Naturalness': '',
                'Notes': ''
            })
        # Add a blank row between different test cases
        rows.append({
            'Model': '',
            'File_ID': '',
            'Text': '',
            'Pronunciation_Accuracy': '',
            'Word_Recognition': '',
            'Overall_Naturalness': '',
            'Notes': ''
        })
    
    # Create DataFrame and save to TSV
    df = pd.DataFrame(rows)
    output_file_name = test_file_id + "_evaluation-sheet.tsv"
    df.to_csv(output_file_name, sep='\t', index=False)
    
    print("Created evaluation sheet with the following structure:")
    print("\nScoring guide:")
    print("1 = Very poor")
    print("2 = Poor")
    print("3 = Fair")
    print("4 = Good")
    print("5 = Excellent")
    
    print("\nSheet contains following models:", models)
    
    print(f"\nTotal number of test cases: {len(test_data)}")
    print(f"Total number of evaluations needed: {len(test_data) * len(models)}")
    print(f"Output file: {output_file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create evaluation sheet template")
    parser.add_argument("--test-file", type=str, required=True,
                      help="Input TSV file with test sentences")
    parser.add_argument("--no-models", type=str, required=True,
                      help="Number of models to compare")
    
    args = parser.parse_args()
    create_evaluation_sheet(args.test_file, args.no_models)
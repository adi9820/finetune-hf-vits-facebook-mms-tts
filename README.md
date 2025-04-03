# Finetune VITS and MMS for Marma TTS

## Introduction

[VITS](https://huggingface.co/docs/transformers/model_doc/vits) is a lightweight, low-latency model for English text-to-speech (TTS). Massively Multilingual Speech ([MMS](https://huggingface.co/docs/transformers/model_doc/mms#speech-synthesis-tts)) is an extension of VITS for multilingual TTS, that supports over [1100 languages](https://huggingface.co/facebook/mms-tts#supported-languages).

Both use the same underlying VITS architecture, consisting of a discriminator and a generator for GAN-based training. They differ in their tokenizers: the VITS tokenizer transforms English input text into phonemes, while the MMS tokenizer transforms input text into character-based tokens.

This repository contains code adapted from [HuggingFace's VITS fine-tuning repository](https://github.com/ylacombe/finetune-hf-vits) with specific modifications for Marma language support.

## License
The VITS checkpoints are released under the permissive [MIT License](https://opensource.org/license/mit/). The MMS checkpoints, on the other hand, are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), a non-commercial license. 

**Note:** Any finetuned models derived from these checkpoints will inherit the same licenses as their respective base models. **Please ensure that you comply with the terms of the applicable license when using or distributing these models.**

## Requirements

1. Clone this repository and install common requirements.

```sh
git clone https://github.com/translatorswb/finetune-hf-vits-marma.git
cd finetune-hf-vits-marma
pip install -r requirements.txt
```

2. Link your Hugging Face account (optional but recommended for saving models)

```bash
git config --global credential.helper store
huggingface-cli login
```
And then enter an authentication token from https://huggingface.co/settings/tokens. Create a new token if you do not have one already. You should make sure that this token has "write" privileges.

3. Build the monotonic alignment search function using cython. This is absolutely necessary since the Python-native-version is awfully slow.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..
```

## Model Setup

```bash
# Create a directory for MMS-Marma
mkdir mms-rmz
cd mms-rmz

# Convert the model checkpoint with discriminator for Marma (using Myanmar as closest relative)
python ../convert_original_discriminator_checkpoint.py \
  --language_code mya \
  --pytorch_dump_folder_path . \
  --push_to_hub false

# Create directory for finetuned models
mkdir -p finetuned
cd ..
```

## Fine-tuning

There are two ways to run the finetuning script, both using command lines. Note that you only need one GPU to finetune VITS/MMS as the models are really lightweight (83M parameters).

### Using the Training Script

This repository includes a convenient script for training:

```bash
# Begin a new training session
./finetune_rmz.sh new

# Monitor training progress
tail -f mms-rmz/finetuned/run_<timestamp>/logs/training.log

# Resume training from checkpoint if needed
./finetune_rmz.sh continue <path-to-checkpoint-dir>
```

### Using Config File (Alternative)

The `finetune_mms_rmz.json` file contains important configuration parameters that can be modified to customize the training process:

- Dataset parameters (dataset_name, split names, column names)
- Training hyperparameters (learning_rate, batch_size, etc.)
- Loss weights (weight_duration, weight_kl, weight_mel, weight_disc, weight_gen, weight_fmaps)
- Logging and output directories

You can launch training directly with:

```sh
accelerate launch run_vits_finetuning.py ./finetune_mms_rmz.json
```

## Training Data Preparation

For best results, prepare your data following these steps:

1. Collect and normalize Marma text data
2. Record audio with a good quality microphone
3. Process recordings to ensure consistent quality
4. Create a dataset with the format expected by the training scripts
5. Upload to HuggingFace using appropriate scripts

The dataset should be structured as a CSV file with columns for audio paths and transcript text.

## Model Selection and Evaluation

After training, select the best checkpoint using our evaluation tools:

```bash
# Analyze training logs
cd analyze_train
python analyze_log.py ../../mms-rmz/finetuned/run_<timestamp>/logs/training.log run_<timestamp>

# Run inference on test sentences with specific checkpoints
cd testing
python batch-inference.py \
  --base-model ../mms-rmz \
  --checkpoint ../mms-rmz/finetuned/run_<timestamp>/checkpoint-<step>/ \
  --input-file testing/rapid-test.tsv \
  --output-dir testing/rapid-test-output

# OR test multiple checkpoints at once
python multi-checkpoint-inference.py \
  --input-file rapid-test.tsv \
  --base-dir ../../mms-rmz/finetuned/run_<timestamp>/

# Create evaluation sheets for subjective assessment
python evaluation-sheet-maker.py \
  --test-file rapid-test.tsv \
  --no-models <number-of-models>
```

The analysis will help identify the best performing checkpoint based on:
- Mel Loss (speech quality)
- Duration Loss (timing accuracy)
- KL Loss (voice diversity)
- Overall weighted score (combined metrics)

## Inference

You can use a finetuned model via the Text-to-Speech (TTS) [pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline) in just a few lines of code.
Just replace the model path with your own model id or path to the model.

```python
from transformers import pipeline
import scipy

model_id = "CLEAR-Global/marmaspeak-tts-v1"  # replace with your model
synthesiser = pipeline("text-to-speech", model_id) # add device=0 if you want to use a GPU

speech = synthesiser("ကိုတော် ဇာမာ နီရေလည်း၊")  # Marma text example

scipy.io.wavfile.write("output.wav", rate=speech["sampling_rate"], data=speech["audio"][0])
```

## Testing Scripts

This repository includes several testing scripts in the `testing/` directory:

- `batch-inference.py`: Generate speech for a batch of sentences from a specific checkpoint
- `multi-checkpoint-inference.py`: Test multiple checkpoints on the same test set
- `evaluation-sheet-maker.py`: Create TSV evaluation sheets for comparing model outputs

These scripts help with both objective and subjective evaluation of model outputs.


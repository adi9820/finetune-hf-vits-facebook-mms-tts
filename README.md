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
git clone https://github.com/adi9820/finetune-hf-vits-facebook-mms-tts
cd finetune-hf-vits-facebook-mms-tts
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
### Run only once

```bash
# Create a directory for MMS-TTS model
mkdir mms-tts
cd mms-tts

# Convert the model checkpoint with discriminator for your language. Do it only once as this clones the model you want to finetune in your hf profile.
python ../convert_original_discriminator_checkpoint.py \
  --language_code hin \
  --pytorch_dump_folder_path . \
  --push_to_hub <name your model>

# Create directory for finetuned models
mkdir -p finetuned
cd ..
```

## Fine-tuning

There are two ways to run the finetuning script, both using command lines. Note that you only need one GPU to finetune VITS/MMS as the models are really lightweight (83M parameters).

### Using Config File 

The `finetune_mms_tts.json` file contains important configuration parameters that can be modified to customize the training process:

- Dataset parameters (dataset_name, split names, column names)
- Training hyperparameters (learning_rate, batch_size, etc.)
- Loss weights (weight_duration, weight_kl, weight_mel, weight_disc, weight_gen, weight_fmaps)
- Logging and output directories

You can launch training directly with:

```sh
accelerate launch run_vits_finetuning.py ./finetune_mms_tts.json
```

### Using the Training Script (Alternative)

This repository includes a convenient script for training:

```bash
# Begin a new training session
./finetune_rmz.sh new

# Monitor training progress
tail -f mms-tts/finetuned/run_<timestamp>/logs/training.log

# Resume training from checkpoint if needed
./finetune_rmz.sh continue <path-to-checkpoint-dir>
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
python analyze_log.py ../../mms-tts/finetuned/run_<timestamp>/logs/training.log run_<timestamp>

# Run inference on test sentences with specific checkpoints
cd testing
python batch-inference.py \
  --base-model ../mms-tts \
  --checkpoint ../mms-tts/finetuned/run_<timestamp>/checkpoint-<step>/ \
  --input-file testing/rapid-test.tsv \
  --output-dir testing/rapid-test-output

# OR test multiple checkpoints at once
python multi-checkpoint-inference.py \
  --input-file rapid-test.tsv \
  --base-dir ../../mms-tts/finetuned/run_<timestamp>/

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

To use your finetuned model replace the model path with your own model id or path to the model.

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import soundfile as sf
import time

model_id = "------"  # replace with your model
# Load the Hindi TTS model and tokenizer
tokenizer = VitsTokenizer.from_pretrained("model_id")
model = VitsModel.from_pretrained("model_id")

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Hindi text
text = "आज का दिन बहुत ही खास है, क्योंकि हम एक नई तकनीक के माध्यम से यह जानने जा रहे हैं कि मशीनें किस प्रकार से मानव भाषा को समझ सकती हैं और उसे ध्वनि में परिवर्तित कर सकती हैं। यह तकनीक न केवल शिक्षा के क्षेत्र में (जैसे कि कक्षा ६ से १२ तक के छात्रों के लिए), बल्कि दैनिक जीवन के कई पहलुओं में भी उपयोगी हो सकती है, जैसे कि ८० लाख दृष्टिहीन व्यक्तियों की सहायता करना या २२ आधिकारिक भारतीय भाषाओं में संवाद करना।"

# Tokenize and move inputs to device
inputs = tokenizer(text, return_tensors="pt")
inputs = {key: val.to(device) for key, val in inputs.items()}

# Optional: Set seed for reproducibility
set_seed(55)

# Force GPU sync before starting timing
if device.type == "cuda":
    torch.cuda.synchronize()
start_time = time.time()

# Generate speech waveform
with torch.no_grad():
    outputs = model(**inputs)

# Force GPU sync after model inference
if device.type == "cuda":
    torch.cuda.synchronize()
end_time = time.time()

mrt = end_time - start_time

# Get waveform from output (move to CPU for saving)
waveform = outputs.waveform[0].cpu()
sampling_rate = model.config.sampling_rate

# Save audio
sf.write("facebook_mms_tts_hin.wav", waveform.numpy(), samplerate=sampling_rate)
print("✅ Saved speech to 'output_hindi.wav'")
print(f"🕒 Model response time: {mrt:.2f} seconds")
```

## Testing Scripts

This repository includes several testing scripts in the `testing/` directory:

- `batch-inference.py`: Generate speech for a batch of sentences from a specific checkpoint
- `multi-checkpoint-inference.py`: Test multiple checkpoints on the same test set
- `evaluation-sheet-maker.py`: Create TSV evaluation sheets for comparing model outputs

These scripts help with both objective and subjective evaluation of model outputs.

## Training Hyperparameters in JSON

| Parameter           | What it does?                                                                                                                 | 
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `"learning_rate"`   | Defines how quickly the model updates its weights                                                                             |
| `"adam_beta1"`      | Controls first-order gradient in Adam optimizer (high value for better stability but makes model less responsive to new data) |
| `"adam_beta2"`      | Controls second-order gradient in Adam (high value for better stability but makes model less responsive to new data)          |
| `"weight_disc"`     | GAN adversarial loss weight (high value for making voice more like trainig voice but will decrease stability in trainig)      |
| `"weight_fmaps"`    | Feature matching loss weight (high value for stability and realism but slower convergence)                                    |
| `"weight_gen"`      | Generator loss weight (drives quality vs. discriminator)                                                                      |
| `"weight_kl"`       | KL divergence loss weight (high value more stable voice but makes voice less expressive)                                      |
| `"weight_duration"` | Duration prediction loss weight (high value better rhythm but makes voice less expressive)                                    |
| `"weight_mel"`      | Mel-spectrogram loss weight (high value more accurate speech but makes voice robotic voice)                                   |

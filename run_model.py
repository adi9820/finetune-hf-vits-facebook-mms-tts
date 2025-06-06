import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import soundfile as sf
import time

# Load the Hindi TTS model and tokenizer
tokenizer = VitsTokenizer.from_pretrained("./20250606103536")
model = VitsModel.from_pretrained("./20250606103536")

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Hindi text
text = """
    एक आदमी डॉक्टर के पास गया और बोला:
    डॉक्टर साहब, मुझे भूलने की बीमारी हो गई है!

    डॉक्टर ने पूछा:
    कब से?

    आदमी बोला:
    कब से क्या?
    """




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
sf.write("./20250606103536/audio_out.wav", waveform.numpy(), samplerate=sampling_rate)
print("✅ Saved speech to 'output_hindi.wav'")
print(f"🕒 Model response time: {mrt:.2f} seconds")

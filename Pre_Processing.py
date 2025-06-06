import os
import torch
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps
import soundfile as sf
from pydub import AudioSegment
import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from indic_transliteration.sanscript import transliterate, ITRANS, DEVANAGARI
from datasets import Dataset, DatasetDict, Audio
import io


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
batch_size = 4


# Load Whisper
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id)


pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


def chunk_audio(path, chunk_length_sec=22, buffer_sec=1.5):
    
    audio = AudioSegment.from_file(path)
    chunk_length_ms = chunk_length_sec * 1000
    buffer_ms = buffer_sec * 1000

    chunks = []
    for start_ms in range(0, len(audio), chunk_length_ms):
        end_ms = min(start_ms + chunk_length_ms, len(audio))
        chunk = audio[start_ms:end_ms]
        padded = AudioSegment.silent(duration=buffer_ms) + chunk + AudioSegment.silent(duration=buffer_ms)
        chunks.append(padded)

    return chunks


def clean_chunk_with_silero(chunk_audiosegment, vad_model, target_sr=16000, threshold=0.75):
    
    # Convert pydub AudioSegment to waveform tensor
    samples = chunk_audiosegment.get_array_of_samples()
    waveform = torch.tensor(samples).float()

    # Normalize from int16 to float32 (-1 to 1)
    waveform /= 32768.0

    # Pydub default is stereo or mono? Convert to mono if stereo
    if chunk_audiosegment.channels > 1:
        waveform = waveform.view(-1, chunk_audiosegment.channels).mean(dim=1)

    # Resample if needed
    orig_sr = chunk_audiosegment.frame_rate
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    # Run VAD to get speech timestamps
    speech_timestamps = get_speech_timestamps(
        waveform,
        vad_model,
        sampling_rate=target_sr,
        return_seconds=False,
        threshold=threshold,
        min_speech_duration_ms=200,
        min_silence_duration_ms=100,
        speech_pad_ms=50
    )

    # Extract speech segments and concatenate
    speech_segments = [waveform[ts["start"]:ts["end"]] for ts in speech_timestamps]

    if speech_segments:
        cleaned_audio = torch.cat(speech_segments)
    else:
        cleaned_audio = torch.zeros(1)

    return cleaned_audio


def clean_n_chunk(input, chunk_length_sec=22, buffer_sec=1.5, target_sr=16000):
    # Load VAD model once
    vad_model = load_silero_vad()

    # Step 1: chunk input audio
    chunks = chunk_audio(input, chunk_length_sec, buffer_sec)
    print(f"Chunks created: {len(chunks)}")

    cleaned_chunks = []

    # Step 2: clean each chunk and store in list (no saving to disk)
    for i, chunk in enumerate(chunks):
        cleaned_waveform = clean_chunk_with_silero(chunk, vad_model, target_sr)
        cleaned_chunks.append(cleaned_waveform)  # Store tensor or numpy array

    return cleaned_chunks


def literal_transliterate(text):
    try:
        return transliterate(text, ITRANS, DEVANAGARI)
    except Exception as e:
        print(f"⚠️ Transliteration failed: {e}")
        return text

    
def transcribe(waveforms):
    # Transcribe with Batching
    results = []
    audio_inputs = []

    print(f"🔁 Total files to transcribe: {len(waveforms)}")

    for waveform in tqdm(waveforms, desc="Preparing audio"):
        try:
            if isinstance (waveform, torch.Tensor):
                waveform = waveform.squeeze().numpy()
            
            audio_input = {"array": waveform, "sampling_rate": 16000}
            audio_inputs.append(audio_input)

        except Exception as e:
            print(f"❌ Error preparing audio: {e}")

    # Process in batches
    print("🚀 Starting batch transcription...")
    for i in tqdm(range(0, len(audio_inputs), batch_size), desc="Transcribing"):
        batch = audio_inputs[i:i+batch_size]

        try:
            outputs = pipe(batch, generate_kwargs={"language": "hi"})

            for idx, output in enumerate(outputs):
                raw_text = output["text"]
                translated_text = literal_transliterate(raw_text)
                results.append({"index": i+idx, "transcription": translated_text})

        except Exception as e:
            print(f"❌ Batch error: {e}")

    df = pd.DataFrame(results)

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    return buffer, waveforms


def upload_to_hf(save_dir, out_parquet, wave_forms, user_id):
    os.makedirs(save_dir, exist_ok=True)

    # Read the in-memory parquet into a DataFrame
    if isinstance(out_parquet, io.BytesIO):
        buffer = out_parquet
    else:
        buffer = io.BytesIO(out_parquet)

    df = pd.read_parquet(buffer)

    assert len(df) == len(wave_forms), "Mismatch between transcripts and waveforms!"

    # Save each waveform to the given temp directory
    audio_paths = []
    for i, waveform in enumerate(wave_forms):
        file_path = os.path.join(save_dir, f"{i:04d}.wav")
        torchaudio.save(file_path, waveform.unsqueeze(0), 16000)
        audio_paths.append(file_path)

    # Prepare HF dataset DataFrame
    df["audio"] = audio_paths
    df = df[["audio", "transcription"]].rename(columns={"transcription": "text"})

    # Convert to HF Dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Wrap and push
    dataset_dict = DatasetDict({"train": dataset})
    hf_repo = f"Anjan9320/{user_id.lower()}"  # optional: customize per user
    dataset_dict.push_to_hub(hf_repo, max_shard_size="500MB")
    print(f"Uploaded dataset to Hugging Face repo: {hf_repo}")


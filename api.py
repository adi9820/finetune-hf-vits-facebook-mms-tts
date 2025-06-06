from flask import Flask, request, render_template_string
import os
import tempfile
from pydub import AudioSegment
import shutil
import time
from datetime import datetime

from Pre_Processing import clean_n_chunk, transcribe, upload_to_hf
from finetune import run_commands

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For Flask sessions (not related to user id)

HTML_FORM = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Audio Upload</title>
</head>
<body>
  <h1>Upload Audio File</h1>
  <form method="post" enctype="multipart/form-data" action="{{ url_for('upload') }}">
    <input type="file" name="audio" required />
    <input type="submit" value="Upload" />
  </form>
  
  {% if message_upload %}
    <p><strong>{{ message_upload }}</strong></p>
  {% endif %}

</body>
</html>
"""

ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_FORM)

@app.route('/upload', methods=['POST'])
def upload():
    print(f"[INFO] Upload route hit. Method: {request.method}, From: {request.remote_addr}")
    
    if 'audio' not in request.files:
        return render_template_string(HTML_FORM, message_upload="No file part")

    file = request.files['audio']
    if file.filename == '':
        return render_template_string(HTML_FORM, message_upload="No selected file")

    if not allowed_file(file.filename):
        return render_template_string(HTML_FORM, message_upload="Invalid file type. Only mp3, mp4, or wav allowed.")
    
    # Generate unique user ID based on current date-time (numbers only)
    user_id = datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"[INFO] Generated user_id: {user_id}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save original uploaded file
            original_path = os.path.join(temp_dir, "input_original")
            file.save(original_path)
            print(f"Original file saved at: {original_path}")

            # Convert to WAV
            audio = AudioSegment.from_file(original_path)
            wav_path = os.path.join(temp_dir, "converted.wav")
            audio.export(wav_path, format="wav")
            print(f"Converted WAV saved at: {wav_path}")

            # Step 1: Chunk the wav file
            chunk_dir = clean_n_chunk(wav_path)  # returns directory path like user_id/chunks
            print("Chunking complete.")

            # Step 2: Preprocess chunks
            parquet_, wave = transcribe(chunk_dir)
            print("Pre-processing complete.")

            # Create subdirectory for storing wav files for upload
            wav_save_dir = os.path.join(temp_dir, "wav_chunks")
            os.makedirs(wav_save_dir, exist_ok=True)

            # Step 3: Upload to HuggingFace Hub
            upload_to_hf(wav_save_dir, parquet_, wave, user_id)
            print("Uploaded to HF.")

            time.sleep(30)  # Wait before finetuning

            # Step 4: Finetune
            run_commands(user_id)
            print("Fine-tuning started.")

            message_upload = f"✅ Success! Your model is being fine-tuned with user ID {user_id}."

    except Exception as e:
        print(f"Error: {e}")
        message_upload = f"❌ Error during processing: {str(e)}"

    return render_template_string(HTML_FORM, message_upload=message_upload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


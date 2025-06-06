import subprocess
import json
import os

def generate_temp_json(username, template_path="finetune_mms_tts.json", temp_path="finetune_temp.json"):
    with open(template_path, 'r') as f:
        config = json.load(f)

    def replace_placeholders(obj):
        if isinstance(obj, str):
            return obj.replace("{username}", username)
        elif isinstance(obj, dict):
            return {k: replace_placeholders(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_placeholders(i) for i in obj]
        return obj

    updated_config = replace_placeholders(config)

    with open(temp_path, 'w') as f:
        json.dump(updated_config, f, indent=4)

    return temp_path

def run_commands(username):
    temp_json = generate_temp_json(username)

    subprocess.run(['python', 'setup.py', 'build_ext', '--inplace'], cwd='monotonic_align', check=True)

    subprocess.run(['python', 'run_vits_finetuning.py', temp_json], check=True)

    # Optional: clean up
    os.remove(temp_json)

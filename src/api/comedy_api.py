#!/usr/bin/env python3
"""
Simple API server for the comedy generator frontend
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

class ComedyGenerator:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_python = self.project_root / "venv" / "bin" / "python"

    def run_python_script(self, script_name, args=None):
        """Run a Python script and capture output"""
        cmd = [str(self.venv_python), str(self.project_root / script_name)]

        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Timeout after 60 seconds", 1
        except Exception as e:
            return "", str(e), 1

    def generate_riff(self, topic):
        """Generate a comedy riff"""
        logger.info(f"Generating riff for topic: {topic}")

        # Create a temporary script that generates with the topic
        script_content = f'''#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def generate_riff(topic):
    model_path = "/Users/RRL_1/.llama/checkpoints/Llama3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = f"Hey guys, welcome back to the podcast. Today we're talking about {{topic}}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "general topics"
    result = generate_riff(topic)
    print(result)
'''

        # Write to temp file and run
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            temp_script = f.name

        try:
            stdout, stderr, code = self.run_python_script(temp_script, [topic])
            if code == 0:
                return stdout.strip()
            else:
                logger.error(f"Script failed: {stderr}")
                return f"Error generating riff: {stderr}"
        finally:
            os.unlink(temp_script)

    def generate_joke(self, topic):
        """Generate a structured joke"""
        logger.info(f"Generating joke for topic: {topic}")

        # Create a temporary script that generates with the topic
        script_content = f'''#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import re

def generate_joke(topic):
    model_path = "/Users/RRL_1/.llama/checkpoints/Llama3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = f"Hey guys, let me tell you a joke about {{topic}}:\\n\\nPremise:"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    content = generated_text[len(prompt):].strip()

    # Try to structure the joke
    lines = content.split('\\n')
    premise = ""
    punchline = ""
    tags = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "premise" in line.lower() and not premise:
            premise = line.replace("Premise:", "").strip()
        elif "punchline" in line.lower() and not punchline:
            punchline = line.replace("Punchline:", "").strip()
        elif "tag" in line.lower() or len(line) > 20:
            if not line.startswith(("Premise", "Punchline")):
                tags.append(line)

    return {{
        "premise": premise or content[:100] + "...",
        "punchline": punchline or content[50:150] + "...",
        "initial_tag": tags[0] if tags else "That's just how it is!",
        "alternate_angle": tags[1] if len(tags) > 1 else "You know what I mean?",
        "additional_tags": tags[2:] if len(tags) > 2 else ["Classic!", "Been there!"]
    }}

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "life"
    result = generate_joke(topic)
    print(result)
'''

        # Write to temp file and run
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            temp_script = f.name

        try:
            stdout, stderr, code = self.run_python_script(temp_script, [topic])
            if code == 0:
                try:
                    return eval(stdout.strip())  # Safe since we control the output
                except:
                    return {{
                        "premise": stdout.strip()[:100] + "...",
                        "punchline": "That's just life!",
                        "initial_tag": "You know what I mean?",
                        "alternate_angle": "Classic situation!",
                        "additional_tags": ["Been there!", "Done that!"]
                    }}
            else:
                logger.error(f"Script failed: {stderr}")
                return {{
                    "premise": f"Error: {stderr}",
                    "punchline": "Sorry, couldn't generate joke",
                    "initial_tag": "",
                    "alternate_angle": "",
                    "additional_tags": []
                }}
        finally:
            os.unlink(temp_script)

generator = ComedyGenerator()

@app.route('/riff', methods=['POST'])
def generate_riff():
    data = request.get_json()
    topic = data.get('topic', 'general topics')

    riff = generator.generate_riff(topic)
    return jsonify({'riff': riff})

@app.route('/joke', methods=['POST'])
def generate_joke():
    data = request.get_json()
    topic = data.get('topic', 'life')

    joke = generator.generate_joke(topic)
    return jsonify({'joke': joke})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("🎭 Starting Llama 3.1-8B Comedy API Server...")
    print("Frontend will be available at: http://localhost:3000")
    print("API endpoints:")
    print("  POST /riff - Generate comedy riff")
    print("  POST /joke - Generate structured joke")
    print("  GET /health - Health check")
    print()
    app.run(debug=True, port=8000)

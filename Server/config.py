# config.py

import os
import torch
from openai import OpenAI

# base model and LoRA repo on Hugging Face
BASE_MODEL = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
LORA_REPO = os.getenv("LORA_REPO_ID", "MiaMao/Autonomous-Insert-LoRA")

# LoRA subfolders
PERSONA_SUBFOLDER = "Personality/checkpoint-32899"
SCENE_SUBFOLDER = "Scene/checkpoint-35821"
TOPIC_WILL_SUBFOLDER = "Topic/Willingness/checkpoint-2500"
STRATEGY_SUBFOLDER = "Topic/Strategy/checkpoint-26250"

# strategy labels
STRATEGY_LABELS = ["comforting", "rational_analysis", "risk_alert"]

MAX_LENGTH = 256

# devices
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRATEGY_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OpenAI client
client = OpenAI()

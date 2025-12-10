

# Autonomous Insert Agent - Python Backend

This repository contains the Python inference backend linking Hugging Face models with a Unity environment. It powers an autonomous agent that evaluates conversational context to determine speaking willingness and generate strategic, short English responses in multi-party scenarios.

The system uses **Qwen2.5-7B-Instruct** as the base model, loaded with specialized LoRA heads for willingness estimation (based on Persona, Scene, and Topic) and strategy selection.

# Model Checkpoints
The LoRA weights are hosted on Hugging Face. Access the repository directly here:
[Hugging Face Model Repo](https://huggingface.co/MiaMao/Autonomous-Insert-LoRA)

-----

## System Visualization

### 1\. Unity Integration View

*(Placeholder for screenshot showing the agent's speech bubble in the Unity scene)*

### 2\. Python Backend Runtime

*(Placeholder for screenshot showing the Python Tkinter GUI and server console logs)*

-----

## Project Structure

```text
.
├── server/
│   ├── config.py        # Environment configuration, model IDs, and device management
│   ├── inference.py     # Base model + multi-LoRA loading and inference pipelines
│   └── server.py        # Main entry: WebSocket servers (TX/RX) and Tkinter UI thread
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

### Prerequisites

  * Python 3.8+
  * CUDA-enabled GPU recommended for inference speed.
  * OpenAI API Key (for generation).

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Set the following environment variables before execution.

| Variable | Description | Default (if unset) |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | **Required.** Your OpenAI API key. | - |
| `BASE_MODEL_ID` | Base LLM path on Hugging Face. | `Qwen/Qwen2.5-7B-Instruct` |
| `LORA_REPO_ID` | LoRA weights path on Hugging Face. | `MiaMao/Autonomous-Insert-LoRA` |

### Execution

```bash
cd server
python server.py
```

On startup, the script will initialize the models and launch two WebSocket endpoints:

  * `ws://127.0.0.1:8765`: Receives context JSON from Unity.
  * `ws://127.0.0.1:8766`: Sends generated utterance JSON to Unity.


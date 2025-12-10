# inference.py

import time
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

import config


# ----- load tokenizer and base model + LoRA heads -----

print("Loading tokenizer from:", config.BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base regression model...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    config.BASE_MODEL,
    num_labels=1,
    torch_dtype=torch.float16 if config.DEVICE.type == "cuda" else torch.float32,
)
base_model.config.pad_token_id = tokenizer.pad_token_id

print("Loading persona LoRA...")
reg_model = PeftModel.from_pretrained(
    base_model,
    config.LORA_REPO,
    adapter_name="persona",
    subfolder=config.PERSONA_SUBFOLDER,
)

print("Loading scene LoRA...")
reg_model.load_adapter(
    config.LORA_REPO,
    adapter_name="scene",
    subfolder=config.SCENE_SUBFOLDER,
)

print("Loading topic willingness LoRA...")
reg_model.load_adapter(
    config.LORA_REPO,
    adapter_name="topic",
    subfolder=config.TOPIC_WILL_SUBFOLDER,
)

reg_model.to(config.DEVICE)
reg_model.eval()
print("Regression model with 3 LoRA heads ready on:", config.DEVICE)

# strategy model is lazy-loaded
strategy_model = None


# ----- willingness helpers -----

@torch.inference_mode()
def run_willingness_with_logs(adapter_name: str, text: str):
    logs = []
    text = (text or "").strip()
    if not text:
        logs.append(f"[{adapter_name}] empty text, return 0.0")
        return 0.0, "\n".join(logs)

    logs.append(f"[{adapter_name}] set_adapter & tokenize, len={len(text)}")
    reg_model.set_adapter(adapter_name)

    enc = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.MAX_LENGTH,
    )
    enc = {k: v.to(config.DEVICE) for k, v in enc.items()}

    t0 = time.time()
    logs.append(f"[{adapter_name}] forward start")
    logits = reg_model(**enc).logits.squeeze(-1).item()
    t1 = time.time()
    logs.append(f"[{adapter_name}] forward done, raw={logits}, time={t1 - t0:.3f}s")

    val = float(logits)
    val = max(0.0, min(1.0, val))
    return val, "\n".join(logs)


def build_persona_text(persona_raw: str, profile_json: str, utterance: str) -> str:
    persona_raw = (persona_raw or "").strip()
    utterance = (utterance or "").strip()

    profile = profile_json.strip() if profile_json else ""
    try:
        profile_obj = json.loads(profile)
        profile = json.dumps(profile_obj, ensure_ascii=False)
    except Exception:
        pass

    parts = []
    if persona_raw:
        parts.append(f"[PERSONA_RAW] {persona_raw}")
    if profile:
        parts.append(f"[PROFILE] {profile}")
    if utterance:
        parts.append(f"[UTTERANCE] {utterance}")
    return "\n\n".join(parts)


def build_scene_text(scene_system: str, scene_user: str) -> str:
    sys = (scene_system or "").strip()
    usr = (scene_user or "").strip()
    return sys + "\n\n" + usr


def build_topic_text(topic_en: str, utterance: str) -> str:
    topic_en = (topic_en or "").strip()
    utterance = (utterance or "").strip()
    parts = []
    if topic_en:
        parts.append(f"[TOPIC_EN] {topic_en}")
    if utterance:
        parts.append(f"[UTTERANCE] {utterance}")
    return "\n\n".join(parts)


# ----- strategy classifier (LoRA, lazy load) -----

@torch.inference_mode()
def run_strategy_with_logs(topic_en: str, utterance: str):
    global strategy_model

    logs = []
    topic_en = (topic_en or "").strip()
    utterance = (utterance or "").strip()

    if not topic_en and not utterance:
        logs.append("[strategy] empty topic & utterance, fallback to comforting")
        return config.STRATEGY_LABELS[0], [1.0] + [0.0] * (len(config.STRATEGY_LABELS) - 1), "\n".join(logs)

    if strategy_model is None:
        logs.append("[strategy] lazy loading model...")
        print("Loading strategy model LoRA...")
        base = AutoModelForSequenceClassification.from_pretrained(
            config.BASE_MODEL,
            num_labels=len(config.STRATEGY_LABELS),
            torch_dtype=torch.float16 if config.STRATEGY_DEVICE.type == "cuda" else torch.float32,
        )
        base.config.pad_token_id = tokenizer.pad_token_id

        _model = PeftModel.from_pretrained(
            base,
            config.LORA_REPO,
            adapter_name="strategy",
            subfolder=config.STRATEGY_SUBFOLDER,
        )
        _model.to(config.STRATEGY_DEVICE)
        _model.eval()
        strategy_model = _model
        print("[strategy] model ready on:", config.STRATEGY_DEVICE)

    text = f"[TOPIC_EN] {topic_en}\n\n[UTTERANCE] {utterance}"
    logs.append(f"[strategy] tokenize, len={len(text)}")

    enc = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.MAX_LENGTH,
    )
    enc = {k: v.to(config.STRATEGY_DEVICE) for k, v in enc.items()}

    t0 = time.time()
    logs.append("[strategy] forward start")
    out = strategy_model(**enc)
    t1 = time.time()

    logits = out.logits[0]
    probs = torch.softmax(logits, dim=-1)
    best_idx = int(torch.argmax(probs).item())
    best_label = config.STRATEGY_LABELS[best_idx]

    logs.append(f"[strategy] forward done, time={t1 - t0:.3f}s")
    logs.append(f"[strategy] logits={logits.tolist()}")
    logs.append(f"[strategy] probs={probs.tolist()}")
    logs.append(f"[strategy] best={best_label} (idx={best_idx})")

    return best_label, probs.tolist(), "\n".join(logs)


# ----- LLM utterance generation -----

def generate_agent_utterance(
    strategy: str,
    topic_en: str,
    utterance: str,
    profile_json: str,
    scene_system: str
) -> str:
    profile_short = (profile_json or "").strip()
    if len(profile_short) > 800:
        profile_short = profile_short[:800] + "...(truncated)"

    scene_short = (scene_system or "").strip()
    if len(scene_short) > 400:
        scene_short = scene_short[:400] + "...(truncated)"

    system_msg = (
        "You are a concise late-night conversational partner. "
        "Based on the given strategy (comforting, rational_analysis, or risk_alert), "
        "you respond in English with exactly ONE short sentence. "
        "The sentence must be declarative and must NOT contain any questions or question marks. "
        "Do not start new topics and do not ask for more information."
    )

    user_msg = f"""Context scene:
{scene_short}

Character profile (JSON-like text):
{profile_short}

Current dialogue topic (topic_en): {topic_en}
User's latest utterance: {utterance}

Chosen strategy: {strategy}

Please output ONE single English sentence that:
- follows the given strategy (comforting / rational_analysis / risk_alert)
- fits a quiet late-night bedroom conversation, gentle and not intrusive
- does NOT ask any questions
- does NOT use a question mark '?'
- does NOT start a new topic
- is self-contained and natural
Only output the sentence itself, with no extra explanation.
"""

    try:
        resp = config.client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=80,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("[agent_generation] error, using fallback:", repr(e))
        if strategy == "comforting":
            return "It sounds like you have been carrying a lot lately, so giving yourself a bit of rest could be very important."
        elif strategy == "rational_analysis":
            return "It may help to pause for a moment, clarify your main goal in this situation, and then decide whether staying up is really necessary."
        else:
            return "Continuing with this level of stress and sleep loss can seriously harm your health, so protecting yourself now is a responsible choice."

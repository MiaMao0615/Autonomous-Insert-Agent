# -*- coding: utf-8 -*-


import json
import time
import asyncio
import threading
from queue import Queue, Empty

import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import websockets


from openai import OpenAI




# ================== OpenAI 客户端 ==================

client = OpenAI()  # 会读取环境变量 OPENAI_API_KEY / OPENAI_BASE_URL 等


# ================== 路径配置 ==================

BASE_MODEL = r"D:\LLM\Qwen2.5-7B-Instruct"

PERSONA_LORA = r"D:\Task_design\personality\FinTune\outputs\qwen7b-lora-persona-will_full\checkpoint-32899"
SCENE_LORA   = r"D:\Task_design\Scene\outputs\qwen7b-lora-will_half_fp16_v2\checkpoint-35821"
TOPIC_LORA   = r"D:\Task_design\Topic\willingness_train\outputs\qwen7b-lora-topic_willingness\checkpoint-2500"

# strategy 3 分类 LoRA
STRATEGY_LORA = r"D:\Task_design\Topic\strategy_train\outputs\qwen7b-lora-topic_strategy\checkpoint-26250"
STRATEGY_LABELS = ["comforting", "rational_analysis", "risk_alert"]

MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== WebSocket 队列和连接管理 ==================

ws_queue: "Queue[dict]" = Queue()
ws_loop = None          # WebSocket 事件循环
ws_connections = set()  # 当前所有已连接的 WebSocket 客户端（Unity）
reply_ws_loop = None
reply_ws_connections = set()
last_sent_final = None  # 上一次发送到 Unity 的 final 值，用于变化检测

import asyncio
import json
import websockets
import threading


# ================== WebSocket 处理函数 ==================

async def ws_handler(websocket):
    """处理 Unity 连接的协程，负责接收消息并写入队列"""
    global ws_connections
    peer = getattr(websocket, "remote_address", None)
    print("Unity 已连接:", peer)

    ws_connections.add(websocket)

    try:
        async for message in websocket:
            print("[WS] 收到原始消息:", message)
            try:
                data = json.loads(message)
                print("[WS] 解析后的 JSON:", json.dumps(data, ensure_ascii=False, indent=2))
                ws_queue.put(data)  # 投递到 Tkinter 主线程
            except json.JSONDecodeError:
                print("[WS] 收到非 JSON，忽略")

    except Exception as e:
        print("[WS] WebSocket 异常:", repr(e))

    finally:
        ws_connections.remove(websocket)
        print("Unity 已断开:", peer)


async def ws_main():
    """启动 WebSocket 服务器（运行在独立线程）"""
    global ws_loop
    ws_loop = asyncio.get_running_loop()
    # 在握手阶段记录请求头，便于诊断非标准握手（兼容不同 websockets 版本的 request 类型）
    def _log_request(path, request):
        try:
            print("[WS] incoming handshake on 8765, path=", path)

            # request 可能是 mapping-like（支持 items），也可能是 websockets.http.Request
            headers = None
            if hasattr(request, "items"):
                try:
                    headers = dict(request.items())
                except Exception:
                    headers = None

            if headers is None and hasattr(request, "headers"):
                try:
                    # request.headers 可能是 mapping-like 或带 items()
                    headers = dict(request.headers.items())
                except Exception:
                    try:
                        headers = dict(request.headers)
                    except Exception:
                        headers = None

            if headers is None and hasattr(request, "raw_headers"):
                try:
                    headers = {k.decode(): v.decode() for k, v in request.raw_headers}
                except Exception:
                    headers = None

            if headers is None:
                # 无法解析 headers，就把 request 对象本身打印出来供诊断
                print("[WS] request object (unhandled type):", type(request), request)
            else:
                for k, v in headers.items():
                    print(f"[WS] header: {k}: {v}")

        except Exception as e:
            print("[WS] log_request error:", repr(e))
        # 返回 None 让默认的握手逻辑继续执行
        return None

    server = await websockets.serve(ws_handler, "0.0.0.0", 8765, process_request=_log_request)
    print("WebSocket 服务器已启动 ws://127.0.0.1:8765")
    await server.wait_closed()


def start_ws_server_in_thread():
    """以独立线程启动 WebSocket 事件循环，不阻塞 Tkinter"""
    def runner():
        asyncio.run(ws_main())
    t = threading.Thread(target=runner, daemon=True)
    t.start()



# ================== 1. 加载 tokenizer & base & LoRA ==================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base regression model...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=1,
    torch_dtype=torch.float16,
)
base_model.config.pad_token_id = tokenizer.pad_token_id

print("Loading persona LoRA...")
reg_model = PeftModel.from_pretrained(
    base_model,
    PERSONA_LORA,
    adapter_name="persona",
)

print("Loading scene LoRA...")
reg_model.load_adapter(
    SCENE_LORA,
    adapter_name="scene",
)

print("Loading topic LoRA...")
reg_model.load_adapter(
    TOPIC_LORA,
    adapter_name="topic",
)

reg_model.to(DEVICE)
reg_model.eval()

print("Regression model with 3 LoRA heads loaded on:", DEVICE)

# ---- 初始化 strategy 模型（不会一开始加载） ----

strategy_model = None  # 初始时不加载 strategy 模型
STRATEGY_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 在 GPU 上运行


# ================== 2. 通用 + 带日志的回归推理函数 ==================

@torch.inference_mode()
def run_willingness_with_logs(adapter_name: str, text: str):
    """
    返回 (数值, 日志字符串)
    """
    logs = []
    text = (text or "").strip()
    if not text:
        logs.append(f"[{adapter_name}] empty text, return 0.0")
        return 0.0, "\n".join(logs)

    logs.append(f"[{adapter_name}] set_adapter & tokenize, len(text)={len(text)}")
    reg_model.set_adapter(adapter_name)

    enc = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    t0 = time.time()
    logs.append(f"[{adapter_name}] forward start")
    logits = reg_model(**enc).logits.squeeze(-1).item()
    t1 = time.time()
    logs.append(f"[{adapter_name}] forward done, raw={logits}, time={t1 - t0:.3f}s")

    val = float(logits)
    val = max(0.0, min(1.0, val))  # clamp 到 [0,1]
    return val, "\n".join(logs)

def build_persona_text(persona_raw: str, profile_json: str, utterance: str) -> str:
    """
    生成完整的 persona 文本内容
    """
    persona_raw = (persona_raw or "").strip()
    utterance = (utterance or "").strip()

    profile = profile_json.strip() if profile_json else ""
    try:
        profile_obj = json.loads(profile)
        profile = json.dumps(profile_obj, ensure_ascii=False)
    except Exception:
        # 如果 profile 不是合法的 JSON 格式，则当普通文本处理
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
# ================== 3. Strategy 分类 & 插话生成 ==================

@torch.inference_mode()
def run_strategy_with_logs(topic_en: str, utterance: str):
    """
    使用 strategy LoRA 做 3 类分类：
      comforting / rational_analysis / risk_alert

    这里改成：
    - 第一次调用时才加载模型（懒加载）
    - 默认放在 GPU 上
    """
    global strategy_model

    logs = []
    topic_en = (topic_en or "").strip()
    utterance = (utterance or "").strip()

    if not topic_en and not utterance:
        logs.append("[strategy] empty topic & utterance, fallback to comforting")
        return STRATEGY_LABELS[0], [1.0] + [0.0] * (len(STRATEGY_LABELS) - 1), "\n".join(logs)

    # ===== 懒加载：只有真的需要时才加载模型 =====
    if strategy_model is None:
        logs.append("[strategy] strategy_model is None, start lazy loading...")
        print("Loading strategy model LoRA...")

        base = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=len(STRATEGY_LABELS),
            torch_dtype=torch.float16 if STRATEGY_DEVICE.type == "cuda" else torch.float32,
        )
        base.config.pad_token_id = tokenizer.pad_token_id

        _model = PeftModel.from_pretrained(
            base,
            STRATEGY_LORA,
            adapter_name="strategy",
        )
        _model.to(STRATEGY_DEVICE)  # 放到 GPU 上
        _model.eval()

        strategy_model = _model
        print("[strategy] Strategy model lazy-loaded on GPU.")

    # ===== 正常推理 =====
    text = f"[TOPIC_EN] {topic_en}\n\n[UTTERANCE] {utterance}"
    logs.append(f"[strategy] tokenize, len(text)={len(text)}")

    enc = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    enc = {k: v.to(STRATEGY_DEVICE) for k, v in enc.items()}

    t0 = time.time()
    logs.append("[strategy] forward start")
    out = strategy_model(**enc)
    t1 = time.time()

    logits = out.logits[0]  # shape [num_labels]
    probs = torch.softmax(logits, dim=-1)
    best_idx = int(torch.argmax(probs).item())
    best_label = STRATEGY_LABELS[best_idx]

    logs.append(f"[strategy] forward done, time={t1 - t0:.3f}s")
    logs.append(f"[strategy] logits={logits.tolist()}")
    logs.append(f"[strategy] probs={probs.tolist()}")
    logs.append(f"[strategy] best={best_label} (idx={best_idx})")

    return best_label, probs.tolist(), "\n".join(logs)


def generate_agent_utterance(strategy: str,
                             topic_en: str,
                             utterance: str,
                             profile_json: str,
                             scene_system: str) -> str:
    """
    Call the LLM to generate ONE short English utterance based on the strategy.
    Requirements:
    - Output in English
    - Exactly one sentence
    - Do NOT ask any questions
    - Do NOT use question marks (?)
    - Do NOT introduce new topics
    """
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
        resp = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # 按你实际可用模型改
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=80,
        )
        text = resp.choices[0].message.content.strip()

        return text
    except Exception as e:
        print("[agent_generation] 调用 LLM 失败，使用 fallback 文本:", repr(e))
        # fallback 也改成英文且无问号
        if strategy == "comforting":
            return "It sounds like you have been carrying a lot lately, so giving yourself a bit of rest could be very important."
        elif strategy == "rational_analysis":
            return "It may help to pause for a moment, clarify your main goal in this situation, and then decide whether staying up is really necessary."
        else:  # risk_alert
            return "Continuing with this level of stress and sleep loss can seriously harm your health, so protecting yourself now is a responsible choice."



# ================== 4. Tkinter GUI ==================

def main():
    root = tk.Tk()
    root.title("Willingness + Strategy Demo (Persona / Scene / Topic) - Tkinter")

    # 左侧：输入区
    left = tk.Frame(root)
    left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ==== 顶部：统一 utterance（给 persona & topic 共用）====
    tk.Label(left, text="Shared Utterance (Persona & Topic 共用):").pack(anchor="w")
    shared_utt_text = tk.Text(left, height=3)
    shared_utt_text.pack(fill=tk.X)

    # persona
    tk.Label(left, text="Persona Raw:").pack(anchor="w")
    persona_raw_text = tk.Text(left, height=3)
    persona_raw_text.pack(fill=tk.X)

    tk.Label(left, text="Profile (JSON or text):").pack(anchor="w")
    profile_text = tk.Text(left, height=3)
    profile_text.pack(fill=tk.X)

    # scene
    tk.Label(left, text="Scene System Prompt / 描述:").pack(anchor="w")
    scene_system_text = tk.Text(left, height=3)
    scene_system_text.pack(fill=tk.X)

    tk.Label(left, text="Scene User Message / 场景细节:").pack(anchor="w")
    scene_user_text = tk.Text(left, height=3)
    scene_user_text.pack(fill=tk.X)

    # topic
    tk.Label(left, text="Topic EN (英文主题):").pack(anchor="w")
    topic_en_text = tk.Text(left, height=3)
    topic_en_text.pack(fill=tk.X)

    # 中间按钮
    run_button = tk.Button(left, text="Run all 3 willingness")
    run_button.pack(pady=10)

    # 右侧：输出 + 权重 + 日志
    right = tk.Frame(root)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    # 数值变量
    persona_var = tk.StringVar(value="N/A")
    scene_var = tk.StringVar(value="N/A")
    topic_var = tk.StringVar(value="N/A")
    final_var = tk.StringVar(value="N/A")
    strategy_var = tk.StringVar(value="N/A")

    # 权重变量（默认都是 1.0）
    persona_w_var = tk.StringVar(value="1.0")
    scene_w_var   = tk.StringVar(value="1.0")
    topic_w_var   = tk.StringVar(value="1.0")

    # Persona
    tk.Label(right, text="Persona willingness (0-1):").pack(anchor="w")
    tk.Label(right, textvariable=persona_var, fg="blue").pack(anchor="w")
    tk.Label(right, text="Persona weight:").pack(anchor="w")
    tk.Entry(right, textvariable=persona_w_var, width=8).pack(anchor="w")

    # Scene
    tk.Label(right, text="Scene willingness (0-1):").pack(anchor="w")
    tk.Label(right, textvariable=scene_var, fg="blue").pack(anchor="w")
    tk.Label(right, text="Scene weight:").pack(anchor="w")
    tk.Entry(right, textvariable=scene_w_var, width=8).pack(anchor="w")

    # Topic
    tk.Label(right, text="Topic willingness (0-1):").pack(anchor="w")
    tk.Label(right, textvariable=topic_var, fg="blue").pack(anchor="w")
    tk.Label(right, text="Topic weight:").pack(anchor="w")
    tk.Entry(right, textvariable=topic_w_var, width=8).pack(anchor="w")

    # Final weighted score
    tk.Label(right, text="Final weighted willingness (0-1):").pack(anchor="w", pady=(10, 0))
    tk.Label(right, textvariable=final_var, fg="purple").pack(anchor="w")

    # Strategy 显示
    tk.Label(right, text="Chosen Strategy:").pack(anchor="w", pady=(10, 0))
    tk.Label(right, textvariable=strategy_var, fg="green").pack(anchor="w")

    # 日志框
    tk.Label(right, text="Debug Logs:").pack(anchor="w", pady=(10, 0))
    log_box = ScrolledText(right, height=25)
    log_box.pack(fill=tk.BOTH, expand=True)


    

    def append_log(text: str):
        log_box.insert(tk.END, text + "\n")
        log_box.see(tk.END)
        print(text)

    def safe_float(s: str, default: float = 1.0) -> float:
        try:
            v = float(s)
            if v < 0:
                return default
            return v
        except Exception:
            return default

    def on_run_clicked():
        global last_sent_final
        # 用于标记本次按下 Run 是否已经发送过带文本的 direct payload
        sent_direct_this_call = False
        # 读取输入
        shared_utt = shared_utt_text.get("1.0", tk.END).strip()

        persona_raw = persona_raw_text.get("1.0", tk.END).strip()
        profile_json = profile_text.get("1.0", tk.END).strip()

        scene_system = scene_system_text.get("1.0", tk.END).strip()
        scene_user = scene_user_text.get("1.0", tk.END).strip()

        topic_en = topic_en_text.get("1.0", tk.END).strip()

        # 读取权重
        w_p = safe_float(persona_w_var.get(), 1.0)
        w_s = safe_float(scene_w_var.get(), 1.0)
        w_t = safe_float(topic_w_var.get(), 1.0)

        logs = []
        logs.append("===== predict_all CALLED =====")
        logs.append(f"shared_utt={repr(shared_utt)}")
        logs.append(f"persona_raw={repr(persona_raw)}")
        logs.append(f"profile_json={repr(profile_json)}")
        logs.append(f"scene_system={repr(scene_system)}")
        logs.append(f"scene_user={repr(scene_user)}")
        logs.append(f"topic_en={repr(topic_en)}")
        logs.append(f"weights: persona={w_p}, scene={w_s}, topic={w_t}")
        logs.append("==============================")

        # persona：用 shared_utt 作为 utterance
        persona_text = build_persona_text(persona_raw, profile_json, shared_utt)
        p_val, log_p = run_willingness_with_logs("persona", persona_text)
        logs.append(log_p)

        # scene：只用 scene_system
        scene_text = build_scene_text(scene_system, scene_user)
        s_val, log_s = run_willingness_with_logs("scene", scene_text)
        logs.append(log_s)

        # topic：也用 shared_utt
        topic_text = build_topic_text(topic_en, shared_utt)
        t_val, log_t = run_willingness_with_logs("topic", topic_text)
        logs.append(log_t)

        # 计算加权结果
        sum_w = w_p + w_s + w_t
        if sum_w > 0:
            final = (w_p * p_val + w_s * s_val + w_t * t_val) / sum_w
        else:
            final = 0.0
        logs.append(f"[predict_all] DONE: persona={p_val}, scene={s_val}, topic={t_val}, final={final}")

        # 更新 UI 数值
        persona_var.set(f"{p_val:.3f}")
        scene_var.set(f"{s_val:.3f}")
        topic_var.set(f"{t_val:.3f}")
        final_var.set(f"{final:.3f}")

        # ====== 7.1 根据 final 决定要不要插话 ======
        THRESHOLD = 0.55  # 大于 0.55 开始插话

        strategy_var.set("N/A")

        if final > THRESHOLD:
            logs.append(f"[strategy_trigger] final={final:.3f} > {THRESHOLD}, 直接生成插话")

            # 直接调用 LLM 生成一句插话
            agent_text = generate_agent_utterance(
                strategy="direct",
                topic_en=topic_en,
                utterance=shared_utt,
                profile_json=profile_json,
                scene_system=scene_system,
            )
            logs.append(f"[agent_generation] generated utterance: {agent_text}")

            # 通过 WebSocket 发回 Unity
            send_payload = {
                "type": "agent_utterance",
                "strategy": "direct",
                "text": agent_text,
                "final_willingness": final,
                "topic_en": topic_en,
            }
            
            logs.append(f"send_payload: {send_payload}")
            # 调试信息：在调用 send_reply_to_unity 之前打印并写入日志，便于确认调用流
            append_log("[DEBUG] about to call send_reply_to_unity")
            print("[DEBUG] about to call send_reply_to_unity")
            send_reply_to_unity(send_payload)
            # 既然我们刚刚发送了带文本的直接插话，则视为已把当前 final 发送给 Unity，
            # 避免随后再立刻发送一条空 text 的 update 导致 Unity 收到重复信息
            try:
                last_sent_final = float(final)
            except Exception:
                last_sent_final = final
            sent_direct_this_call = True
        else:
            # 即便没有生成文本，也在 send 被按下且 final 与上次发送的值不同的时候，向 Unity 发送更新
            pass
        # ====== 7.2 若 final 值与上次发送不同，则发送一次仅包含 final 的更新（在按 send 后） ======
        try:
            # 如果本次已经发送了 direct payload，则不要再发送 update（避免重复）
            if not sent_direct_this_call:
                if last_sent_final is None or abs(final - last_sent_final) > 1e-3:
                    update_payload = {
                        "type": "agent_utterance",
                        "strategy": "update",
                        "text": "",
                        "final_willingness": final,
                        "topic_en": topic_en,
                    }
                    logs.append(f"send_payload(update): {update_payload}")
                    append_log("[DEBUG] about to call send_reply_to_unity for update")
                    print("[DEBUG] about to call send_reply_to_unity for update")
                    send_reply_to_unity(update_payload)
                    last_sent_final = float(final)
        except Exception as e:
            logs.append(f"[DEBUG] failed to send update payload: {e!r}")
            print("[DEBUG] failed to send update payload:", repr(e))
        # 写日志
        full_logs = "\n".join(logs)
        append_log(full_logs)

    run_button.config(command=on_run_clicked)

    # 调试按钮：手动触发 send_reply_to_unity，便于排查是否函数可达
    def test_send():
        payload = {
            "type": "agent_utterance",
            "strategy": "debug",
            "text": "这是一个调试发送，用于验证 send_reply_to_unity 是否被调用。",
            "final_willingness": 1.0,
            "topic_en": "debug_topic",
        }
        append_log("[TEST] manual test_send triggered")
        print("[TEST] manual test_send calling send_reply_to_unity")
        try:
            send_reply_to_unity(payload)
        except Exception as e:
            append_log(f"[TEST] send_reply_to_unity raised: {e!r}")
            print("[TEST] send_reply_to_unity raised:", repr(e))

    test_button = tk.Button(left, text="Test Send to Unity", command=test_send)
    test_button.pack(pady=4)


    async def reply_ws_handler(websocket):
        global reply_ws_connections
        peer = getattr(websocket, "remote_address", None)
        print("[ReplyWS] Unity 回传连接成功:", peer)

        reply_ws_connections.add(websocket)

        try:
            async for msg in websocket:
                print("[ReplyWS] 收到 Unity 消息（一般不会用到）:", msg)
        except Exception as e:
            print("[ReplyWS] 回传连接异常:", e)
        finally:
            reply_ws_connections.remove(websocket)
            print("[ReplyWS] Unity 回传断开:", peer)

    async def reply_ws_main():
        global reply_ws_loop
        reply_ws_loop = asyncio.get_running_loop()

        # 为回传服务器同样记录握手头，便于诊断 Unity 对 8766 的连接问题
        def _log_request_reply(path, request):
            try:
                print("[ReplyWS] incoming handshake on 8766, path=", path)
                # 重用上面的解析逻辑
                headers = None
                if hasattr(request, "items"):
                    try:
                        headers = dict(request.items())
                    except Exception:
                        headers = None
                if headers is None and hasattr(request, "headers"):
                    try:
                        headers = dict(request.headers.items())
                    except Exception:
                        try:
                            headers = dict(request.headers)
                        except Exception:
                            headers = None
                if headers is None and hasattr(request, "raw_headers"):
                    try:
                        headers = {k.decode(): v.decode() for k, v in request.raw_headers}
                    except Exception:
                        headers = None
                if headers is None:
                    print("[ReplyWS] request object (unhandled type):", type(request), request)
                else:
                    for k, v in headers.items():
                        print(f"[ReplyWS] header: {k}: {v}")
            except Exception as e:
                print("[ReplyWS] log_request error:", repr(e))
            return None

        server = await websockets.serve(reply_ws_handler, "0.0.0.0", 8766, process_request=_log_request_reply)
        print("[ReplyWS] 回传服务器已启动 ws://127.0.0.1:8766")
        await server.wait_closed()

    def start_reply_server():
        def runner():
            asyncio.run(reply_ws_main())
        t = threading.Thread(target=runner, daemon=True)
        t.start()

    def send_reply_to_unity(payload: dict):
        print("[ReplyWS] 准备发送给 Unity:", payload)
        # 诊断信息：打印当前回传连接集合及其 remote_address
        try:
            print(f"[ReplyWS] reply_ws_connections count={len(reply_ws_connections)}")
            for i, ws in enumerate(list(reply_ws_connections)):
                peer = getattr(ws, "remote_address", None)
                print(f"[ReplyWS] connection[{i}] peer={peer}, ws_obj={ws}")
        except Exception as e:
            print("[ReplyWS] error when listing reply_ws_connections:", repr(e))

        # 如果当前没有回传连接，等待短时间尝试让 Unity 建立连接（最多等待 2 秒），期间不会阻塞太久
        wait_start = time.time()
        while not reply_ws_connections and (time.time() - wait_start) < 2.0:
            time.sleep(0.05)

        if not reply_ws_connections:
            print("[ReplyWS] 没有 Unity 回传连接，无法发送")
            return

        if reply_ws_loop is None:
            print("[ReplyWS] 事件循环未初始化")
            return

        message = json.dumps(payload, ensure_ascii=False)

        async def _send(ws, msg):
            try:
                await ws.send(msg)
                print("[ReplyWS] 已发送:", msg)
            except Exception as e:
                print("[ReplyWS] 发送失败:", e)

        for ws in list(reply_ws_connections):
            asyncio.run_coroutine_threadsafe(_send(ws, message), reply_ws_loop)
    # ========= WebSocket → Tkinter 映射 =========





    
    def apply_update_from_unity(data: dict):
        dtype = data.get("type", "")

        if dtype == "persona_profile":
            background = data.get("background", "")
            traits = data.get("personality_traits", [])
            speaking_style = data.get("speaking_style", "")
            values = data.get("values", "")
            utt = data.get("utterance", "") or ""

            profile_obj = {
                "background": background,
                "personality_traits": traits,
                "speaking_style": speaking_style,
                "values": values,
            }
            profile_str = json.dumps(profile_obj, ensure_ascii=False, indent=2)

            profile_text.delete("1.0", tk.END)
            profile_text.insert(tk.END, profile_str)

            # Persona Raw 这里留空（不参与模型输入）
            persona_raw_text.delete("1.0", tk.END)
            persona_raw_text.insert(tk.END, "")

            # Shared utterance
            shared_utt_text.delete("1.0", tk.END)
            shared_utt_text.insert(tk.END, utt)

            append_log("[WS] 已更新 persona_profile 内容。")

            # 只要有 utterance，就自动跑一轮
            if utt.strip():
                append_log("[WS] 检测到非空 utterance，自动开始推理。")
                on_run_clicked()
            else:
                append_log("[WS] persona_profile 没有 utterance，不触发推理。")

        elif dtype == "topic":
            topic = data.get("topic", "")
            topic_en_text.delete("1.0", tk.END)
            topic_en_text.insert(tk.END, topic)
            append_log(f"[WS] 已更新 topic: {topic!r}")

        elif dtype == "scene_prompt":
            prompt = data.get("prompt", "")
            scene_system_text.delete("1.0", tk.END)
            scene_system_text.insert(tk.END, prompt)
            scene_user_text.delete("1.0", tk.END)
            scene_user_text.insert(tk.END, "")
            append_log("[WS] 已更新 scene_prompt 到 Scene System。")

        else:
            append_log(f"[WS] 收到未知类型数据: {dtype}, data={data}")

    

    def poll_ws_queue():
        try:
            while True:
                data = ws_queue.get_nowait()
                apply_update_from_unity(data)
        except Empty:
            pass

        root.after(50, poll_ws_queue)

    # 启动 WebSocket 服务器线程 + 队列轮询
    # 先启动回传服务器（8766），确保回传通道就绪，之后再启动主接收服务器（8765）
    start_reply_server()
    append_log("[ReplyWS] 回传服务器线程已启动 (端口 8766)。")
    start_ws_server_in_thread()
    append_log("[WS] WebSocket 服务器线程已启动 (端口 8765)。")

    poll_ws_queue()

    root.mainloop()


if __name__ == "__main__":
    main()

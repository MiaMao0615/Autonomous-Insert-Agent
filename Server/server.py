# server.py

import json
import time
import asyncio
import threading
from queue import Queue, Empty

import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import websockets

from inference import (
    build_persona_text,
    build_scene_text,
    build_topic_text,
    run_willingness_with_logs,
    run_strategy_with_logs,
    generate_agent_utterance,
)

# queues and connection sets
ws_queue: "Queue[dict]" = Queue()
ws_loop = None
ws_connections = set()

reply_ws_loop = None
reply_ws_connections = set()
last_sent_final = None


# ---------- WebSocket server (recv from Unity, 8765) ----------

async def ws_handler(websocket):
    global ws_connections
    peer = getattr(websocket, "remote_address", None)
    print("Unity connected:", peer)

    ws_connections.add(websocket)
    try:
        async for message in websocket:
            print("[WS] raw:", message)
            try:
                data = json.loads(message)
                print("[WS] parsed:", json.dumps(data, ensure_ascii=False, indent=2))
                ws_queue.put(data)
            except json.JSONDecodeError:
                print("[WS] non-JSON message, ignored")
    except Exception as e:
        print("[WS] error:", repr(e))
    finally:
        ws_connections.remove(websocket)
        print("Unity disconnected:", peer)


async def ws_main():
    global ws_loop
    ws_loop = asyncio.get_running_loop()

    def _log_request(path, request):
        try:
            print("[WS] handshake on 8765, path:", path)
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
                print("[WS] request object:", type(request), request)
            else:
                for k, v in headers.items():
                    print(f"[WS] header: {k}: {v}")
        except Exception as e:
            print("[WS] log_request error:", repr(e))
        return None

    server = await websockets.serve(ws_handler, "0.0.0.0", 8765, process_request=_log_request)
    print("WebSocket server started: ws://127.0.0.1:8765")
    await server.wait_closed()


def start_ws_server_in_thread():
    def runner():
        asyncio.run(ws_main())
    t = threading.Thread(target=runner, daemon=True)
    t.start()


# ---------- reply WebSocket (send to Unity, 8766) ----------

async def reply_ws_handler(websocket):
    global reply_ws_connections
    peer = getattr(websocket, "remote_address", None)
    print("[ReplyWS] Unity connected:", peer)
    reply_ws_connections.add(websocket)
    try:
        async for msg in websocket:
            print("[ReplyWS] msg from Unity:", msg)
    except Exception as e:
        print("[ReplyWS] error:", e)
    finally:
        reply_ws_connections.remove(websocket)
        print("[ReplyWS] Unity disconnected:", peer)


async def reply_ws_main():
    global reply_ws_loop
    reply_ws_loop = asyncio.get_running_loop()

    def _log_request_reply(path, request):
        try:
            print("[ReplyWS] handshake on 8766, path:", path)
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
                print("[ReplyWS] request object:", type(request), request)
            else:
                for k, v in headers.items():
                    print(f"[ReplyWS] header: {k}: {v}")
        except Exception as e:
            print("[ReplyWS] log_request error:", repr(e))
        return None

    server = await websockets.serve(reply_ws_handler, "0.0.0.0", 8766, process_request=_log_request_reply)
    print("[ReplyWS] server started: ws://127.0.0.1:8766")
    await server.wait_closed()


def start_reply_server():
    def runner():
        asyncio.run(reply_ws_main())
    t = threading.Thread(target=runner, daemon=True)
    t.start()


def send_reply_to_unity(payload: dict):
    print("[ReplyWS] send payload:", payload)
    try:
        print(f"[ReplyWS] connections: {len(reply_ws_connections)}")
        for i, ws in enumerate(list(reply_ws_connections)):
            peer = getattr(ws, "remote_address", None)
            print(f"[ReplyWS] conn[{i}] peer={peer}, ws={ws}")
    except Exception as e:
        print("[ReplyWS] listing connections failed:", repr(e))

    wait_start = time.time()
    while not reply_ws_connections and (time.time() - wait_start) < 2.0:
        time.sleep(0.05)

    if not reply_ws_connections:
        print("[ReplyWS] no Unity connection, abort")
        return

    if reply_ws_loop is None:
        print("[ReplyWS] event loop not ready")
        return

    msg = json.dumps(payload, ensure_ascii=False)

    async def _send(ws, m):
        try:
            await ws.send(m)
            print("[ReplyWS] sent:", m)
        except Exception as e:
            print("[ReplyWS] send error:", e)

    for ws in list(reply_ws_connections):
        asyncio.run_coroutine_threadsafe(_send(ws, msg), reply_ws_loop)


# ---------- main Tkinter app ----------

def main():
    global last_sent_final

    root = tk.Tk()
    root.title("Autonomous Insert Agent - Willingness & Strategy")

    # left pane: inputs
    left = tk.Frame(root)
    left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    tk.Label(left, text="Shared Utterance:").pack(anchor="w")
    shared_utt_text = tk.Text(left, height=3)
    shared_utt_text.pack(fill=tk.X)

    tk.Label(left, text="Persona Raw:").pack(anchor="w")
    persona_raw_text = tk.Text(left, height=3)
    persona_raw_text.pack(fill=tk.X)

    tk.Label(left, text="Profile (JSON or text):").pack(anchor="w")
    profile_text = tk.Text(left, height=3)
    profile_text.pack(fill=tk.X)

    tk.Label(left, text="Scene System Prompt:").pack(anchor="w")
    scene_system_text = tk.Text(left, height=3)
    scene_system_text.pack(fill=tk.X)

    tk.Label(left, text="Scene User Message:").pack(anchor="w")
    scene_user_text = tk.Text(left, height=3)
    scene_user_text.pack(fill=tk.X)

    tk.Label(left, text="Topic EN:").pack(anchor="w")
    topic_en_text = tk.Text(left, height=3)
    topic_en_text.pack(fill=tk.X)

    run_button = tk.Button(left, text="Run all 3 willingness")
    run_button.pack(pady=10)

    # right pane: outputs
    right = tk.Frame(root)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    persona_var = tk.StringVar(value="N/A")
    scene_var = tk.StringVar(value="N/A")
    topic_var = tk.StringVar(value="N/A")
    final_var = tk.StringVar(value="N/A")
    strategy_var = tk.StringVar(value="N/A")

    persona_w_var = tk.StringVar(value="1.0")
    scene_w_var = tk.StringVar(value="1.0")
    topic_w_var = tk.StringVar(value="1.0")

    tk.Label(right, text="Persona willingness (0-1):").pack(anchor="w")
    tk.Label(right, textvariable=persona_var, fg="blue").pack(anchor="w")
    tk.Label(right, text="Persona weight:").pack(anchor="w")
    tk.Entry(right, textvariable=persona_w_var, width=8).pack(anchor="w")

    tk.Label(right, text="Scene willingness (0-1):").pack(anchor="w")
    tk.Label(right, textvariable=scene_var, fg="blue").pack(anchor="w")
    tk.Label(right, text="Scene weight:").pack(anchor="w")
    tk.Entry(right, textvariable=scene_w_var, width=8).pack(anchor="w")

    tk.Label(right, text="Topic willingness (0-1):").pack(anchor="w")
    tk.Label(right, textvariable=topic_var, fg="blue").pack(anchor="w")
    tk.Label(right, text="Topic weight:").pack(anchor="w")
    tk.Entry(right, textvariable=topic_w_var, width=8).pack(anchor="w")

    tk.Label(right, text="Final weighted willingness (0-1):").pack(anchor="w", pady=(10, 0))
    tk.Label(right, textvariable=final_var, fg="purple").pack(anchor="w")

    tk.Label(right, text="Chosen Strategy:").pack(anchor="w", pady=(10, 0))
    tk.Label(right, textvariable=strategy_var, fg="green").pack(anchor="w")

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

    # ---- main inference button ----

    def on_run_clicked():
        nonlocal last_sent_final
        sent_direct_this_call = False

        shared_utt = shared_utt_text.get("1.0", tk.END).strip()
        persona_raw = persona_raw_text.get("1.0", tk.END).strip()
        profile_json = profile_text.get("1.0", tk.END).strip()
        scene_system = scene_system_text.get("1.0", tk.END).strip()
        scene_user = scene_user_text.get("1.0", tk.END).strip()
        topic_en = topic_en_text.get("1.0", tk.END).strip()

        w_p = safe_float(persona_w_var.get(), 1.0)
        w_s = safe_float(scene_w_var.get(), 1.0)
        w_t = safe_float(topic_w_var.get(), 1.0)

        logs = []
        logs.append("===== predict_all =====")
        logs.append(f"shared_utt={repr(shared_utt)}")
        logs.append(f"persona_raw={repr(persona_raw)}")
        logs.append(f"profile_json={repr(profile_json)}")
        logs.append(f"scene_system={repr(scene_system)}")
        logs.append(f"scene_user={repr(scene_user)}")
        logs.append(f"topic_en={repr(topic_en)}")
        logs.append(f"weights: persona={w_p}, scene={w_s}, topic={w_t}")

        persona_text = build_persona_text(persona_raw, profile_json, shared_utt)
        p_val, log_p = run_willingness_with_logs("persona", persona_text)
        logs.append(log_p)

        scene_text = build_scene_text(scene_system, scene_user)
        s_val, log_s = run_willingness_with_logs("scene", scene_text)
        logs.append(log_s)

        topic_text = build_topic_text(topic_en, shared_utt)
        t_val, log_t = run_willingness_with_logs("topic", topic_text)
        logs.append(log_t)

        sum_w = w_p + w_s + w_t
        final = (w_p * p_val + w_s * s_val + w_t * t_val) / sum_w if sum_w > 0 else 0.0

        logs.append(f"[predict_all] persona={p_val}, scene={s_val}, topic={t_val}, final={final}")

        persona_var.set(f"{p_val:.3f}")
        scene_var.set(f"{s_val:.3f}")
        topic_var.set(f"{t_val:.3f}")
        final_var.set(f"{final:.3f}")

        THRESHOLD = 0.55
        strategy_var.set("N/A")

        if final > THRESHOLD:
            logs.append(f"[strategy_trigger] final={final:.3f} > {THRESHOLD}")
            best_label, probs, log_strategy = run_strategy_with_logs(topic_en, shared_utt)
            logs.append(log_strategy)
            strategy_var.set(best_label)

            agent_text = generate_agent_utterance(
                strategy=best_label,
                topic_en=topic_en,
                utterance=shared_utt,
                profile_json=profile_json,
                scene_system=scene_system,
            )
            logs.append(f"[agent_generation] text={agent_text}")

            payload = {
                "type": "agent_utterance",
                "strategy": best_label,
                "text": agent_text,
                "final_willingness": final,
                "topic_en": topic_en,
            }
            logs.append(f"send_payload: {payload}")
            append_log("[DEBUG] send_reply_to_unity (direct)")
            send_reply_to_unity(payload)

            try:
                last_sent_final = float(final)
            except Exception:
                last_sent_final = final
            sent_direct_this_call = True

        if not sent_direct_this_call:
            try:
                if last_sent_final is None or abs(final - last_sent_final) > 1e-3:
                    payload = {
                        "type": "agent_utterance",
                        "strategy": "update",
                        "text": "",
                        "final_willingness": final,
                        "topic_en": topic_en,
                    }
                    logs.append(f"send_payload(update): {payload}")
                    append_log("[DEBUG] send_reply_to_unity (update)")
                    send_reply_to_unity(payload)
                    last_sent_final = float(final)
            except Exception as e:
                logs.append(f"[DEBUG] update send failed: {e!r}")

        append_log("\n".join(logs))

    run_button.config(command=on_run_clicked)

    # test button

    def test_send():
        payload = {
            "type": "agent_utterance",
            "strategy": "debug",
            "text": "This is a debug message from backend.",
            "final_willingness": 1.0,
            "topic_en": "debug_topic",
        }
        append_log("[TEST] test_send triggered")
        try:
            send_reply_to_unity(payload)
        except Exception as e:
            append_log(f"[TEST] send_reply_to_unity error: {e!r}")

    test_button = tk.Button(left, text="Test Send to Unity", command=test_send)
    test_button.pack(pady=4)

    # WS → Tkinter mapping

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

            persona_raw_text.delete("1.0", tk.END)
            persona_raw_text.insert(tk.END, "")

            shared_utt_text.delete("1.0", tk.END)
            shared_utt_text.insert(tk.END, utt)

            append_log("[WS] persona_profile updated")

            if utt.strip():
                append_log("[WS] non-empty utterance, auto run")
                on_run_clicked()
            else:
                append_log("[WS] empty utterance, no auto run")

        elif dtype == "topic":
            topic = data.get("topic", "")
            topic_en_text.delete("1.0", tk.END)
            topic_en_text.insert(tk.END, topic)
            append_log(f"[WS] topic updated: {topic!r}")

        elif dtype == "scene_prompt":
            prompt = data.get("prompt", "")
            scene_system_text.delete("1.0", tk.END)
            scene_system_text.insert(tk.END, prompt)
            scene_user_text.delete("1.0", tk.END)
            scene_user_text.insert(tk.END, "")
            append_log("[WS] scene_prompt updated")

        else:
            append_log(f"[WS] unknown type: {dtype}, data={data}")

    def poll_ws_queue():
        try:
            while True:
                data = ws_queue.get_nowait()
                apply_update_from_unity(data)
        except Empty:
            pass
        root.after(50, poll_ws_queue)

    start_reply_server()
    append_log("[ReplyWS] server thread started (8766)")
    start_ws_server_in_thread()
    append_log("[WS] server thread started (8765)")

    poll_ws_queue()
    root.mainloop()


if __name__ == "__main__":
    main()

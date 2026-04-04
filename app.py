"""
app.py — HuggingFace Spaces entry point.
All source code lives in src/ to avoid conflict with this app.py file.
"""

import sys
import os
import threading
import tempfile
import uuid
import time

sys.path.insert(0, os.path.dirname(__file__))

import uvicorn
import gradio as gr
import requests
import aiofiles
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.asr import WhisperASR
from src.llm import VoiceLLMChain
from src.tts import TTSEngine
from config import settings

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

asr = WhisperASR(model_size=settings.WHISPER_MODEL)
llm = VoiceLLMChain()
tts = TTSEngine(output_dir="temp_audio")
print("All models loaded.")

# ── FastAPI ───────────────────────────────────────────────────────────────────
api = FastAPI(title="Multilingual Voice Chatbot")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
api.mount("/audio", StaticFiles(directory="temp_audio"), name="audio")


@api.get("/health")
async def health():
    return {"status": "ok", "whisper_model": asr.model_size, "device": asr.device}


@api.post("/voice-loop")
async def voice_loop(file: UploadFile = File(...), session_id: str = "default"):
    suffix = Path(file.filename).suffix or ".wav"
    audio_path = TEMP_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
    async with aiofiles.open(audio_path, "wb") as f:
        await f.write(await file.read())
    try:
        asr_result = asr.transcribe(audio_path)
        reply_text = llm.chat(
            text=asr_result.text,
            language=asr_result.language,
            session_id=session_id,
        )
        reply_audio = tts.speak(text=reply_text, language=asr_result.language)
        return {
            "transcript": asr_result.text,
            "language": asr_result.language,
            "language_probability": asr_result.language_probability,
            "reply_text": reply_text,
            "audio_url": f"/audio/{reply_audio.name}",
            "session_id": session_id,
        }
    finally:
        if audio_path.exists():
            audio_path.unlink()


@api.post("/chat")
async def chat(body: dict):
    reply = llm.chat(
        text=body.get("text", ""),
        language=body.get("language", "en"),
        session_id=body.get("session_id", "default"),
    )
    return {"reply": reply, "session_id": body.get("session_id", "default")}


def run_api():
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="warning")

threading.Thread(target=run_api, daemon=True).start()
time.sleep(3)

# ── Gradio UI ─────────────────────────────────────────────────────────────────
LANGUAGE_FLAGS = {
    "ar": "Arabic", "en": "English", "fr": "French",
    "de": "German", "es": "Spanish", "it": "Italian",
    "zh": "Chinese", "ja": "Japanese", "pt": "Portuguese",
    "ru": "Russian", "tr": "Turkish", "nl": "Dutch",
}


def run_voice_loop(audio_input, session_id, chat_history):
    if audio_input is None:
        return chat_history, None, "Please record or upload audio first."
    try:
        with open(audio_input, "rb") as f:
            response = requests.post(
                "http://localhost:8000/voice-loop",
                files={"file": ("audio.wav", f, "audio/wav")},
                params={"session_id": session_id},
                timeout=120,
            )
        if response.status_code != 200:
            return chat_history, None, f"Error: {response.json().get('detail', 'Unknown')}"

        data = response.json()
        audio_resp = requests.get(f"http://localhost:8000{data['audio_url']}", timeout=15)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(audio_resp.content)
        tmp.close()

        lang = LANGUAGE_FLAGS.get(data["language"], data["language"].upper())
        conf = f"{data['language_probability'] * 100:.0f}%"
        chat_history.append((f"[{lang} · {conf}]\n{data['transcript']}", data["reply_text"]))
        return chat_history, tmp.name, f"Detected: {lang} ({conf})"
    except Exception as e:
        return chat_history, None, f"Error: {str(e)}"


def new_session():
    return [], None, "", str(uuid.uuid4())[:8]


with gr.Blocks(title="Multilingual Voice Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ Multilingual Voice Chatbot
    Speak in any language — the bot transcribes, replies, and speaks back.

    **Supported:** Arabic · English · French · German · Spanish · and 94 more
    """)
    session_id = gr.State(value=str(uuid.uuid4())[:8])
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Your voice")
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or upload")
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=3)
                new_btn  = gr.Button("New chat", variant="secondary", scale=1)
            status_box   = gr.Textbox(label="Status", interactive=False)
            gr.Markdown("### Bot reply")
            audio_output = gr.Audio(label="Play reply", type="filepath", autoplay=True)
        with gr.Column(scale=1):
            gr.Markdown("### Conversation")
            chatbot = gr.Chatbot(label="Chat", height=480)

    send_btn.click(run_voice_loop, [audio_input, session_id, chatbot], [chatbot, audio_output, status_box])
    new_btn.click(new_session, [], [chatbot, audio_output, status_box, session_id])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
"""
app.py — HuggingFace Spaces entry point.

Runs FastAPI and Gradio together in one process.
The conflict between app.py and app/ folder is solved by
importing modules directly instead of through the app package.
"""

import sys
import os

# Add the project root to path so imports work correctly
sys.path.insert(0, os.path.dirname(__file__))

import threading
import tempfile
import uuid

import uvicorn
import gradio as gr
import requests
from pathlib import Path

# Import directly — avoid the app.main naming conflict
from app.asr import WhisperASR
from app.llm import VoiceLLMChain
from app.tts import TTSEngine
from config import settings

# ── Load models once ──────────────────────────────────────────────────────────
print("Loading models...")
asr = WhisperASR(model_size=settings.WHISPER_MODEL)
llm = VoiceLLMChain()
tts = TTSEngine(output_dir="temp_audio")
print("All models loaded.")

TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

# ── Build FastAPI app with models pre-loaded ──────────────────────────────────
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import aiofiles

api = FastAPI(title="Multilingual Voice Chatbot")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        reply_audio_path = tts.speak(text=reply_text, language=asr_result.language)

        return {
            "transcript": asr_result.text,
            "language": asr_result.language,
            "language_probability": asr_result.language_probability,
            "reply_text": reply_text,
            "audio_url": f"/audio/{reply_audio_path.name}",
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


# ── Start FastAPI in background thread ────────────────────────────────────────

def run_api():
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="warning")

threading.Thread(target=run_api, daemon=True).start()

# Give the API a moment to start
import time
time.sleep(3)

# ── Gradio UI ─────────────────────────────────────────────────────────────────

LANGUAGE_FLAGS = {
    "ar": "Arabic",  "en": "English", "fr": "French",
    "de": "German",  "es": "Spanish", "it": "Italian",
    "zh": "Chinese", "ja": "Japanese","pt": "Portuguese",
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

        data       = response.json()
        transcript = data["transcript"]
        language   = data["language"]
        lang_prob  = data["language_probability"]
        reply_text = data["reply_text"]
        audio_url  = data["audio_url"]

        audio_resp = requests.get(f"http://localhost:8000{audio_url}", timeout=15)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(audio_resp.content)
        tmp.close()

        lang_label = LANGUAGE_FLAGS.get(language, language.upper())
        confidence = f"{lang_prob * 100:.0f}%"
        chat_history.append((
            f"[{lang_label} · {confidence}]\n{transcript}",
            reply_text
        ))
        return chat_history, tmp.name, f"Detected: {lang_label} ({confidence})"

    except Exception as e:
        return chat_history, None, f"Error: {str(e)}"


def new_session():
    return [], None, "", str(uuid.uuid4())[:8]


with gr.Blocks(title="Multilingual Voice Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Multilingual Voice Chatbot
    Speak in any language — the bot transcribes, replies, and speaks back.

    **Supported:** Arabic · English · French · German · Spanish · and 94 more
    """)

    session_id = gr.State(value=str(uuid.uuid4())[:8])

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Your voice")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or upload audio",
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=3)
                new_btn  = gr.Button("New chat", variant="secondary", scale=1)
            status_box   = gr.Textbox(label="Status", interactive=False)
            gr.Markdown("### Bot reply")
            audio_output = gr.Audio(
                label="Play reply",
                type="filepath",
                autoplay=True,
            )

        with gr.Column(scale=1):
            gr.Markdown("### Conversation")
            chatbot = gr.Chatbot(label="Chat", height=480)

    send_btn.click(
        fn=run_voice_loop,
        inputs=[audio_input, session_id, chatbot],
        outputs=[chatbot, audio_output, status_box],
    )
    new_btn.click(
        fn=new_session,
        inputs=[],
        outputs=[chatbot, audio_output, status_box, session_id],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
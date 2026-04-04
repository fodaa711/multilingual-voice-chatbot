"""
app.py — HuggingFace Spaces entry point.

Runs FastAPI and Gradio in the same process:
  - FastAPI handles the AI pipeline (ASR + LLM + TTS)
  - Gradio provides the web UI
  - Both share the same loaded models — no duplicate memory usage

HuggingFace Spaces expects a file called app.py at the root
and expects it to launch on port 7860.
"""

import threading
import uvicorn
import gradio as gr
import requests
import uuid
import tempfile
import os

from app.main import app as fastapi_app
from app.asr import WhisperASR
from app.llm import VoiceLLMChain
from app.tts import TTSEngine
from config import settings
from pathlib import Path

# ── Shared model instances ─────────────────────────────────────────────────────
# Load once and reuse in both FastAPI and Gradio
print("Loading models...")
asr = WhisperASR(model_size=settings.WHISPER_MODEL)
llm = VoiceLLMChain()
tts = TTSEngine(output_dir="temp_audio")

# Inject into FastAPI app state manually (since we bypass lifespan here)
fastapi_app.state.asr = asr
fastapi_app.state.llm = llm
fastapi_app.state.tts = tts
print("All models loaded.")

TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

LANGUAGE_FLAGS = {
    "ar": "Arabic",  "en": "English", "fr": "French",
    "de": "German",  "es": "Spanish", "it": "Italian",
    "zh": "Chinese", "ja": "Japanese","pt": "Portuguese",
    "ru": "Russian", "tr": "Turkish", "nl": "Dutch",
}


# ── Start FastAPI in background thread ────────────────────────────────────────

def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="warning")

threading.Thread(target=run_fastapi, daemon=True).start()


# ── Gradio functions ───────────────────────────────────────────────────────────

def run_voice_loop(audio_input, session_id, chat_history):
    if audio_input is None:
        return chat_history, None, "Please record or upload audio first."

    try:
        # Call the FastAPI endpoint running in the background
        with open(audio_input, "rb") as f:
            response = requests.post(
                "http://localhost:8000/voice-loop",
                files={"file": ("audio.wav", f, "audio/wav")},
                params={"session_id": session_id},
                timeout=120,
            )

        if response.status_code != 200:
            error = response.json().get("detail", "Unknown error")
            return chat_history, None, f"Error: {error}"

        data        = response.json()
        transcript  = data["transcript"]
        language    = data["language"]
        lang_prob   = data["language_probability"]
        reply_text  = data["reply_text"]
        audio_url   = data["audio_url"]

        # Download the reply audio
        audio_response = requests.get(
            f"http://localhost:8000{audio_url}", timeout=15
        )
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp3", prefix="reply_"
        )
        tmp.write(audio_response.content)
        tmp.close()

        lang_label  = LANGUAGE_FLAGS.get(language, language.upper())
        confidence  = f"{lang_prob * 100:.0f}%"
        user_msg    = f"[{lang_label} · {confidence}]\n{transcript}"
        bot_msg     = reply_text

        chat_history.append((user_msg, bot_msg))
        status = f"Detected: {lang_label} ({confidence} confidence)"
        return chat_history, tmp.name, status

    except Exception as e:
        return chat_history, None, f"Error: {str(e)}"


def new_session():
    return [], None, "", str(uuid.uuid4())[:8]


# ── Build Gradio UI ────────────────────────────────────────────────────────────

with gr.Blocks(title="Multilingual Voice Chatbot", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # Multilingual Voice Chatbot
    Speak in any language — the bot transcribes, replies, and speaks back in the same language.

    **Supported:** Arabic · English · French · German · Spanish · and 94 more languages
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

    gr.Markdown("### Sample files")
    gr.Examples(
        examples=[
            ["samples/ar_sample.wav"],
            ["samples/en_sample.wav"],
            ["samples/fr_sample.wav"],
        ],
        inputs=[audio_input],
    )

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


# ── Launch ─────────────────────────────────────────────────────────────────────
# HuggingFace Spaces requires port 7860
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
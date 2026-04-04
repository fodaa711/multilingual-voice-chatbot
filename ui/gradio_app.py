"""
ui/gradio_app.py — Gradio web interface for the Multilingual Voice Chatbot.

HOW THIS WORKS:
  This UI talks to your FastAPI server via HTTP requests.
  It does NOT import app/ directly — it calls your API endpoints
  exactly like a real user would, which means:
    - The FastAPI server must be running on localhost:8000
    - The Gradio app runs separately on localhost:7860
    - They communicate over HTTP

  This is the correct architecture — UI and API are separate.

WHAT THE UI PROVIDES:
  - Microphone recording OR file upload
  - One click to run the full voice loop
  - Shows transcript + detected language
  - Shows the LLM reply text
  - Plays the audio reply directly in the browser
  - Chat history so you can see the full conversation
  - Session management (new conversation button)
  - Supports all languages Whisper detects
"""

import uuid
import requests
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

LANGUAGE_FLAGS = {
    "ar": "🇸🇦 Arabic",
    "en": "🇬🇧 English",
    "fr": "🇫🇷 French",
    "de": "🇩🇪 German",
    "es": "🇪🇸 Spanish",
    "it": "🇮🇹 Italian",
    "zh": "🇨🇳 Chinese",
    "ja": "🇯🇵 Japanese",
    "pt": "🇧🇷 Portuguese",
    "ru": "🇷🇺 Russian",
    "tr": "🇹🇷 Turkish",
    "nl": "🇳🇱 Dutch",
}


# ── Core function — called when user clicks Send ──────────────────────────────

def run_voice_loop(audio_input, session_id, chat_history):
    """
    Takes the recorded/uploaded audio, sends it to /voice-loop,
    and returns the updated chat history + audio reply.

    Args:
        audio_input:  path to the audio file (from Gradio mic or upload)
        session_id:   unique session string for conversation memory
        chat_history: list of (user_text, bot_text) tuples for display

    Returns:
        updated chat_history, audio reply path, status message
    """

    # Guard: no audio provided
    if audio_input is None:
        return chat_history, None, "⚠️ Please record or upload audio first."

    # Check the API server is reachable
    try:
        requests.get(f"{API_BASE}/health", timeout=3)
    except requests.exceptions.ConnectionError:
        return (
            chat_history,
            None,
            "❌ Cannot reach the API server at localhost:8000. "
            "Make sure uvicorn is running."
        )

    # Call POST /voice-loop
    try:
        with open(audio_input, "rb") as f:
            response = requests.post(
                f"{API_BASE}/voice-loop",
                files={"file": ("audio.wav", f, "audio/wav")},
                params={"session_id": session_id},
                timeout=60,    # whisper can take a moment on CPU
            )

        if response.status_code != 200:
            error = response.json().get("detail", "Unknown error")
            return chat_history, None, f"❌ API error: {error}"

        data = response.json()

        # ── Extract results ────────────────────────────────────────────────
        transcript   = data["transcript"]
        language     = data["language"]
        lang_prob    = data["language_probability"]
        reply_text   = data["reply_text"]
        audio_url    = data["audio_url"]

        # ── Download the reply audio from the API ──────────────────────────
        audio_response = requests.get(f"{API_BASE}{audio_url}", timeout=15)
        audio_bytes = audio_response.content

        # Save to a temp file Gradio can play
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp3", prefix="reply_"
        )
        tmp.write(audio_bytes)
        tmp.close()
        reply_audio_path = tmp.name

        # ── Build chat history entry ───────────────────────────────────────
        lang_label = LANGUAGE_FLAGS.get(language, f"🌐 {language.upper()}")
        confidence = f"{lang_prob * 100:.0f}%"

        user_message = f"🎤 [{lang_label} · {confidence}]\n{transcript}"
        bot_message  = f"🤖 {reply_text}"

        chat_history.append((user_message, bot_message))

        status = f"✅ Detected: {lang_label} ({confidence} confidence)"
        return chat_history, reply_audio_path, status

    except Exception as e:
        return chat_history, None, f"❌ Error: {str(e)}"


def new_session():
    """Generate a fresh session ID and clear the chat history."""
    return [], None, "", str(uuid.uuid4())[:8]


def check_server():
    """Check if the FastAPI server is running."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        data = r.json()
        return (
            f"✅ Server online — "
            f"Whisper: {data['whisper_model']} · "
            f"Device: {data['device']}"
        )
    except Exception:
        return "❌ Server offline — run: uvicorn app.main:app --reload"


# ── Build the Gradio UI ───────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="Multilingual Voice Chatbot",
        theme=gr.themes.Soft(),
    ) as demo:

        # ── Header ─────────────────────────────────────────────────────────
        gr.Markdown("""
        # 🎙️ Multilingual Voice Chatbot
        Speak in any language → AI transcribes → replies in the same language → speaks back.

        **Supported:** Arabic 🇸🇦 · English 🇬🇧 · French 🇫🇷 · German 🇩🇪 · Spanish 🇪🇸 · and 94 more
        """)

        # ── Server status ───────────────────────────────────────────────────
        with gr.Row():
            server_status = gr.Textbox(
                label="Server status",
                value=check_server(),
                interactive=False,
                scale=4,
            )
            refresh_btn = gr.Button("🔄 Refresh", scale=1)

        # ── Session ID (hidden) ─────────────────────────────────────────────
        session_id = gr.State(value=str(uuid.uuid4())[:8])

        # ── Main layout ─────────────────────────────────────────────────────
        with gr.Row():

            # Left column — input
            with gr.Column(scale=1):
                gr.Markdown("### Your voice")

                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or upload audio",
                )

                with gr.Row():
                    send_btn = gr.Button(
                        "🚀 Send",
                        variant="primary",
                        scale=3,
                    )
                    new_btn = gr.Button(
                        "🗑️ New chat",
                        variant="secondary",
                        scale=1,
                    )

                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1,
                )

                gr.Markdown("### Bot reply (audio)")
                audio_output = gr.Audio(
                    label="Play reply",
                    type="filepath",
                    autoplay=True,       # plays automatically when ready
                )

            # Right column — chat history
            with gr.Column(scale=1):
                gr.Markdown("### Conversation history")
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=480,
                    bubble_full_width=False,
                )

        # ── Example files ───────────────────────────────────────────────────
        gr.Markdown("### Try with sample files")
        gr.Examples(
            examples=[
                ["samples/ar_sample.wav"],
                ["samples/en_sample.wav"],
                ["samples/fr_sample.wav"],
            ],
            inputs=[audio_input],
            label="Click a sample to load it",
        )

        # ── Event handlers ──────────────────────────────────────────────────

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

        refresh_btn.click(
            fn=check_server,
            inputs=[],
            outputs=[server_status],
        )

    return demo


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,         # set True to get a public gradio.live URL
        inbrowser=True,      # opens browser automatically
    )
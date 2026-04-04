"""
app/main.py — FastAPI app, all routes.

Full pipeline:
  GET  /health        → server + model status
  POST /transcribe    → audio → text + language        (Phase 1)
  POST /chat          → text → LLM reply text          (Phase 2)
  POST /voice-loop    → audio → audio (full loop)      (Phase 2)
  DELETE /chat/{id}   → clear session memory
"""

import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.asr import WhisperASR
from app.llm import VoiceLLMChain
from app.tts import TTSEngine
from app.schemas import (
    TranscribeResponse,
    HealthResponse,
    ErrorResponse,
    ChatRequest,
    ChatResponse,
    VoiceLoopResponse,
)
from config import settings

log = logging.getLogger(__name__)

TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — loading models...")
    app.state.asr = WhisperASR(model_size=settings.WHISPER_MODEL)
    app.state.llm = VoiceLLMChain()
    app.state.tts = TTSEngine(output_dir=TEMP_DIR)
    log.info("All models loaded. Server ready.")
    yield
    log.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multilingual Voice Chatbot",
    description=(
        "Full voice loop: audio in → Whisper ASR → Groq LLM → gTTS → audio out.\n\n"
        "Supports Arabic, English, French and 96 other languages."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated audio files at /audio/<filename>
app.mount("/audio", StaticFiles(directory=str(TEMP_DIR)), name="audio")


# ── Helper ────────────────────────────────────────────────────────────────────

async def save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename).suffix or ".wav"
    dest = TEMP_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
    async with aiofiles.open(dest, "wb") as f:
        content = await file.read()
        await f.write(content)
    return dest


ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3", "audio/flac",
    "audio/x-flac", "audio/mp4", "audio/m4a",
    "audio/ogg", "audio/webm", "application/octet-stream",
}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Server + model status",
    tags=["System"],
)
async def health():
    asr: WhisperASR = app.state.asr
    return HealthResponse(
        status="ok",
        whisper_model=asr.model_size,
        device=asr.device,
    )


@app.post(
    "/transcribe",
    response_model=TranscribeResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Upload audio → get transcript + detected language",
    tags=["ASR"],
)
async def transcribe(
    file: UploadFile = File(..., description="Audio file — .wav .mp3 .flac .m4a")
):
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{file.content_type}'. Send an audio file."
        )
    audio_path = await save_upload(file)
    try:
        return app.state.asr.transcribe(audio_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if audio_path.exists():
            audio_path.unlink()


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Text + language → LLM reply in same language",
    tags=["LLM"],
)
async def chat(body: ChatRequest):
    try:
        reply = app.state.llm.chat(
            text=body.text,
            language=body.language,
            session_id=body.session_id,
        )
        return ChatResponse(reply=reply, session_id=body.session_id)
    except Exception as e:
        log.exception("LLM chat failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/voice-loop",
    response_model=VoiceLoopResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Full pipeline: audio in → audio out",
    tags=["Voice Loop"],
)
async def voice_loop(
    file: UploadFile = File(..., description="User's voice audio file"),
    session_id: str = "default",
):
    """
    The complete pipeline in one endpoint:

    1. Upload your audio file
    2. Whisper transcribes it + detects language
    3. Groq LLM generates a reply in the same language
    4. gTTS converts the reply to speech
    5. Returns the transcript, reply text, and a URL to the audio reply

    Play the audio by opening the audio_url in your browser
    or using an audio player in your frontend.
    """
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{file.content_type}'."
        )

    audio_path = await save_upload(file)
    reply_audio_path = None

    try:
        # Step 1 — ASR
        asr_result = app.state.asr.transcribe(audio_path)
        log.info(f"ASR → [{asr_result.language}] {asr_result.text[:60]}")

        # Step 2 — LLM
        reply_text = app.state.llm.chat(
            text=asr_result.text,
            language=asr_result.language,
            session_id=session_id,
        )
        log.info(f"LLM → {reply_text[:60]}")

        # Step 3 — TTS
        reply_audio_path = app.state.tts.speak(
            text=reply_text,
            language=asr_result.language,
        )

        audio_url = f"/audio/{reply_audio_path.name}"

        return VoiceLoopResponse(
            transcript=asr_result.text,
            language=asr_result.language,
            language_probability=asr_result.language_probability,
            reply_text=reply_text,
            audio_url=audio_url,
            session_id=session_id,
        )

    except Exception as e:
        # Clean up reply audio if it was created
        if reply_audio_path and reply_audio_path.exists():
            reply_audio_path.unlink()
        log.exception("Voice loop failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up the uploaded input file
        if audio_path.exists():
            audio_path.unlink()


@app.delete(
    "/chat/{session_id}",
    summary="Clear conversation memory for a session",
    tags=["LLM"],
)
async def clear_session(session_id: str):
    app.state.llm.clear_session(session_id)
    return {"cleared": session_id}
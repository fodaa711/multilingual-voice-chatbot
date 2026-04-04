"""
samples/download_samples.py
Generates sample audio files using gTTS for testing Whisper.
No dataset download needed.
"""

from pathlib import Path
from gtts import gTTS

SAMPLES = [
    ("ar", "مرحبا، كيف حالك؟ أنا بخير شكراً.", "ar_sample.wav"),
    ("en", "Hello, how are you? I am doing well, thank you.", "en_sample.wav"),
    ("fr", "Bonjour, comment allez-vous? Je vais très bien merci.", "fr_sample.wav"),
]

SAMPLES_DIR = Path(__file__).parent

for lang, text, filename in SAMPLES:
    out = SAMPLES_DIR / filename
    print(f"Generating {filename} ({lang}) ...")
    tts = gTTS(text=text, lang=lang)
    tts.save(str(out))
    print(f"  Saved → {out.name}")

print("\nDone. Run: pytest tests/test_asr.py -v")
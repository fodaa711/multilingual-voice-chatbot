\---

title: Multilingual Voice Chatbot

emoji: 🎙️

colorFrom: blue

colorTo: purple

sdk: gradio

sdk\_version: "4.44.0"

python\_version: "3.11"

app\_file: app.py

pinned: false

\---



\# 🎙️ Multilingual Voice Chatbot



A full voice loop AI chatbot that understands and responds in \*\*99+ languages\*\*.



\*\*Pipeline:\*\* Record voice → Whisper transcribes → Groq LLM replies → gTTS speaks back



\## How to use



1\. Click the microphone button and speak in any language

2\. Click \*\*Send\*\*

3\. The bot transcribes what you said, replies in the same language, and speaks back



\## Supported Languages



Arabic, English, French, German, Spanish, Italian, Chinese, Japanese, Portuguese, Russian, Turkish, Dutch and 87 more.



\## Tech Stack



| Layer | Technology |

|-------|-----------|

| API | FastAPI + Uvicorn |

| Speech-to-Text | faster-whisper (OpenAI Whisper) |

| Language Model | Groq API (Llama 3.1 8B) |

| Text-to-Speech | gTTS (Google TTS) |

| UI | Gradio |



\## Project Structure



```

multilingual-voice-chatbot/

├── app/

│   ├── main.py        # FastAPI app + all routes

│   ├── asr.py         # Whisper transcription

│   ├── llm.py         # LangChain + Groq chain

│   ├── tts.py         # gTTS text-to-speech

│   └── schemas.py     # Pydantic models

├── app.py             # HuggingFace Spaces entry point

├── config.py          # Settings

└── requirements.txt

```



\## API Endpoints



| Method | Endpoint | Description |

|--------|----------|-------------|

| GET | `/health` | Server + model status |

| POST | `/transcribe` | Audio → transcript + language |

| POST | `/chat` | Text → LLM reply |

| POST | `/voice-loop` | Audio in → audio out |



\## Local Setup



```bash

git clone https://github.com/fodaa711/multilingual-voice-chatbot

cd multilingual-voice-chatbot

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env

\# Add your GROQ\_API\_KEY to .env

uvicorn app.main:app --reload

```



\## Environment Variables



```env

WHISPER\_MODEL=base

LLM\_PROVIDER=groq

GROQ\_API\_KEY=your\_groq\_api\_key\_here

GROQ\_MODEL=llama-3.1-8b-instant

TTS\_ENGINE=gtts

```



Get a free Groq API key at https://console.groq.com



\## License



MIT License


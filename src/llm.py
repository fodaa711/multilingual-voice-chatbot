"""
app/llm.py — LangChain chain using Groq as the LLM provider.

What this file does:
  1. Connects to Groq API using your API key
  2. Keeps a conversation memory per session (remembers what was said before)
  3. Injects a system prompt that tells the model to always reply
     in the same language the user spoke
  4. Exposes one method: chat(text, language, session_id) → reply string

HOW LANGCHAIN WORKS (quick explanation):
  - A "chain" is just: prompt template → LLM → output parser
  - The prompt template has slots: {system}, {history}, {human_input}
  - LangChain fills those slots and sends the full message to Groq
  - Memory stores previous turns so the bot remembers context
"""

import logging
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from config import settings

log = logging.getLogger(__name__)


class VoiceLLMChain:
    """
    Manages a Groq-powered conversation chain.

    One instance is created at server startup and shared across requests.
    Each user session gets its own memory (stored in self._sessions dict).

    Usage:
        llm = VoiceLLMChain()
        reply = llm.chat("مرحبا كيف حالك", language="ar", session_id="user_123")
        print(reply)  # "أنا بخير شكراً، كيف يمكنني مساعدتك؟"
    """

    def __init__(self):
        log.info(f"Loading Groq LLM: {settings.GROQ_MODEL} ...")

        # The Groq LLM — temperature=0.7 means slightly creative
        # but not random. Good for conversation.
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=0.7,
            max_tokens=512,   # keep replies short for TTS
        )

        # Store one memory object per session_id
        # so different users don't share conversation history
        self._sessions: dict[str, ConversationBufferWindowMemory] = {}

        log.info("Groq LLM ready.")

    # ── Public method ─────────────────────────────────────────────────────────

    def chat(
        self,
        text: str,
        language: str = "en",
        session_id: str = "default",
    ) -> str:
        """
        Send a message and get a reply.

        Args:
            text:       what the user said (already transcribed by Whisper)
            language:   detected language code e.g. "ar", "en", "fr"
            session_id: unique ID per user — keeps memories separate

        Returns:
            The LLM reply as a plain string
        """
        memory = self._get_or_create_memory(session_id)
        chain  = self._build_chain(memory, language)

        log.info(f"[{session_id}] [{language}] → {text[:60]}")

        response = chain.predict(input=text)
        reply = response.strip()

        log.info(f"[{session_id}] ← {reply[:60]}")
        return reply

    def clear_session(self, session_id: str) -> None:
        """Wipe the conversation history for a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            log.info(f"Cleared session: {session_id}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_or_create_memory(
        self, session_id: str
    ) -> ConversationBufferWindowMemory:
        """
        Get existing memory for this session or create a new one.

        ConversationBufferWindowMemory keeps only the last k=10 turns
        so the context window doesn't grow forever.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationBufferWindowMemory(
                k=10,                        # remember last 10 turns
                return_messages=True,        # return Message objects not strings
                memory_key="history",
            )
        return self._sessions[session_id]

    def _build_chain(
        self,
        memory: ConversationBufferWindowMemory,
        language: str,
    ) -> ConversationChain:
        """
        Build a conversation chain with a language-aware system prompt.

        The system prompt is built dynamically so it always reminds the
        model which language it detected — this is what makes the bot
        reply in Arabic when it hears Arabic, English when it hears English.
        """
        lang_names = {
            "ar": "Arabic", "en": "English", "fr": "French",
            "de": "German", "es": "Spanish", "it": "Italian",
            "zh": "Chinese", "ja": "Japanese", "pt": "Portuguese",
            "ru": "Russian", "tr": "Turkish", "nl": "Dutch",
        }
        lang_name = lang_names.get(language, language.upper())

        system_content = (
            f"{settings.SYSTEM_PROMPT}\n\n"
            f"The user is speaking {lang_name}. "
            f"You MUST reply in {lang_name} only. "
            f"Do not switch languages under any circumstances."
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_content),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ])

        return ConversationChain(
            llm=self.llm,
            memory=memory,
            prompt=prompt,
            verbose=False,
        )
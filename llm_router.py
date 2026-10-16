"""LLM router: Groq -> Gemini -> Ollama, with exponential backoff."""
import logging
import warnings
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()
log = logging.getLogger("llm_router")


# Current models first; the originally requested (now retired) models are kept last.
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-1.5-flash"]


def _first_working(models, call):
    last = None
    for m in models:
        try:
            return call(m)
        except Exception as e:
            last = e
            msg = str(e).lower()
            if not ("404" in msg or "not found" in msg or "model_not_found" in msg or "decommission" in msg):
                raise
    raise last


class LLMRouter:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.last_provider: Optional[str] = None

    def _with_backoff(self, fn, name: str):
        last = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as e:
                last = e
                msg = str(e).lower()
                retryable = any(k in msg for k in ("429", "rate", "quota", "timeout", "503", "overloaded"))
                if not retryable or attempt == self.max_retries - 1:
                    break
                delay = self.base_delay * (2 ** attempt)
                log.warning("%s retry %d in %.1fs: %s", name, attempt + 1, delay, e)
                time.sleep(delay)
        raise last

    def _groq(self, prompt: str, system_prompt: str) -> str:
        from groq import Groq
        client = Groq(api_key=self.groq_key)
        return _first_working(GROQ_MODELS, lambda m: client.chat.completions.create(
            model=m,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=0.3).choices[0].message.content)

    def _gemini(self, prompt: str, system_prompt: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_key)
        return _first_working(GEMINI_MODELS, lambda m: genai.GenerativeModel(
            m, system_instruction=system_prompt).generate_content(prompt).text)

    def _ollama(self, prompt: str, system_prompt: str) -> str:
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model": "llama3.2", "prompt": prompt, "system": system_prompt, "stream": False},
                          timeout=120)
        r.raise_for_status()
        return r.json()["response"]

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        providers = []
        if self.groq_key:
            providers.append(("groq", self._groq))
        if self.gemini_key:
            providers.append(("gemini", self._gemini))
        if os.getenv("OLLAMA_ENABLED", "").strip().lower() == "true":
            providers.append(("ollama", self._ollama))
        errors = []
        for name, fn in providers:
            try:
                out = self._with_backoff(lambda: fn(prompt, system_prompt), name)
                self.last_provider = name
                return out
            except Exception as e:
                log.warning("provider %s failed: %s", name, e)
                errors.append(f"{name}: {e}")
        raise RuntimeError("All LLM providers failed: " + " | ".join(errors))

from __future__ import annotations
import time
from typing import Callable, Optional, Protocol


class SpeechEngine(Protocol):
    def speak(self, text: str) -> None: ...


class Pyttsx3Engine:
    """Real engine backed by pyttsx3."""

    def __init__(self) -> None:
        import pyttsx3
        self._eng = pyttsx3.init()

    def speak(self, text: str) -> None:
        self._eng.say(text)
        self._eng.runAndWait()


class Narrator:
    def __init__(
        self,
        engine: SpeechEngine,
        min_interval_s: float = 3.0,
        now_fn: Optional[Callable[[], float]] = None,
        enabled: bool = True,
    ) -> None:
        self.engine = engine
        self.min_interval_s = min_interval_s
        self._now = now_fn or time.monotonic
        self.enabled = enabled
        self._last_text: Optional[str] = None
        self._last_time: float = -float("inf")

    def maybe_speak(self, text: str) -> None:
        if not self.enabled or not text:
            return
        now = self._now()
        if text == self._last_text and (now - self._last_time) < self.min_interval_s:
            return
        self.engine.speak(text)
        self._last_text = text
        self._last_time = now

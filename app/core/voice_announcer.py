"""AdaptiveEye Voice Speech & Audio Feedback Engine.

Provides asynchronous spoken audio feedback for click and scroll gestures
(e.g. "Single Click", "Double Click", "Scroll Up", "Scroll Down") using
native Windows SAPI Speech Synthesis with zero performance overhead.
"""

from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Optional
from app.core.logger import get_logger

logger = get_logger("VoiceAnnouncer")


class VoiceAnnouncer:
    """Asynchronous, rate-limited spoken voice announcer for system events."""

    _instance: Optional["VoiceAnnouncer"] = None
    _lock = threading.Lock()

    def __init__(self, enabled: bool = True, speech_rate: int = 2) -> None:
        self.enabled = enabled
        self.speech_rate = speech_rate
        self._queue: queue.Queue = queue.Queue(maxsize=10)
        self._last_speech_time: dict = {}
        self._is_running = True

        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()
        logger.info("VoiceAnnouncer initialized (enabled=%s, rate=%d)", self.enabled, self.speech_rate)

    @classmethod
    def get_instance(cls) -> "VoiceAnnouncer":
        """Singleton accessor for global voice announcer."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def speak(self, phrase: str, min_interval_s: float = 0.0) -> None:
        """Enqueue phrase to be spoken asynchronously.

        Args:
            phrase: Spoken sentence (e.g. "Single Click", "Scroll Up").
            min_interval_s: Optional minimum cooldown in seconds before repeating this phrase.
        """
        if not self.enabled or not phrase:
            return

        now = time.perf_counter()
        if min_interval_s > 0.0:
            last_time = self._last_speech_time.get(phrase, -999.0)
            if now - last_time < min_interval_s:
                return
            self._last_speech_time[phrase] = now

        try:
            self._queue.put_nowait(phrase)
        except queue.Full:
            pass

    def _speech_worker(self) -> None:
        """Background thread consuming phrases and invoking Windows SAPI."""
        while self._is_running:
            try:
                phrase = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                clean = phrase.replace("'", "").replace('"', '').strip()
                ps_cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{clean}')"
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    timeout=5.0,
                )
            except Exception as exc:
                logger.debug("Voice synthesis dispatch exception: %s", exc)
            finally:
                self._queue.task_done()

    def shutdown(self) -> None:
        """Cleanly terminate worker thread."""
        self._is_running = False


# Convenient helper functions
def announce_single_click() -> None:
    VoiceAnnouncer.get_instance().speak("Single Click", min_interval_s=0.8)


def announce_double_click() -> None:
    VoiceAnnouncer.get_instance().speak("Double Click", min_interval_s=0.8)


def announce_scroll_up() -> None:
    VoiceAnnouncer.get_instance().speak("Scroll Up", min_interval_s=1.2)


def announce_scroll_down() -> None:
    VoiceAnnouncer.get_instance().speak("Scroll Down", min_interval_s=1.2)

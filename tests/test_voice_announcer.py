"""Unit tests for VoiceAnnouncer."""

import time
import pytest
from app.core.voice_announcer import (
    VoiceAnnouncer,
    announce_double_click,
    announce_scroll_down,
    announce_scroll_up,
    announce_single_click,
)


def test_voice_announcer_singleton():
    """Verify singleton instance behavior."""
    v1 = VoiceAnnouncer.get_instance()
    v2 = VoiceAnnouncer.get_instance()
    assert v1 is v2
    assert v1.enabled is True


def test_voice_announcer_rate_limit():
    """Verify speech cooldown avoids spamming."""
    announcer = VoiceAnnouncer(enabled=True)
    announcer.speak("Scroll Up", min_interval_s=2.0)
    # Second immediate call should be throttled by min_interval_s
    initial_q_size = announcer._queue.qsize()
    announcer.speak("Scroll Up", min_interval_s=2.0)
    assert announcer._queue.qsize() == initial_q_size


def test_helper_functions_run_without_error():
    """Verify convenience helpers execute safely."""
    announce_single_click()
    announce_double_click()
    announce_scroll_up()
    announce_scroll_down()

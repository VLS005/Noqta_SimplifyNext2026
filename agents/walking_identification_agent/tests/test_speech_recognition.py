"""
Tests for the offline speech recognition pipeline (VAD + grammar-constrained
pocketsphinx). Uses REAL audio fixtures (synthesized once via espeak, not
regenerated at test time) so these tests don't depend on espeak/sox being
installed on whatever machine runs them - only pocketsphinx (already a
project dependency) is needed to run these tests.

Run with: python tests/test_speech_recognition.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.speech.speech_to_text import recognize_confirmation

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _fixture(name: str) -> str:
    return os.path.join(_FIXTURES, name)


def test_recognizes_yes():
    result = recognize_confirmation(_fixture("confirm_yes.wav"))
    assert result == "yes", f"Expected 'yes', got {result!r}"


def test_recognizes_no():
    result = recognize_confirmation(_fixture("confirm_no.wav"))
    assert result == "no", f"Expected 'no', got {result!r}"


def test_recognizes_help():
    result = recognize_confirmation(_fixture("confirm_help.wav"))
    assert result == "help", f"Expected 'help', got {result!r}"


def test_silence_returns_none_not_a_false_word():
    """CRITICAL test - this is the bug we specifically found and fixed.
    Without the VAD gate, silence gets misheard as 'no' at HIGHER confidence
    than genuine speech. This must return None, meaning 'no answer yet' -
    callers (haptic_audio.py) are responsible for treating None as a safe
    default, NOT as an explicit user response."""
    result = recognize_confirmation(_fixture("confirm_silence.wav"))
    assert result is None, (
        f"Expected None (no speech detected) for silence, got {result!r} - "
        f"if this is 'no', the VAD gate in speech_to_text.py has regressed "
        f"and silence is being misinterpreted as an explicit user response again."
    )


def test_wrong_sample_rate_raises_clear_error():
    """pocketsphinx silently produces garbage on wrong sample rates rather
    than erroring - our wrapper must catch this explicitly instead of
    letting it fail silently."""
    try:
        recognize_confirmation(_fixture("confirm_yes_wrong_samplerate.wav"))
        assert False, "Expected a ValueError for 22050Hz audio (expected 16000Hz)"
    except ValueError as e:
        assert "16000" in str(e) and "22050" in str(e), f"Error message unclear: {e}"


if __name__ == "__main__":
    test_recognizes_yes()
    print("PASS: recognizes 'yes'")

    test_recognizes_no()
    print("PASS: recognizes 'no'")

    test_recognizes_help()
    print("PASS: recognizes 'help'")

    test_silence_returns_none_not_a_false_word()
    print("PASS: silence correctly returns None, not a false 'no'")

    test_wrong_sample_rate_raises_clear_error()
    print("PASS: wrong sample rate raises a clear error instead of silent garbage")

    print("\nAll speech recognition tests passed.")

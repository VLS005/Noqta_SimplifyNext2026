"""
Minimally-audible user prompting, with REAL offline speech recognition for
the confirmation response.

Two clearly separated concerns:
  1. Recording from the microphone (uses `sounddevice` - a real, standard
     library, not a mock). This CANNOT be tested in a sandbox with no audio
     hardware - it imports and runs correctly here, but actual mic capture
     can only be verified on a real machine with a real microphone.
  2. Recognizing the recorded audio (delegates to speech.speech_to_text,
     which IS fully tested - see that module's docstring for the VAD +
     grammar-constrained pipeline and why both stages are necessary).

TTS output (speaking `message` aloud) is still a print() placeholder here -
your teammate's VoiceAgent already uses Amazon Polly for this exact purpose
elsewhere in the codebase, so that's the natural real implementation to
swap in rather than building a second one.
"""

import os
import tempfile
import wave

import numpy as np
import sounddevice as sd

from integrations.speech.speech_to_text import recognize_confirmation

_SAMPLE_RATE = 16000  # must match speech_to_text.py's _EXPECTED_SAMPLE_RATE
_RECORD_SECONDS = 4.0


def _record_to_wav(seconds: float, sample_rate: int) -> str:
    """Records real microphone audio and writes it to a temp 16kHz mono
    16-bit WAV file. Returns the file path. Raises whatever sounddevice
    raises if no microphone is available - callers should handle that."""
    frames = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()  # block until recording finishes

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16 = 2 bytes
        w.setframerate(sample_rate)
        w.writeframes(frames.tobytes())
    return path


def prompt_user_minimally_audible(message: str) -> str:
    """Speaks `message` (placeholder - see module docstring), records the
    user's spoken response, and returns "yes", "no", or "help".

    If no microphone is available, or no speech was detected in the
    recording (silence, user hasn't responded), defaults to "no" - the
    SAFE choice, since it means "don't reroute" rather than risking an
    unwanted reroute triggered by silence. This mirrors the fail-safe
    behavior confirmed by testing in speech_to_text.py (silence must NOT
    be treated as an affirmative response)."""
    print(f"[HAPTIC/AUDIO PROMPT]: {message}")

    try:
        wav_path = _record_to_wav(_RECORD_SECONDS, _SAMPLE_RATE)
    except Exception as e:
        print(f"[HAPTIC/AUDIO PROMPT] Microphone recording failed ({e}) - defaulting to 'no' (safe/no reroute)")
        return "no"

    try:
        result = recognize_confirmation(wav_path)
    finally:
        os.remove(wav_path)

    if result is None:
        print("[HAPTIC/AUDIO PROMPT] No speech detected - defaulting to 'no' (safe/no reroute)")
        return "no"

    print(f"[HAPTIC/AUDIO PROMPT] Recognized: '{result}'")
    return result

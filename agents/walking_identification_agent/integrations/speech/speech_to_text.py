"""
Offline speech recognition for confirming the disorientation prompt
("are you lost?" -> user says "yes"/"no"/"help").

Deliberately offline (no cloud API, no network call, no API key) for two
reasons, both grounded in this team's own hackathon notes:
  1. Your mentor's advice: live cloud speech-to-text is fragile on stage -
     avoid it for anything demo-critical.
  2. A constrained 3-word vocabulary is exactly the case where a small
     offline model is a GOOD fit, not a compromise - we don't need general
     speech understanding, just to detect one of three specific words.

TWO-STAGE PIPELINE (this ordering matters - discovered by testing, not
assumed):
  Stage 1 - VAD (voice activity detection): checks whether the recording
    actually contains speech energy at all, using pocketsphinx's Vad class.
  Stage 2 - Grammar-constrained recognition: ONLY runs if Stage 1 found
    speech. Uses a JSGF grammar (confirmation.gram) restricting the decoder
    to exactly {"yes", "no", "help"} - not a soft preference, a hard
    constraint on what it's allowed to output.

WHY STAGE 1 IS NOT OPTIONAL:
Testing this directly (not assumed) showed that pure silence fed into the
grammar-constrained decoder ALONE gets misheard as "no" - and at HIGHER
confidence (score=0.95) than genuine speech saying "no" (score=0.84).
A confidence threshold cannot fix this, since silence scores higher, not
lower. Skipping the VAD stage would mean an agent that hears "no" (falsely
dismissing a safety alert) every time the user simply doesn't respond yet.

KNOWN LIMITATION: grammar-constrained recognition cannot say "I don't
understand" - if real speech IS detected but the words spoken aren't in the
grammar (e.g. the user says "sure" instead of "yes"), it will force-fit the
closest acoustic match among the 3 allowed words, which may be wrong. This
is a real accuracy tradeoff of constrained-vocabulary offline recognition,
not a bug to silently paper over - see recognize_confirmation()'s return
value semantics below.

AUDIO FORMAT REQUIREMENT (found by testing, not documented anywhere
obvious): input audio MUST be 16kHz, 16-bit, mono PCM. Feeding pocketsphinx
the wrong sample rate (e.g. 22050Hz, a common default from many TTS/mic
libraries) produces garbage transcriptions with no error raised - it fails
silently, which makes this an easy mistake to ship undetected. If your
phone mic library defaults to a different rate, resample before calling
recognize_confirmation().
"""

import os
import wave
from typing import Optional

from pocketsphinx import Decoder, Vad, get_model_path

_GRAMMAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confirmation.gram")
_EXPECTED_SAMPLE_RATE = 16000


def _has_speech(pcm_bytes: bytes, vad_mode=None) -> bool:
    """Stage 1: does this audio contain any actual speech at all?"""
    vad = Vad(mode=vad_mode if vad_mode is not None else Vad.MEDIUM_STRICT)
    frame_bytes = vad.frame_bytes
    for i in range(0, len(pcm_bytes) - frame_bytes, frame_bytes):
        frame = pcm_bytes[i:i + frame_bytes]
        if vad.is_speech(frame):
            return True
    return False


def _build_decoder() -> Decoder:
    model_path = get_model_path()
    en_us_path = os.path.join(model_path, "en-us")
    return Decoder(
        hmm=os.path.join(en_us_path, "en-us"),
        dict=os.path.join(en_us_path, "cmudict-en-us.dict"),
        jsgf=_GRAMMAR_PATH,
        lm=False,  # required - pocketsphinx rejects jsgf+lm both being set
    )


def recognize_confirmation(wav_path: str) -> Optional[str]:
    """
    Returns "yes", "no", "help" if that word was recognized with actual
    speech present, or None if no speech was detected at all (silence,
    background noise, user hasn't responded yet).

    Callers should treat None as "no answer yet" - NOT as a false alarm /
    dismissal. Only a real "no" should be treated as the user declining.

    Raises ValueError if the WAV file isn't 16kHz mono 16-bit PCM - fails
    loudly rather than silently producing garbage, unlike pocketsphinx
    itself which does the latter.
    """
    with wave.open(wav_path, "rb") as w:
        if w.getframerate() != _EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"Expected {_EXPECTED_SAMPLE_RATE}Hz audio, got {w.getframerate()}Hz. "
                f"Resample before calling recognize_confirmation() - pocketsphinx "
                f"will silently produce garbage on the wrong sample rate rather "
                f"than raising an error itself."
            )
        if w.getnchannels() != 1:
            raise ValueError(f"Expected mono audio, got {w.getnchannels()} channels.")
        pcm_bytes = w.readframes(w.getnframes())

    if not _has_speech(pcm_bytes):
        return None  # Stage 1 gate: no speech at all - don't even attempt Stage 2

    decoder = _build_decoder()
    decoder.start_utt()
    decoder.process_raw(pcm_bytes, False, True)
    decoder.end_utt()
    hyp = decoder.hyp()
    return hyp.hypstr if hyp and hyp.hypstr else None

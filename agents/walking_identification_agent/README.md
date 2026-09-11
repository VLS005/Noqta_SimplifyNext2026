# Walking Identification Agent (State & Safety Agent)

Detects irregular walking patterns (circling, unusually slow pace, wall-hugging)
during a navigation session, runs a weather -> obstruction -> disorientation
decision tree, and hands off to the Routing Agent when a reroute is needed.

## Folder structure

```
walking_identification_agent/
├── main.py                     # entry point / demo runner
├── agent.py                    # StateSafetyAgent orchestrator
├── config/
│   └── settings.py             # AgentConfig: thresholds, polling, cooldown
├── domain/
│   ├── models.py                # GPSPoint, SensorReading, enums
│   ├── detection.py              # circling / slow-pace / wall-hugging detection
│   └── decision.py               # weather -> obstruction -> lost decision tree
├── api_schemas/
│   └── routing_payload.py        # message contract sent to the Routing Agent
├── integrations/
│   ├── weather_client.py         # mocked - swap for a real weather API
│   ├── vision_client.py          # mocked - swap for triggered-snapshot vision calls
│   ├── haptic_audio.py           # mocked - swap for real haptic/audio output
│   └── routing_client.py         # mocked - swap for your real inter-agent transport
├── utils/
│   └── geo.py                    # haversine distance helper
└── tests/
    └── test_detection.py
```

## Running it

```bash
# Run the demo (simulated sensor stream: normal walking, then circling/slowing)
python main.py

# Run tests
python tests/test_detection.py
```

## Integration points for the rest of the team

- **Personalisation Agent**: supplies `baseline_pace_spm` passed into `StateSafetyAgent.__init__`.
- **Routing Agent**: receives the `RerouteRequest` payload defined in `api_schemas/routing_payload.py`
  — share that file (or `JSON_SCHEMA` from it) with whoever owns the Routing Agent.
- **Camera / Vision**: `integrations/vision_client.py` is where the single triggered-snapshot
  calls happen (no continuous streaming, per the hackathon architecture decision).

## Known simplifications (MVP scope)

- Weather and vision (obstruction/anchor-point detection) calls are mocked -
  swap the function bodies in `integrations/` for real API calls once the
  happy-path demo works.
- `alert_cooldown_sec` in `config/settings.py` prevents nagging the user with
  repeated prompts while a condition persists - tune this during testing.
- `attempt_weather_placeholder` in `config/settings.py` defaults to False -
  the Routing Agent has no "weather" endpoint yet, so this stays off by
  default to avoid an unexplained failure appearing during a live demo.
  See `api_schemas/inbound_weather.py` for details.

## Speech recognition (confirmation prompts: "yes"/"no"/"help")

Offline, not cloud-based - see `integrations/speech/speech_to_text.py`'s
docstring for the full reasoning. Summary:

- Uses `pocketsphinx` with a JSGF grammar (`confirmation.gram`) constraining
  recognition to exactly {"yes", "no", "help"} - not general speech
  understanding, which this doesn't need.
- A voice-activity-detection (VAD) gate runs BEFORE recognition. This is
  NOT optional - testing found that silence fed directly into the
  grammar-constrained recognizer gets misheard as "no" at HIGHER confidence
  than genuine speech. Skipping VAD would mean the agent falsely "hears" a
  dismissal every time the user simply hasn't responded yet.
- Requires 16kHz mono 16-bit PCM audio. pocketsphinx silently produces
  garbage on the wrong sample rate rather than erroring - our wrapper
  raises a clear `ValueError` instead.
- Recording itself (`integrations/haptic_audio.py`) uses `sounddevice` for
  real microphone capture. This needs the PortAudio system library:
  `brew install portaudio` on macOS, `apt-get install portaudio19-dev` on
  Linux, before `pip install -r requirements.txt` will fully work.
- If the microphone is unavailable, or no speech is detected, the agent
  defaults to `"no"` (the safe choice - it means "don't reroute" rather
  than risking an unwanted reroute triggered by silence).
- Text-to-speech (actually *speaking* the prompt aloud) is still a `print()`
  placeholder - the Routing Agent's `VoiceAgent` already uses Amazon Polly
  for this exact purpose, which is the natural real implementation to reuse
  rather than building a second one.
- Tests use committed real audio fixtures (`tests/fixtures/`) synthesized
  once via `espeak`, so running the tests doesn't require `espeak`/`sox` to
  be installed - only `pocketsphinx` (already a dependency).

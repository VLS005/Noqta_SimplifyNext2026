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

- Weather, vision, and haptic/audio calls are mocked - swap the function bodies
  in `integrations/` for real API calls once the happy-path demo works.
- `alert_cooldown_sec` in `config/settings.py` prevents nagging the user with
  repeated prompts while a condition persists - tune this during testing.

# Smart Cane Transit Orchestrator Agent

Prakriti's bus-identification component for the SimplifyNext 2026 Physical AI
smart cane. It autonomously combines a live transit feed with evidence from the
cane-mounted camera, applies a bounded safety policy, sends a command to the
handle's directional haptic actuators, and either waits/boards or hands a
structured accessibility request to the Routing Agent.

The webpage included here is a **hardware simulation and telemetry dashboard**,
not the end-user product. In the intended device, the camera and IMU are mounted
on the cane and the urgent outputs are physical handle vibrations. A phone may
act as an optional connectivity/companion gateway.

## Why this is agentic

This is a goal-directed `perceive -> reconcile -> decide -> act` loop, not one
LLM call. It monitors the ETA, autonomously triggers the cane camera inside the
arrival window, resolves API/camera conflicts, initiates rerouting without
another user command, exposes its decision trace, and applies deterministic
guardrails before telling a blind person to board.

## Run the working demo

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Windows: Copy-Item .env.example .env
uvicorn app.main:app --reload --port 8001
```

Open `http://localhost:8001/demo` for the cane hardware simulator, or use
`http://localhost:8001/docs` to inspect the API. Change `demo_scenario` to `on_time`, `late`,
`wrong_bus`, `off_service`, or `conflict` to demonstrate each autonomous branch.

## Live AWS mode

Set `DEMO_MODE=false`, configure normal AWS credentials, enable the selected
multimodal model in Amazon Bedrock, and set `LTA_ACCOUNT_KEY`. The cane gateway
sends a triggered frame as `cane_image_base64` plus `cane_telemetry`. The service
uses Bedrock Converse with temperature 0 and requires structured visual evidence.
It never continuously streams video.

Set `ROUTING_AGENT_URL` when Lakshmi's service is ready. The agent will POST its
handoff to `{ROUTING_AGENT_URL}/routes/accessible`. Until then, a deterministic
demo route is returned so your flow remains independently presentable.

## Frontend contract

`POST /agent/check-bus` accepts journey details, cane telemetry, and an optional
triggered cane-camera image. The hardware gateway must translate
`haptic_command.pattern` and `haptic_command.actuators` into motor instructions.
The companion interface may speak `spoken_instruction`, but urgent signals should
remain haptic so ambient traffic sound is not masked. Never infer safety from the
text alone; use the returned `action` enum.

Suggested handle layout: `LEFT`, `CENTER`, and `RIGHT` vibration actuators.
`SWEEP-FORWARD` confirms the expected bus; `ONE-LONG` means do not board;
`SCAN-AGAIN` requests a safer camera orientation; `LONG-SHORT-LONG` means the
agent is rerouting.

## Tests

```bash
pytest -q
```

The tests cover confirmed arrival, late bus rerouting, wrong bus rejection,
off-service override, API/vision conflict, and the accessibility handoff.

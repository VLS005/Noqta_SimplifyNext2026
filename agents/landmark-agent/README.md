# Landmark Verification Agent

Triggered camera snapshot + Bedrock vision to confirm a landmark the traveller
was told to look for (for example “the entrance is by the red pillar”). This is
the pivot away from parsing a passerby’s spoken directions.

The companion app sends **text** for the landmark description and a **still
JPEG/PNG** from the cane or phone camera. The agent never runs live
speech-to-text and never streams video.

## Run

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --port 8003
```

Open `http://localhost:8003/docs`. `GET /health` reports `demo_mode` and
`credentials_source` (`settings` when `.env` has sandbox keys, otherwise
`default_chain`).

Paste hackathon `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
`AWS_SESSION_TOKEN` into `agents/landmark-agent/.env`. Those are passed
explicitly into boto3 so this process never silently uses `~/.aws`.

Ports in this repo: Routing `8000`, Transit `8001`, Personalisation `8002`,
this agent `8003`.

## Endpoints

| Method | Path | Role |
|--------|------|------|
| `POST` | `/landmark/set` | Store the landmark description for a `session_id` |
| `POST` | `/landmark/scan` | Inspect one snapshot against that description |
| `POST` | `/landmark/clear` | Drop the stored landmark (journey ended or found) |

`POST /landmark/scan` in `DEMO_MODE=true` does not call Bedrock. Optional
`demo_result` of `"found"` or `"not_found"` forces a canned spoken line; if
omitted, scans alternate not-found then found.

## Live AWS mode

Set `DEMO_MODE=false`, use AWS credentials that can invoke
`apac.amazon.nova-pro-v1:0` in `ap-southeast-1` (same Bedrock vision setup as
the Transit Orchestrator). Send `image_base64` plus `image_media_type`
(`image/jpeg` by default). The model is asked for strict JSON
`{visible, position, distance, confidence}`. Unparseable output is treated as
`visible: false` so a garbled reply never becomes a false “found”.

Guidance strings are short spoken instructions, not screen copy.

## Tests

```bash
pytest -q
```

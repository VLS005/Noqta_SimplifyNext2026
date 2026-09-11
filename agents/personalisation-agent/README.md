# Personalisation Agent (Spatial Memory Agent)

Tracks walking pace per user and retrieves nearby obstacle memories for the
SimplifyNext 2026 smart-cane navigation assistant. It is the HTTP service the
Routing Agent (Lakshmi) calls during a journey.

This agent writes to the **shared** DynamoDB table `DoraDB`. It does not create
its own table.

## Run

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Windows: Copy-Item .env.example .env
uvicorn app.main:app --reload --port 8002
```

Open `http://localhost:8002/docs` for the API. `GET /health` reports demo mode
and the table name.

Set `DEMO_MODE=false`, configure normal AWS credentials for `us-east-1`, and
enable Amazon Titan Text Embeddings V2 in Bedrock for live embeddings. Pace
read/write always uses DynamoDB (`DoraDB`); demo mode only substitutes a
deterministic embedding so the process can start without Bedrock.

## Routing Agent contract

Direct testing API (unchanged):

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/journey/start` | Begin in-memory step/time tracking |
| `POST` | `/journey/tick` | Accumulate `steps_since_last_tick` |
| `POST` | `/journey/end` | Persist pace + rolling average |
| `GET` | `/users/{user_id}/baseline-pace` | Rolling average, or default `95.0` |
| `POST` | `/memory/query` | Vector search for nearby obstacle memories |
| `POST` | `/memory/store` | Store a new obstacle observation |

Compatibility shim for Lakshmi's Routing Agent (`agents/routing-orchestrator/`):

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/inbound/start-timer` | Accepts Routing's `StartTimerEventMessage` as-is |
| `POST` | `/inbound/end-timer` | `{ "session_id": str }` — ends the journey started above |

Routing currently POSTs to `PERSONALIZATION_AGENT_URL`, which defaults to
`http://localhost:8001/inbound/start-timer`. This agent listens on **8002**.
That is a one-line env change on Routing's side (no code change):

```
PERSONALIZATION_AGENT_URL=http://localhost:8002/inbound/start-timer
```

After a journey ends, this agent also mirrors pace to Routing's DoraDB key
`PACE#<user_id>` / `PACE#current` (`average_pace_mps`) and best-effort POSTs
`ROUTING_AGENT_PACE_URL` (default `http://localhost:8000/api/inbound/pace`).
A failed Routing POST is logged and never fails the end-journey response.

Pace m/s uses `DEFAULT_STRIDE_LENGTH_M` (default 0.7 m) as an approximate
stride, not a measured value.

Journey state (step count, start time) is in-process memory keyed by
`journey_id`. Restarting the service drops in-flight journeys; completed paces
live in DynamoDB.

## Shared DoraDB key scheme

Do not invent other key prefixes. Other agents already share this table.

| Item | PK | SK |
|---|---|---|
| Pace journey record | `USER#<user_id>` | `PACE#<timestamp>` |
| Pace rolling profile | `USER#<user_id>` | `PROFILE` |
| Routing ETA mirror | `PACE#<user_id>` | `PACE#current` |
| Obstacle memory | `MEMORY#<geohash>` | `OBSTACLE#<timestamp>` |

Geohash precision is 7 characters (~150 m cells) so nearby obstacles land in
the same partition.

## Vector index (one-time)

`DoraDB` currently has **no** vector indexes. `/memory/store` will still write
items, but `/memory/query` returns `{ "matches": [] }` until the index exists
and is `ACTIVE`.

Create it once with:

```bash
CREATE_VECTOR_INDEX_IF_MISSING=true uvicorn app.main:app --port 8002
```

That calls `UpdateTable` with:

- `IndexName`: `obstacle-memory-index`
- `VectorAttribute`: `embedding`
- `Dimensions`: **1024** (Titan Text Embeddings V2 default; also supports 512 / 256)
- `DistanceFunction`: `COSINE` (score 0 = identical, 2 = opposite)
- `SearchSchema` HASH: `PK` so each search is scoped to `MEMORY#<geohash>`

Leave `CREATE_VECTOR_INDEX_IF_MISSING=false` afterwards. Index creation
backfills asynchronously; `SearchVectors` can 400 until `IndexStatus` is
`ACTIVE`. Query failures are swallowed and returned as an empty match list.

Requires boto3 **1.43.64+** (DynamoDB vector search GA, August 2026).

## Tests

```bash
pytest -q
```

Pace tests use moto against a mock `DoraDB`. Memory vector-search tests mock
the boto3 client directly — moto does not implement `SearchVectors` yet.

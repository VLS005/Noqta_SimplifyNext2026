"""End-to-end smoke test against a local Personalisation Agent (localhost:8002)."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8002"


def request(method: str, path: str, body: dict | None = None) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            parsed = json.loads(raw) if raw else {}
            return resp.status, parsed
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed = json.loads(raw) if raw else {"detail": exc.reason}
        except json.JSONDecodeError:
            parsed = {"detail": raw.decode("utf-8", errors="replace")}
        return exc.code, parsed


def labeled(title: str, status: int, payload: object) -> None:
    print(f"\n=== {title} ===", flush=True)
    print(f"HTTP {status}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)


def main() -> int:
    failures: list[str] = []

    status, payload = request(
        "POST",
        "/journey/start",
        {
            "user_id": "smoketest",
            "route_id": "r1",
            "gps": {"lat": 1.3521, "lon": 103.8198},
            "timestamp": 1000,
        },
    )
    labeled("1. POST /journey/start", status, payload)
    journey_id = payload.get("journey_id") if isinstance(payload, dict) else None
    if status != 200 or not journey_id:
        failures.append("step 1: start did not return journey_id")
        print("\nFAIL: could not start journey; remaining steps skipped", flush=True)
        return 1

    timestamp = 1000
    for i in range(1, 4):
        timestamp += 60
        tick_status, tick_payload = request(
            "POST",
            "/journey/tick",
            {
                "journey_id": journey_id,
                "steps_since_last_tick": 50,
                "timestamp": timestamp,
            },
        )
        labeled(f"2.{i} POST /journey/tick", tick_status, tick_payload)
        if tick_status != 200:
            failures.append(f"step 2: tick {i} HTTP {tick_status}")

    status, ended = request("POST", "/journey/end", {"journey_id": journey_id})
    labeled("3. POST /journey/end", status, ended)
    if status != 200:
        failures.append(f"step 3: end HTTP {status}")
    journey_pace = ended.get("journey_pace_spm") if isinstance(ended, dict) else None
    rolling_avg = ended.get("rolling_avg_pace_spm") if isinstance(ended, dict) else None
    print(f"journey_pace_spm={journey_pace}", flush=True)
    print(f"rolling_avg_pace_spm={rolling_avg}", flush=True)

    status, baseline = request("GET", "/users/smoketest/baseline-pace")
    labeled("4. GET /users/smoketest/baseline-pace", status, baseline)
    if not isinstance(baseline, dict) or baseline.get("is_default") is not False:
        failures.append("step 4: is_default is not false")
    if (
        isinstance(baseline, dict)
        and rolling_avg is not None
        and abs(float(baseline.get("baseline_pace_spm", 0)) - float(rolling_avg)) > 0.01
    ):
        failures.append("step 4: baseline_pace_spm does not match rolling_avg_pace_spm")

    status, stored = request(
        "POST",
        "/memory/store",
        {
            "lat": 1.3521,
            "lon": 103.8198,
            "day_of_week": "Tuesday",
            "time_of_day": "morning",
            "description": "construction barrier blocking left side",
        },
    )
    labeled("5. POST /memory/store", status, stored)
    if status != 200 or not (isinstance(stored, dict) and stored.get("stored") is True):
        failures.append("step 5: store did not return stored=true")
    else:
        print(f"stored={stored.get('stored')} memory_id={stored.get('memory_id')}", flush=True)

    # Vector search is eventually consistent; retry a few times but still FAIL if empty.
    matches: list = []
    status = 0
    payload = {}
    for attempt in range(1, 6):
        if attempt > 1:
            time.sleep(2)
        status, payload = request(
            "POST",
            "/memory/query",
            {
                "lat": 1.3521,
                "lon": 103.8198,
                "day_of_week": "Tuesday",
                "time_of_day": "morning",
            },
        )
        labeled(f"6. POST /memory/query (attempt {attempt})", status, payload)
        if isinstance(payload, dict):
            matches = payload.get("matches") or []
        if matches:
            break

    hit = next(
        (
            m
            for m in matches
            if isinstance(m, dict)
            and "construction barrier" in str(m.get("description", "")).lower()
            and float(m.get("confidence") or 0) > 0
        ),
        None,
    )
    print("\n=== SUMMARY ===", flush=True)
    if hit is None:
        print(
            "FAIL: vector index may not be ACTIVE yet, or embedding call failed -- "
            "check step 1 and DEMO_MODE",
            flush=True,
        )
        if failures:
            print("Other failures: " + "; ".join(failures), flush=True)
        return 1

    if failures:
        print("FAIL: memory match found, but earlier steps failed: " + "; ".join(failures), flush=True)
        return 1

    print(
        "PASS: found match "
        f"description={hit.get('description')!r} confidence={hit.get('confidence')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

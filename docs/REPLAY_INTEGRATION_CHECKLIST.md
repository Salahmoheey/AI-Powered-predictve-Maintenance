# Replay <-> real backend: integration checklist

`tests/test_replay_contract.py` proves the replay and the validator agree using a **mock** server. The check below
must still be done once against the **real** FastAPI backend and recorded in Section 4/9/11 of the Stage 5 report.

## Before you start
- [ ] Backend running, PostgreSQL up, a user with engineer/admin role (JWT token).
- [ ] Backend accepts the 24 channel names in `sensor_type` (`setting_1..3`, `sensor_1..21`).
- [ ] Machines created for the engines you will replay; `data/machine_mapping.json` filled with their real ids.

## Steps
1. Dry run, no backend needed: `python src/replay.py --dry-run --units 1 --limit 3` -> prints one body with 24 readings.
2. Smoke test (5 cycles, one engine):
   ```bash
   export HEXAMIND_API_URL=http://localhost:8000/api/v1
   export HEXAMIND_API_TOKEN=<jwt>
   python src/replay.py --units 1 --limit 5 --delay 0.2
   ```
   Expected: 5 lines with `status=201` and `"inserted": 24`.
3. `GET /machines/{id}/sensor-logs` -> 120 rows (5 cycles x 24), 5 distinct timestamps, increasing with the cycle.
4. Negative tests (expect `422`, nothing stored): send a body with a `NaN`, a missing required channel, a future timestamp,
   a duplicate reading, and an unknown `sensor_type`. Send 1,001 readings -> `413`.
5. Replay 30+ cycles of one engine (`--units 1 --limit 31`), then `POST /machines/{id}/predictions`.
   - With fewer than 30 cycles it must answer `422 INSUFFICIENT_DATA`.
   - Log the 62-value feature vector the backend built for the newest cycle and compare it with
     `artifacts/golden_example.json` (engine 1, cycle 31): all values within 1e-9.
6. Full replay of one engine, then of all 100 (`python src/replay.py`), watching latency (< 200 ms per request target).

## Record the result
| Check | Result | Date | Who |
|---|---|---|---|
| Smoke test (step 2-3) | | | |
| Negative tests (step 4) | | | |
| Feature vector equals golden example (step 5) | | | |
| Full replay (step 6) | | | |

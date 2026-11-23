"""Replay raw NASA FD001 sensor cycles to the HexaMind backend API.

Each engine cycle is sent as ONE request to `POST /machines/{machine_id}/sensor-logs`
(Stage 4, section 6) with the 24 raw channels in long format:

    {"readings": [{"sensor_type": "sensor_2", "value": 642.3, "timestamp": "2026-11-21T10:00:00Z"}, ...]}

All readings of one cycle share one timestamp, and timestamps increase with the cycle
number, so the backend can recover the cycle order from `timestamp` alone. See
docs/API_INPUT_CONTRACT.md.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import json
import os
import time

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "test_FD001.txt"
MAPPING_PATH = PROJECT_ROOT / "data" / "machine_mapping.json"
COLUMNS = (["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
           + [f"sensor_{i}" for i in range(1, 22)])
CHANNELS = COLUMNS[2:]
DEFAULT_CYCLE_SECONDS = 60


def load_machine_mapping(mapping_path=MAPPING_PATH):
    """Load explicit NASA unit_id -> backend machine_id mapping from JSON."""
    path = Path(mapping_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Mapping file not found: {path}. Create it from machine_mapping.example.json "
            "and fill in real backend machine IDs; never assume unit_id == machine_id."
        )
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Machine mapping must be a non-empty JSON object")
    return {int(unit): str(machine_id) for unit, machine_id in mapping.items()}


def prepare_request_body(record, timestamp):
    """Convert one raw NASA row to the sensor-logs request body (raw values, no scaling)."""
    iso = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "readings": [
            {"sensor_type": channel, "value": float(record[channel]), "timestamp": iso}
            for channel in CHANNELS
        ]
    }


class ReplaySource:
    """Iterate through raw FD001 rows in engine/cycle order, yielding (unit_id, cycle, body).

    Timestamps are simulated and never in the future: the last cycle of every engine is
    stamped `cycle_seconds` before `now`, earlier cycles are spaced `cycle_seconds` apart.
    """

    def __init__(self, file_path=RAW_PATH, units=None, limit=None,
                 cycle_seconds=DEFAULT_CYCLE_SECONDS, now=None):
        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Raw NASA file not found: {self.file_path}")
        data = pd.read_csv(self.file_path, sep=r"\s+", header=None, names=COLUMNS)
        self.lengths = data.groupby("unit_id")["cycle"].max().to_dict()  # full engine lengths
        if units:
            data = data[data["unit_id"].isin(units)]
        data = data.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        if limit is not None:
            data = data.groupby("unit_id", sort=False).head(limit).reset_index(drop=True)
        self.data = data
        self.cycle_seconds = cycle_seconds
        self.now = now or datetime.now(timezone.utc)
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.data):
            raise StopIteration
        record = self.data.iloc[self.position].to_dict()
        self.position += 1
        unit_id, cycle = int(record["unit_id"]), int(record["cycle"])
        # Stamp relative to the engine's full length so a --limit run still keeps the order.
        total = int(self.lengths[unit_id])
        timestamp = self.now - timedelta(seconds=(total - cycle + 1) * self.cycle_seconds)
        return unit_id, cycle, prepare_request_body(record, timestamp)

    def reset(self):
        """Restart replay from the first engine cycle."""
        self.position = 0


def send_record(session, base_url, machine_id, body, token=None, timeout=15):
    """POST one cycle to the backend's sensor-logs endpoint and return the response."""
    url = f"{base_url.rstrip('/')}/machines/{machine_id}/sensor-logs"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = session.post(url, json=body, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Replay FD001 test cycles to the HexaMind API")
    parser.add_argument("--units", type=int, nargs="+", help="NASA unit_ids to replay (default: all)")
    parser.add_argument("--limit", type=int, help="max cycles per engine (smoke tests)")
    parser.add_argument("--delay", type=float,
                        default=float(os.getenv("REPLAY_DELAY_SECONDS", "1")),
                        help="seconds to wait between requests")
    parser.add_argument("--cycle-seconds", type=int,
                        default=int(os.getenv("REPLAY_CYCLE_SECONDS", DEFAULT_CYCLE_SECONDS)),
                        help="simulated time between two cycles of an engine")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the first request body and exit; no backend needed")
    return parser.parse_args(argv)


def main(argv=None):
    """Replay raw test cycles; configure API URL, token, mapping file, and delay."""
    args = parse_args(argv)
    replay = ReplaySource(units=args.units, limit=args.limit, cycle_seconds=args.cycle_seconds)
    if args.dry_run:
        unit_id, cycle, body = next(replay)
        print(f"unit_id={unit_id} cycle={cycle}")
        print(json.dumps(body, indent=2))
        return
    base_url = os.getenv("HEXAMIND_API_URL")
    if not base_url:
        raise EnvironmentError("Set HEXAMIND_API_URL, e.g. http://localhost:8000/api/v1")
    token = os.getenv("HEXAMIND_API_TOKEN")
    mapping = load_machine_mapping()
    with requests.Session() as session:
        for unit_id, cycle, body in replay:
            if unit_id not in mapping:
                raise KeyError(f"No machine_id mapping configured for NASA unit_id {unit_id}")
            response = send_record(session, base_url, mapping[unit_id], body, token=token)
            print(f"unit_id={unit_id} machine_id={mapping[unit_id]} cycle={cycle} "
                  f"status={response.status_code} {response.text[:80]}")
            time.sleep(args.delay)


if __name__ == "__main__":
    main()

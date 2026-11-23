"""Replay -> (mock) backend contract test. A real-backend run is described in
docs/REPLAY_INTEGRATION_CHECKLIST.md."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay import ReplaySource, send_record  # noqa: E402
from serving import load_artifacts, raw_window_to_features, readings_to_frame  # noqa: E402

RECEIVED = []


class MockBackend(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RECEIVED.append((self.path, self.headers.get("Authorization"), body))
        payload = json.dumps({"inserted": len(body["readings"]), "first_log_id": 1, "last_log_id": 24}).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), MockBackend)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/api/v1"
    httpd.shutdown()


def test_replay_posts_the_stage4_body_and_backend_can_rebuild_features(server):
    RECEIVED.clear()
    replay = ReplaySource(units=[1], limit=31)
    with requests.Session() as session:
        for unit_id, cycle, body in replay:
            response = send_record(session, server, "machine-7", body, token="secret")
            assert response.status_code == 201
    assert len(RECEIVED) == 31
    path, auth, first = RECEIVED[0]
    assert path == "/api/v1/machines/machine-7/sensor-logs" and auth == "Bearer secret"
    assert list(first) == ["readings"] and len(first["readings"]) == 24
    assert {r["timestamp"] for r in first["readings"]} == {first["readings"][0]["timestamp"]}
    assert set(first["readings"][0]) == {"sensor_type", "value", "timestamp"}
    # Round trip: what the replay sent is accepted by the backend-side validator and
    # produces a 62-value feature vector in the documented order.
    artifacts = load_artifacts()
    readings = [r for _, _, body in RECEIVED for r in body["readings"]]
    window = readings_to_frame(readings, artifacts)
    assert len(window) == 31 and window.index.is_monotonic_increasing
    vector = raw_window_to_features(window, artifacts, min_rows=30)
    assert list(vector) == artifacts["model_feature_columns"]


def test_replay_timestamps_increase_with_cycle_and_never_lie_in_the_future():
    replay = ReplaySource(units=[1, 2], limit=5)
    stamps = {}
    for unit_id, cycle, body in replay:
        stamps.setdefault(unit_id, []).append(body["readings"][0]["timestamp"])
    for values in stamps.values():
        assert values == sorted(values) and len(set(values)) == len(values)
    from datetime import datetime, timezone
    assert max(v for values in stamps.values() for v in values) <= \
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

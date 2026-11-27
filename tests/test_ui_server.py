"""Tests for the token label-quality inspector UI server (ui/server.py)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ui.server import build_server, inspect_labels

SENTENCES = [
    {
        "tokens": ["The", "cat", "sat"],
        "labels": [0, 0, 1],
        "pred_probs": [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.1, 0.85, 0.05]],
    },
    {
        "tokens": ["Dogs", "run", "fast"],
        "labels": [2, 0, 0],
        "pred_probs": [[0.05, 0.1, 0.85], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1]],
    },
]


@pytest.fixture()
def server():
    srv = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/inspect",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_inspect_endpoint_runs_real_ranking(server):
    status, payload = _post(server, {"sentences": SENTENCES, "threshold": 0.7})
    assert status == 200
    assert len(payload["sentence_scores"]) == 2
    assert len(payload["token_scores"][0]) == 3
    assert payload["issues"]  # the low-confidence token is flagged
    assert payload["token_score_method"] == "self_confidence"


def test_result_is_json_serializable():
    result = inspect_labels(SENTENCES, threshold=0.7)
    json.dumps(result)  # must not raise


def test_health_and_index_served(server):
    health = json.loads(urllib.request.urlopen(server + "/api/health").read())
    assert "self_confidence" in health["token_score_methods"]
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "inspector" in html and "/api/inspect" in html


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"sentences": []}, "sentences must be a non-empty list"),
        ({"sentences": SENTENCES, "threshold": 1.5}, "threshold must be in (0, 1)"),
        ({"sentences": SENTENCES, "token_score_method": "magic"}, "token_score_method must be one of"),
        (
            {"sentences": [{"tokens": ["a"], "labels": [5], "pred_probs": [[0.5, 0.5]]}]},
            "labels[0] must be in [0, 1]",
        ),
        (
            {"sentences": [{"tokens": ["a"], "labels": [0], "pred_probs": [[0.5]]}]},
            "must have shape (tokens, classes)",
        ),
    ],
)
def test_invalid_inputs_rejected_with_clear_message(server, payload, message):
    status, body = _post(server, payload)
    assert status == 400
    assert message in body["error"]

def test_index_has_accessibility_features(server):
    """The served page ships skip-link navigation, a main landmark, and reduced-motion guard."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'class="skip-link"' in html, "missing skip link for keyboard users"
    assert '<main class="layout">' in html, "missing main landmark"
    assert 'role="alert"' in html, "error box must announce to assistive tech"
    assert "prefers-reduced-motion" in html, "missing reduced-motion guard"


def test_index_has_loading_and_theme_states(server):
    """The page shows busy feedback during a run and follows the OS light/dark theme."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'aria-busy' in html, "form must announce busy state during a run"
    assert 'class="spinner"' in html, "missing visible busy indicator"
    assert "prefers-color-scheme" in html, "page must follow the OS light/dark theme"


def test_oversized_declared_body_is_rejected_not_hung(server):
    """A lying Content-Length must get a fast 413, never park a handler thread."""
    import socket
    from urllib.parse import urlparse

    u = urlparse(server)
    with socket.create_connection((u.hostname, u.port), timeout=5) as sock:
        sock.sendall(
            b"POST /api/inspect HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
            b"Content-Length: 999999999\r\n\r\n{}"
        )
        first = sock.recv(4096).split(b"\r\n", 1)[0]
    assert b"413" in first, f"expected 413 for oversized body, got {first!r}"


def test_index_disables_heuristic_caching(server):
    """The UI shell must not be heuristically cached — stale HTML after edits was
    observed live because custom do_GET handlers send no cache validators."""
    resp = urllib.request.urlopen(server + "/")
    assert resp.headers.get("Cache-Control") == "no-cache"

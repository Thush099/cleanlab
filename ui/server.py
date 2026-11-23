"""Lightweight local web UI server for cleanlab token-classification label quality.

Serves the label-issue inspector in ``ui/index.html`` and exposes a JSON API that runs the
real ``get_label_quality_scores`` / ``issues_from_scores`` contribution. Standard library
only. Start with ``python ui/server.py`` and open the printed URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from cleanlab.token_classification.rank import (  # noqa: E402
    get_label_quality_scores,
    issues_from_scores,
)

INDEX_HTML = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")

TOKEN_SCORE_METHODS = ("self_confidence", "normalized_margin", "confidence_weighted_entropy")
MAX_SENTENCES = 200


def inspect_labels(
    sentences: list[dict[str, Any]],
    threshold: float = 0.5,
    token_score_method: str = "self_confidence",
) -> dict[str, Any]:
    """Score token labels for quality and flag likely issues.

    Each sentence is ``{"tokens": [...], "labels": [...], "pred_probs": [[...]]}``. Raises
    ValueError with a clear message for invalid inputs.
    """
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("sentences must be a non-empty list")
    if len(sentences) > MAX_SENTENCES:
        raise ValueError(f"sentences must contain at most {MAX_SENTENCES} entries")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    if token_score_method not in TOKEN_SCORE_METHODS:
        raise ValueError(
            f"token_score_method must be one of {list(TOKEN_SCORE_METHODS)}, got {token_score_method!r}"
        )

    labels: list = []
    pred_probs: list = []
    tokens: list = []
    for index, entry in enumerate(sentences):
        if not isinstance(entry, dict):
            raise ValueError(f"sentences[{index}] must be an object")
        toks = entry.get("tokens")
        labs = entry.get("labels")
        probs = entry.get("pred_probs")
        if not isinstance(toks, list) or not toks:
            raise ValueError(f"sentences[{index}].tokens must be a non-empty list")
        if not isinstance(labs, list) or len(labs) != len(toks):
            raise ValueError(
                f"sentences[{index}].labels must be a list with one entry per token"
            )
        if not isinstance(probs, list) or len(probs) != len(toks):
            raise ValueError(
                f"sentences[{index}].pred_probs must be a matrix with one row per token"
            )
        matrix = np.asarray(probs, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            raise ValueError(
                f"sentences[{index}].pred_probs must have shape (tokens, classes) with >= 2 classes"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"sentences[{index}].pred_probs must be finite")
        for li, lab in enumerate(labs):
            if isinstance(lab, bool) or not isinstance(lab, (int, np.integer)):
                raise ValueError(f"sentences[{index}].labels[{li}] must be an integer class id")
            if not 0 <= int(lab) < matrix.shape[1]:
                raise ValueError(
                    f"sentences[{index}].labels[{li}] must be in [0, {matrix.shape[1] - 1}]"
                )
        tokens.append([str(t) for t in toks])
        labels.append([int(l) for l in labs])
        pred_probs.append(matrix)

    sentence_scores, token_scores = get_label_quality_scores(
        labels,
        pred_probs,
        tokens=tokens,
        token_score_method=token_score_method,
    )
    issues = issues_from_scores(
        np.asarray(sentence_scores), token_scores=token_scores, threshold=threshold
    )

    out_tokens = []
    for token_series in token_scores:
        out_tokens.append(
            [
                {"token": str(idx), "score": round(float(val), 4)}
                for idx, val in token_series.items()
            ]
        )
    out_issues = []
    for issue in issues:
        if isinstance(issue, (list, tuple)) and len(issue) == 2:
            out_issues.append({"sentence": int(issue[0]), "token": int(issue[1])})
        else:
            out_issues.append({"sentence": int(issue)})

    return {
        "sentence_scores": [round(float(s), 4) for s in sentence_scores],
        "token_scores": out_tokens,
        "issues": out_issues,
        "threshold": threshold,
        "token_score_method": token_score_method,
    }


def build_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create (but do not start) the UI HTTP server bound to ``host:port``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json(
                    {"status": "ok", "token_score_methods": list(TOKEN_SCORE_METHODS)}
                )
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/api/inspect":
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                result = inspect_labels(
                    sentences=payload.get("sentences", []),
                    threshold=payload.get("threshold", 0.5),
                    token_score_method=payload.get("token_score_method", "self_confidence"),
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the token label-quality inspector UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8132)
    args = parser.parse_args(argv)

    server = build_server(args.host, args.port)
    print(f"Token label-quality inspector running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
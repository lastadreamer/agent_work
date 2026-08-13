"""Local web UI for human/human, human/AI, and AI/AI play."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from xiangqi_engine.config import deepcopy_config, load_config
from xiangqi_engine.play.session import PlaySession

STATIC_DIR = Path(__file__).resolve().parent / "static"


def static_relpath(url_path: str) -> str | None:
    """Map a GET path to a file under STATIC_DIR, or None if not a static request."""
    if url_path == "/":
        return "index.html"
    if url_path.startswith("/static/"):
        rel = url_path[len("/static/") :]
    else:
        rel = url_path.lstrip("/")
    if not rel or ".." in rel.split("/"):
        return None
    return rel


class _Handler(BaseHTTPRequestHandler):
    session: PlaySession
    lock: threading.Lock

    def log_message(self, fmt: str, *args) -> None:
        print("[play]", self.address_string(), fmt % args)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, raw, "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            with self.lock:
                self._json(self.session.state())
            return
        rel = static_relpath(path)
        if rel is None:
            self._send(404, b"not found", "text/plain")
            return
        file_path = (STATIC_DIR / rel).resolve()
        if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.is_file():
            self._send(404, b"not found", "text/plain")
            return
        data = file_path.read_bytes()
        mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        if file_path.suffix == ".js":
            mime = "application/javascript; charset=utf-8"
        elif file_path.suffix == ".css":
            mime = "text/css; charset=utf-8"
        elif file_path.suffix == ".html":
            mime = "text/html; charset=utf-8"
        self._send(200, data, mime)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        body = self._read_json()
        with self.lock:
            if path == "/api/new":
                self._json(
                    self.session.new_game(
                        red=body.get("red", "human"),
                        black=body.get("black", "ai"),
                        simulations=body.get("simulations"),
                        checkpoint=body.get("checkpoint"),
                    )
                )
                return
            if path == "/api/move":
                self._json(self.session.move(str(body.get("iccs", ""))))
                return
            if path == "/api/undo":
                if body.get("human_turn"):
                    self._json(self.session.undo_human_turn())
                else:
                    self._json(self.session.undo(int(body.get("plies", 1))))
                return
            if path == "/api/ai":
                self._json(self.session.ai_move())
                return
        self._send(404, b"not found", "text/plain")


def serve(host: str, port: int, session: PlaySession) -> None:
    handler = type(
        "PlayHandler",
        (_Handler,),
        {"session": session, "lock": threading.Lock()},
    )
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"打开浏览器: http://{host}:{port}", flush=True)
    httpd.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Xiangqi play UI (human/AI)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--simulations", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = deepcopy_config(load_config(args.config))
    play = cfg.get("play", {})
    host = args.host or play.get("host", "127.0.0.1")
    port = int(args.port or play.get("port", 8765))
    session = PlaySession(cfg)
    session.new_game(
        red=play.get("red", "human"),
        black=play.get("black", "ai"),
        simulations=args.simulations,
        checkpoint=args.checkpoint if args.checkpoint is not None else play.get("checkpoint", ""),
    )
    serve(host, port, session)


if __name__ == "__main__":
    main()

"""
ViewerHub: bridges the engine to the browser 3D viewer.

Serves viewer/ over HTTP and pushes JSON events over a WebSocket. broadcast()
is thread-safe because tools execute in worker threads.
"""

from __future__ import annotations

import asyncio
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import websockets

VIEWER_DIR = Path(__file__).resolve().parents[2] / "viewer"


class _QuietHandler(SimpleHTTPRequestHandler):
    model_path: Path | None = None  # the active manifest's .glb, served at /model.glb

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path.split("?")[0] == "/model.glb" and _QuietHandler.model_path:
            data = _QuietHandler.model_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "model/gltf-binary")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()


class ViewerHub:
    def __init__(self, host: str = "127.0.0.1", http_port: int = 8765, ws_port: int = 8766):
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_server = None
        self._http_server: ThreadingHTTPServer | None = None
        self._snapshot: dict = {}  # last manifest/state event, replayed to new clients
        self.on_message = None     # callback(dict) for requests from the viewer (e.g. machine picker)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.http_port}/"

    @property
    def has_clients(self) -> bool:
        return bool(self._clients)

    @staticmethod
    def _manifest_event(manifest, library: list[str]) -> dict:
        return {"type": "manifest", "name": manifest.name, "source": manifest.source,
                "parts": [{"id": p.get("id"), "name": p.get("name")} for p in manifest.parts],
                "library": library or [manifest.name]}

    def set_manifest(self, manifest, library: list[str] | None = None):
        """Switch machines: every viewer rebuilds its model."""
        event = self._manifest_event(manifest, library or [])
        _QuietHandler.model_path = getattr(manifest, "model_path", None)
        for key in ("step",):
            self._snapshot.pop(key, None)
        self._snapshot["manifest"] = event
        self.broadcast(event)

    async def start(self, manifest=None, library: list[str] | None = None):
        self._loop = asyncio.get_running_loop()
        if manifest is not None:
            self._snapshot["manifest"] = self._manifest_event(manifest, library or [])
            _QuietHandler.model_path = getattr(manifest, "model_path", None)
        self._ws_server = await websockets.serve(self._handle, self.host, self.ws_port)
        handler = functools.partial(_QuietHandler, directory=str(VIEWER_DIR))
        self._http_server = ThreadingHTTPServer((self.host, self.http_port), handler)
        threading.Thread(target=self._http_server.serve_forever, daemon=True).start()

    async def stop(self):
        if self._ws_server:
            self._ws_server.close()
            await self._ws_server.wait_closed()
        if self._http_server:
            self._http_server.shutdown()

    async def _handle(self, connection):
        self._clients.add(connection)
        try:
            for event in self._snapshot.values():
                await connection.send(json.dumps(event))
            async for raw in connection:
                try:
                    message = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if self.on_message and isinstance(message, dict):
                    self.on_message(message)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(connection)

    def broadcast(self, event: dict):
        if not self._loop or self._loop.is_closed():
            return
        if event.get("type") in ("step", "status", "spec_stats"):
            self._snapshot[event["type"]] = event
        elif event.get("type") == "reset":
            self._snapshot.pop("step", None)
        payload = json.dumps(event, ensure_ascii=False, default=str)
        try:
            self._loop.call_soon_threadsafe(self._send_all, payload)
        except RuntimeError:
            pass

    def _send_all(self, payload: str):
        for client in list(self._clients):
            asyncio.ensure_future(self._safe_send(client, payload))

    async def _safe_send(self, client, payload: str):
        try:
            await client.send(payload)
        except websockets.ConnectionClosed:
            self._clients.discard(client)


class RecordingHub:
    """Stand-in hub for tests: records events instead of sending them."""

    def set_manifest(self, manifest, library=None):
        self.broadcast(ViewerHub._manifest_event(manifest, library or []))

    def __init__(self, has_clients: bool = True):
        self.events: list[dict] = []
        self._has_clients = has_clients

    @property
    def has_clients(self) -> bool:
        return self._has_clients

    def broadcast(self, event: dict):
        self.events.append(event)

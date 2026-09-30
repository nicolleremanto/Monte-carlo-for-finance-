"""Serveur local de l'interface (bibliothèque standard uniquement).

    python app/serve.py            # puis ouvrir http://localhost:8000/app/

Sert les fichiers du dépôt et expose ``POST /api/<fonction>`` qui appelle
``bridge.call`` avec CPython : calculs plus rapides qu'en WebAssembly et
fonctionnement hors ligne pour le calcul (l'interface détecte le serveur et
n'a alors pas besoin de Pyodide).
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import bridge  # noqa: E402


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _send_json(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") == "/api/ping":
            return self._send_json(json.dumps({"ok": True, "engine": f"CPython {sys.version.split()[0]}"}))
        if self.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/app/")
            self.end_headers()
            return None
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/"):
            return self._send_json(json.dumps({"ok": False, "error": "route inconnue"}), 404)
        name = self.path[len("/api/") :].strip("/")
        length = int(self.headers.get("Content-Length", 0))
        args = self.rfile.read(length).decode("utf-8") if length else "{}"
        return self._send_json(bridge.call(name, args))

    def log_message(self, fmt, *args):  # journal discret
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("  " + fmt % args + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interface web de mcfin")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}/app/"
    print(f"Interface mcfin : {url}  (Ctrl+C pour arrêter)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\narrêt")


if __name__ == "__main__":
    main()

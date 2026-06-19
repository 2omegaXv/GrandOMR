#!/usr/bin/env python3
"""GrandOMR local viewer/bridge for MuseScore note selection."""

import argparse
import json
import mimetypes
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


class BridgeState:
    def __init__(self, root_dir: Path, manifest: dict, notes: list[dict]) -> None:
        self.root_dir = root_dir
        self.manifest = manifest
        self.notes = notes
        self.lock = threading.Lock()
        self.sequence = 0
        self.selector = None
        self.selected = {}
        self.score_path = ""
        self.note_count = 0
        self.last_poll = 0.0

    def set_note(self, omr_id: str) -> dict | None:
        note = next((n for n in self.notes if n.get("omrId") == omr_id), None)
        if note is None:
            return None
        with self.lock:
            self.sequence += 1
            raw_selector = note["selector"]
            selector = {
                "partIdx": raw_selector.get("partIdx", 0),
                "staffIdx": raw_selector.get("staffIdx", 0),
                "voiceIdx": raw_selector.get("voiceIdx", 0),
                "measureIdx": raw_selector.get("measureIdx", 0),
                "beat": raw_selector.get("beat", 0),
                "pitch": raw_selector.get("pitch"),
                "noteIndex": raw_selector.get("noteIndex", 0),
            }
            selector["sequence"] = self.sequence
            selector["omrId"] = omr_id
            self.selector = selector
            self.selected[str(self.sequence)] = {
                "pending": True,
                "ok": None,
                "message": "",
                "omrId": omr_id,
            }
            return selector

    def next_selector(self, last_sequence: int) -> dict:
        with self.lock:
            self.last_poll = time.time()
            if self.selector is None or self.sequence <= last_sequence:
                return {"selector": None}
            return {"selector": self.selector}

    def ack(self, sequence: str, ok: bool, message: str) -> None:
        with self.lock:
            self.selected[str(sequence)] = {
                "pending": False,
                "ok": ok,
                "message": message,
                "omrId": self.selected.get(str(sequence), {}).get("omrId", ""),
            }

    def status(self) -> dict:
        with self.lock:
            return {
                "sequence": self.sequence,
                "scorePath": self.score_path,
                "noteCount": self.note_count,
                "musescoreSeenSecondsAgo": None
                if self.last_poll == 0
                else round(time.time() - self.last_poll, 2),
                "selected": self.selected,
            }


STATE: BridgeState | None = None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        assert STATE is not None
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_html(render_viewer())
            elif parsed.path == "/api/data":
                self.send_json({
                    "manifest": STATE.manifest,
                    "notes": STATE.notes,
                })
            elif parsed.path == "/api/select":
                params = parse_qs(parsed.query)
                omr_id = params.get("omrId", [""])[0]
                selector = STATE.set_note(omr_id)
                if selector is None:
                    self.send_json({"ok": False, "message": "Unknown omrId"})
                else:
                    self.send_json({"ok": True, "selector": selector})
            elif parsed.path == "/api/status":
                self.send_json(STATE.status())
            elif parsed.path == "/next":
                params = parse_qs(parsed.query)
                last_sequence = int(params.get("lastSequence", ["-1"])[0])
                self.send_json(STATE.next_selector(last_sequence))
            elif parsed.path == "/selected":
                params = parse_qs(parsed.query)
                sequence = params.get("sequence", [""])[0]
                ok = params.get("ok", ["false"])[0].lower() == "true"
                message = params.get("message", [""])[0]
                STATE.ack(sequence, ok, message)
                self.send_json({"ok": True})
            elif parsed.path == "/register":
                params = parse_qs(parsed.query)
                with STATE.lock:
                    STATE.score_path = params.get("scorePath", [""])[0]
                    try:
                        STATE.note_count = int(params.get("noteCount", ["0"])[0])
                    except ValueError:
                        STATE.note_count = 0
                    STATE.last_poll = time.time()
                self.send_text("ok")
            elif parsed.path.startswith("/files/"):
                rel = unquote(parsed.path[len("/files/"):])
                self.send_file((STATE.root_dir / rel).resolve())
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_json({"ok": False, "message": str(exc)}, status=500)

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        assert STATE is not None
        root = STATE.root_dir.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            self.send_error(404)
            return
        if not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def render_viewer() -> str:
    return r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>GrandOMR Viewer</title>
  <style>
    body { margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #e9edf2; color: #161a1d; }
    #toolbar { position: sticky; top: 0; z-index: 10; display: flex; gap: 10px; align-items: center;
      padding: 10px 14px; background: #ffffff; border-bottom: 1px solid #c7ced8; }
    #toolbar input { width: 72px; padding: 5px 7px; }
    #status { margin-left: auto; max-width: 560px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #pages { width: min(1100px, calc(100vw - 28px)); margin: 14px auto 40px; }
    .page { position: relative; margin: 0 auto 18px; background: #fff; box-shadow: 0 1px 5px rgba(0,0,0,.22); }
    .page img { display: block; width: 100%; height: auto; }
    .note { position: absolute; border: 2px solid transparent; box-sizing: border-box; cursor: pointer; }
    .note:hover { border-color: #1f78ff; background: rgba(31,120,255,.16); }
    .note.pending { border-color: #ff9d00; background: rgba(255,157,0,.22); }
    .note.selected { border-color: #00a36c; background: rgba(0,163,108,.22); }
  </style>
</head>
<body>
  <div id="toolbar">
    <strong>GrandOMR Viewer</strong>
    <label>Page <input id="pageInput" type="number" min="1" value="1"></label>
    <button id="goBtn">Go</button>
    <span id="status">Loading</span>
  </div>
  <div id="pages"></div>
  <script>
    let notes = [];
    let selectedSeq = null;
    const statusEl = document.getElementById('status');
    const pagesEl = document.getElementById('pages');

    function fileUrl(path) {
      return '/files/' + encodeURIComponent(path).replaceAll('%2F', '/');
    }

    async function loadData() {
      const res = await fetch('/api/data');
      const data = await res.json();
      notes = data.notes;
      renderPages(data.manifest.pages);
      statusEl.textContent = `Loaded ${notes.length} notes`;
      setInterval(refreshStatus, 700);
    }

    function renderPages(pages) {
      pagesEl.innerHTML = '';
      for (const page of pages) {
        const pageNotes = notes.filter(n => n.pageIndex === page.pageIndex);
        const div = document.createElement('div');
        div.className = 'page';
        div.id = `page-${page.pageIndex + 1}`;
        div.style.aspectRatio = `${page.width} / ${page.height}`;
        const img = document.createElement('img');
        img.src = fileUrl(page.imagePath);
        div.appendChild(img);
        for (const note of pageNotes) {
          const [x1, y1, x2, y2] = note.bbox;
          const box = document.createElement('div');
          box.className = 'note';
          box.dataset.omrId = note.omrId;
          box.title = `${note.omrId} ${JSON.stringify(note.selector)}`;
          box.style.left = `${100 * x1 / page.width}%`;
          box.style.top = `${100 * y1 / page.height}%`;
          box.style.width = `${100 * (x2 - x1) / page.width}%`;
          box.style.height = `${100 * (y2 - y1) / page.height}%`;
          box.addEventListener('click', ev => {
            ev.stopPropagation();
            selectNote(note.omrId);
          });
          div.appendChild(box);
        }
        pagesEl.appendChild(div);
      }
    }

    async function selectNote(omrId) {
      document.querySelectorAll('.note').forEach(n => n.classList.remove('pending', 'selected'));
      const el = document.querySelector(`[data-omr-id="${omrId}"]`);
      if (el) el.classList.add('pending');
      const res = await fetch('/api/select?omrId=' + encodeURIComponent(omrId));
      const data = await res.json();
      if (!data.ok) {
        alert(data.message || 'Selection failed');
        return;
      }
      selectedSeq = String(data.selector.sequence);
      statusEl.textContent = `Sent ${omrId}`;
    }

    async function refreshStatus() {
      const res = await fetch('/api/status');
      const data = await res.json();
      const seen = data.musescoreSeenSecondsAgo;
      statusEl.textContent = `MuseScore ${seen === null ? 'not seen' : 'seen ' + seen + 's ago'}; notes scanned ${data.noteCount}`;
      if (selectedSeq && data.selected && data.selected[selectedSeq]) {
        const ack = data.selected[selectedSeq];
        if (!ack.pending) {
          const el = document.querySelector(`[data-omr-id="${ack.omrId}"]`);
          document.querySelectorAll('.note').forEach(n => n.classList.remove('pending', 'selected'));
          if (ack.ok && el) {
            el.classList.add('selected');
          } else if (!ack.ok) {
            alert(ack.message || 'MuseScore selection failed');
          }
          selectedSeq = null;
        }
      }
    }

    document.getElementById('goBtn').addEventListener('click', () => {
      const page = Number(document.getElementById('pageInput').value || '1');
      const el = document.getElementById(`page-${page}`);
      if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
    });
    loadData();
  </script>
</body>
</html>
"""


def load_bundle(path: Path) -> tuple[Path, dict, list[dict]]:
    if path.is_dir():
        root_dir = path
        manifest_path = path / "manifest.json"
    else:
        manifest_path = path
        root_dir = path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    notes_path = root_dir / manifest.get("notesPath", "notes.json")
    notes_doc = json.loads(notes_path.read_text(encoding="utf-8"))
    return root_dir, manifest, notes_doc.get("notes", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="GrandOMR viewer/bridge")
    parser.add_argument("bundle", help="Plugin output directory or manifest.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    global STATE
    root_dir, manifest, notes = load_bundle(Path(args.bundle))
    STATE = BridgeState(root_dir, manifest, notes)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"GrandOMR viewer: {url}")
    print(f"Bundle: {root_dir}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

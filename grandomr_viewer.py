#!/usr/bin/env python3
"""GrandOMR local viewer/bridge for MuseScore note selection."""

import argparse
import json
import mimetypes
import shutil
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


class BridgeState:
    def __init__(self, root_dir: Path, manifest: dict, notes: list[dict], debug: bool = False) -> None:
        self.root_dir = root_dir
        self.manifest = manifest
        self.notes = notes
        self.debug = debug
        self.note_by_id = {note.get("omrId"): note for note in notes if note.get("omrId")}
        self.lock = threading.Lock()
        self.sequence = 0
        self.selector = None
        self.command_sequence = 0
        self.command = None
        self.selected = {}
        self.score_path = ""
        self.note_count = 0
        self.score_mode = "unknown"
        self.current_part_idx = None
        self.current_part_name = ""
        self.part_names = [
            str(part.get("name") or part.get("partName") or f"Part {idx + 1}")
            for idx, part in enumerate(manifest.get("parts", []))
        ]
        self.last_poll = 0.0

    def set_note(self, omr_id: str, mode: str = "main", part_idx: int | None = None) -> dict | None:
        note = self.note_by_id.get(omr_id)
        if note is None:
            return None
        score_note_index = note.get("scoreNoteIndex") if mode == "main" else note.get("partScoreNoteIndex")
        if score_note_index is None:
            raise ValueError(f"No MuseScore note index for {omr_id} in mode={mode}")
        if mode == "part" and part_idx is not None and note.get("partIdx") != part_idx:
            raise ValueError(
                f"Clicked note belongs to part {note.get('partIdx')}, but viewer is set to part {part_idx}"
            )
        with self.lock:
            self.sequence += 1
            selector = {
                "scoreNoteIndex": score_note_index,
                "omrId": omr_id,
                "mode": mode,
                "partIdx": part_idx,
            }
            selector["sequence"] = self.sequence
            self.selector = selector
            self.selected[str(self.sequence)] = {
                "pending": True,
                "ok": None,
                "message": "",
                "omrId": omr_id,
            }
            return selector

    def set_command(self, name: str) -> dict:
        with self.lock:
            self.command_sequence += 1
            self.command = {
                "sequence": self.command_sequence,
                "name": name,
            }
            return self.command

    def update_score_state(
        self,
        score_path: str,
        note_count: int,
        score_mode: str,
        current_part_idx: int | None,
        current_part_name: str,
        part_names: list[str],
    ) -> None:
        with self.lock:
            if current_part_idx is None and score_mode == "part" and current_part_name:
                try:
                    current_part_idx = self.part_names.index(current_part_name)
                except ValueError:
                    current_part_idx = None
            self.score_path = score_path
            self.note_count = note_count
            self.score_mode = score_mode
            self.current_part_idx = current_part_idx
            self.current_part_name = current_part_name
            if part_names:
                self.part_names = part_names
            self.last_poll = time.time()

    def next_selector(self, last_sequence: int, last_command_sequence: int = -1) -> dict:
        with self.lock:
            self.last_poll = time.time()
            selector = None
            command = None
            if self.selector is not None and self.sequence > last_sequence:
                selector = self.selector
            if self.command is not None and self.command_sequence > last_command_sequence:
                command = self.command
            return {"selector": selector, "command": command}

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
                "commandSequence": self.command_sequence,
                "scorePath": self.score_path,
                "noteCount": self.note_count,
                "scoreMode": self.score_mode,
                "currentPartIdx": self.current_part_idx,
                "currentPartName": self.current_part_name,
                "partNames": self.part_names,
                "musescoreSeenSecondsAgo": None
                if self.last_poll == 0
                else round(time.time() - self.last_poll, 2),
                "selected": self.selected,
                "debug": self.debug,
            }


STATE: BridgeState | None = None


def find_musescore(explicit_path: str | None) -> str:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"MuseScore CLI not found: {explicit_path}")
        return str(path)
    for candidate in ("MuseScore4.exe", "MuseScore4"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "MuseScore CLI not found. Pass --musescore \"C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe\""
    )


def run_score_elements(musescore: str, score_path: Path) -> list[dict]:
    result = subprocess.run(
        [musescore, "--score-elements", "-f", str(score_path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            "MuseScore --score-elements failed with code "
            f"{result.returncode}\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MuseScore --score-elements returned invalid JSON: {exc}") from exc


def midi_from_name(name: str | None) -> int | None:
    if not name:
        return None
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    text = str(name)
    if len(text) < 2 or text[0] not in base:
        return None
    idx = 1
    alter = 0
    while idx < len(text) and text[idx] in ("#", "b"):
        alter += 1 if text[idx] == "#" else -1
        idx += 1
    try:
        octave = int(text[idx:])
    except ValueError:
        return None
    return (octave + 1) * 12 + base[text[0]] + alter


def build_score_note_map(score_elements: list[dict]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    all_note_rows: list[dict] = []
    element_order = 0
    for part_idx, part in enumerate(score_elements):
        pending_notes_by_slot: dict[tuple, list[dict]] = {}
        pending_chord_by_slot: dict[tuple, dict] = {}
        for element in part.get("elements", []):
            element_type = element.get("type")
            slot = (
                element.get("staffIdx"),
                element.get("voiceIdx"),
                element.get("measureIdx"),
                element.get("beat"),
            )
            if element_type == "Note":
                note_row = {
                    "partIdx": part_idx,
                    "staffIdx": element.get("staffIdx"),
                    "voiceIdx": element.get("voiceIdx"),
                    "measureIdx": element.get("measureIdx"),
                    "beat": element.get("beat"),
                    "name": element.get("name"),
                    "pitch": midi_from_name(element.get("name")),
                    "duration": element.get("duration"),
                    "_elementOrder": element_order,
                }
                all_note_rows.append(note_row)
                pending_notes_by_slot.setdefault(slot, []).append(note_row)
                pending_chord_by_slot.pop(slot, None)
                element_order += 1
            elif element_type == "Chord":
                notes = element.get("notes") or []
                chord_rows = []
                for chord_note_index, note in enumerate(notes):
                    note_row = {
                        "partIdx": part_idx,
                        "staffIdx": element.get("staffIdx"),
                        "voiceIdx": element.get("voiceIdx"),
                        "measureIdx": element.get("measureIdx"),
                        "beat": element.get("beat"),
                        "name": note.get("name"),
                        "pitch": midi_from_name(note.get("name")),
                        "duration": element.get("duration"),
                        "chordNoteIndex": chord_note_index,
                        "chordNoteCount": len(notes),
                        "_elementOrder": element_order,
                    }
                    all_note_rows.append(note_row)
                    chord_rows.append(note_row)
                    element_order += 1
                if chord_rows:
                    pending_chord_by_slot[slot] = {"rows": chord_rows, "nextLyricIndex": 0}
                    pending_notes_by_slot.pop(slot, None)
            elif element_type == "Lyrics":
                text = str(element.get("text") or "")
                if not text.startswith("GOMR:"):
                    continue
                omr_id = text[len("GOMR:") :]
                pending_chord = pending_chord_by_slot.get(slot)
                if pending_chord is not None:
                    lyric_index = pending_chord["nextLyricIndex"]
                    chord_rows = pending_chord["rows"]
                    if lyric_index < len(chord_rows):
                        note_row = chord_rows[lyric_index]
                        pending_chord["nextLyricIndex"] = lyric_index + 1
                    else:
                        note_row = None
                else:
                    pending_notes = pending_notes_by_slot.get(slot)
                    note_row = pending_notes.pop() if pending_notes else None
                if note_row is not None:
                    note_row["omrId"] = omr_id
                    rows[omr_id] = note_row
    scan_rows = sorted(
        all_note_rows,
        key=lambda row: (
            int(row.get("staffIdx") or 0) * 4 + int(row.get("voiceIdx") or 0),
            int(row.get("measureIdx") or 0),
            float(row.get("beat") or 0),
            row.get("_elementOrder", 0),
        ),
    )
    for score_note_index, row in enumerate(scan_rows):
        row["scoreNoteIndex"] = score_note_index
    part_rows: dict[int, list[dict]] = {}
    for row in all_note_rows:
        part_rows.setdefault(int(row.get("partIdx") or 0), []).append(row)
    for part_idx, rows_for_part in part_rows.items():
        min_staff_idx = min(int(row.get("staffIdx") or 0) for row in rows_for_part)
        sorted_part_rows = sorted(
            rows_for_part,
            key=lambda row: (
                int(row.get("staffIdx") or 0) - min_staff_idx,
                int(row.get("voiceIdx") or 0),
                int(row.get("measureIdx") or 0),
                float(row.get("beat") or 0),
                row.get("_elementOrder", 0),
            ),
        )
        for part_score_note_index, row in enumerate(sorted_part_rows):
            row["partScoreNoteIndex"] = part_score_note_index
    for row in rows.values():
        row.pop("_elementOrder", None)
    return rows


def attach_score_note_indices(root_dir: Path, manifest: dict, notes: list[dict], musescore: str) -> None:
    tagged_path = root_dir / manifest.get("taggedMusicxmlPath", "score.tagged.musicxml")
    if not tagged_path.is_file():
        raise FileNotFoundError(f"Tagged MusicXML not found: {tagged_path}")
    score_elements = run_score_elements(musescore, tagged_path)
    if not manifest.get("parts"):
        manifest["parts"] = [
            {"partIdx": idx, "name": str(part.get("name") or part.get("partName") or f"Part {idx + 1}")}
            for idx, part in enumerate(score_elements)
        ]
    id_map = build_score_note_map(score_elements)
    missing = []
    for note in notes:
        omr_id = note.get("omrId")
        mapped = id_map.get(omr_id)
        if mapped is None:
            missing.append(omr_id)
            continue
        note.update(mapped)
        if note.get("partIdx") is not None and note.get("partName") is None:
            part_idx = int(note.get("partIdx") or 0)
            parts = manifest.get("parts", [])
            if 0 <= part_idx < len(parts):
                note["partName"] = parts[part_idx].get("name") or parts[part_idx].get("partName")
    if missing:
        print(f"[Viewer] Warning: {len(missing)} clickable notes have no MuseScore scoreNoteIndex")
    notes[:] = [note for note in notes if note.get("scoreNoteIndex") is not None]
    print(f"[Viewer] MuseScore id map: {len(id_map)} mapped notes from {tagged_path}")


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
                    "debug": STATE.debug,
                })
            elif parsed.path == "/api/select":
                params = parse_qs(parsed.query)
                omr_id = params.get("omrId", [""])[0]
                mode = params.get("mode", ["main"])[0]
                part_idx_text = params.get("partIdx", [""])[0]
                part_idx = None
                if part_idx_text != "":
                    part_idx = int(part_idx_text)
                selector = STATE.set_note(omr_id, mode=mode, part_idx=part_idx)
                if selector is None:
                    self.send_json({"ok": False, "message": "Unknown omrId"})
                else:
                    self.send_json({"ok": True, "selector": selector})
            elif parsed.path == "/api/play-pause":
                command = STATE.set_command("playPause")
                self.send_json({"ok": True, "command": command})
            elif parsed.path == "/api/open-parts":
                command = STATE.set_command("openParts")
                self.send_json({"ok": True, "command": command})
            elif parsed.path == "/api/status":
                self.send_json(STATE.status())
            elif parsed.path == "/next":
                params = parse_qs(parsed.query)
                last_sequence = int(params.get("lastSequence", ["-1"])[0])
                last_command_sequence = int(params.get("lastCommandSequence", ["-1"])[0])
                self.send_json(STATE.next_selector(last_sequence, last_command_sequence))
            elif parsed.path == "/selected":
                params = parse_qs(parsed.query)
                sequence = params.get("sequence", [""])[0]
                ok = params.get("ok", ["false"])[0].lower() == "true"
                message = params.get("message", [""])[0]
                STATE.ack(sequence, ok, message)
                self.send_json({"ok": True})
            elif parsed.path == "/register":
                params = parse_qs(parsed.query)
                try:
                    note_count = int(params.get("noteCount", ["0"])[0])
                except ValueError:
                    note_count = 0
                current_part_idx = None
                current_part_idx_text = params.get("currentPartIdx", [""])[0]
                if current_part_idx_text != "":
                    try:
                        current_part_idx = int(current_part_idx_text)
                    except ValueError:
                        current_part_idx = None
                part_names_text = params.get("partNames", [""])[0]
                part_names = [name for name in part_names_text.split("|") if name]
                STATE.update_score_state(
                    score_path=params.get("scorePath", [""])[0],
                    note_count=note_count,
                    score_mode=params.get("scoreMode", ["unknown"])[0],
                    current_part_idx=current_part_idx,
                    current_part_name=params.get("currentPartName", [""])[0],
                    part_names=part_names,
                )
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
    #detailPanel { display: none; position: sticky; top: 49px; z-index: 9; background: #f7f9fc; border-bottom: 1px solid #c7ced8;
      padding: 8px 14px; grid-template-columns: 1fr auto; gap: 8px; align-items: start; }
    #detailText { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font: 12px Consolas, monospace; max-height: 150px; overflow: auto; }
    #copyBtn { padding: 5px 9px; }
    #modeSelect { max-width: 160px; padding: 5px 7px; }
    #modeDisplay { display: inline-block; min-width: 110px; max-width: 180px; overflow: hidden; text-overflow: ellipsis;
      white-space: nowrap; padding: 5px 7px; border: 1px solid #c7ced8; background: #f4f6f8; color: #30363d; }
    #autoListenLabel { display: inline-flex; gap: 4px; align-items: center; white-space: nowrap; }
    #autoListenLabel input { width: auto; margin: 0; }
    #modeWarning { display: none; color: #c62828; font-weight: 600; white-space: nowrap; }
    #partsNotice { display: none; position: fixed; right: 18px; top: 62px; z-index: 30; width: min(420px, calc(100vw - 36px));
      background: #fff; border: 1px solid #b8c2cf; box-shadow: 0 3px 18px rgba(0,0,0,.22); padding: 12px 14px; }
    #partsNotice strong { display: block; margin-bottom: 6px; }
    #partsNotice p { margin: 0 0 10px; line-height: 1.35; }
    #partsNotice button { padding: 5px 9px; }
    #helpModal { display: none; position: fixed; inset: 0; z-index: 40; background: rgba(0,0,0,.35); }
    #helpDialog { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); width: min(760px, calc(100vw - 36px));
      max-height: min(760px, calc(100vh - 36px)); overflow: auto; background: #fff; border: 1px solid #aeb7c2;
      box-shadow: 0 6px 28px rgba(0,0,0,.28); padding: 18px 20px; }
    #helpDialog h2 { margin: 0 0 12px; font-size: 20px; }
    #helpDialog h3 { margin: 18px 0 8px; font-size: 15px; }
    #helpDialog p, #helpDialog li { line-height: 1.45; }
    #helpDialog code { background: #eef2f6; padding: 1px 4px; }
    #helpDialog .helpActions { display: flex; justify-content: flex-end; margin-top: 14px; }
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
    <select id="modeSelect"></select>
    <span id="modeDisplay">Main score</span>
    <label id="autoListenLabel"><input id="autoListen" type="checkbox" checked> Auto listen</label>
    <span id="modeWarning">Viewer part does not match the active MuseScore tab.</span>
    <button id="partsBtn">Parts</button>
    <button id="playPauseBtn">Play/Pause</button>
    <button id="helpBtn">Help</button>
    <span id="status">Loading</span>
  </div>
  <div id="partsNotice">
    <strong>Open a part in MuseScore</strong>
    <p>The MuseScore Parts window should be open now. If you see it, choose the part you want to open. If nothing appeared, click the main score in MuseScore first, then open Parts manually and choose a part. This message will close automatically once a new part tab is detected.</p>
    <button id="closeNoticeBtn">Close</button>
  </div>
  <div id="helpModal">
    <div id="helpDialog">
      <h2>GrandOMR Viewer Help</h2>
      <p>This page is not a standalone MuseScore replacement. It works together with the MusicXML generated by GrandOMR and the <code>GrandOMR Plugin</code> running inside MuseScore.</p>
      <h3>Basic usage</h3>
      <ol>
        <li>Run GrandOMR with <code>--plugin-output</code> so it writes a plugin bundle containing the MusicXML used by the plugin, page images, and note mapping metadata.</li>
        <li>Start this viewer with <code>python grandomr_viewer.py &lt;plugin-output-folder&gt;</code>. If MuseScore is not in <code>PATH</code>, pass <code>--musescore "&lt;path-to-MuseScore4.exe&gt;"</code>.</li>
        <li>Open the GrandOMR MusicXML written by the <code>-o</code> option in MuseScore. Do not open the MusicXML inside the plugin bundle.</li>
        <li>In MuseScore, enable and run <code>GrandOMR Plugin</code>. Wait until this page shows <code>MuseScore: Connected</code>; note selection and playback controls should only be used after that.</li>
        <li>Once connected, click a note box in this page to select the matching note in MuseScore.</li>
      </ol>
      <h3>Part tab selection</h3>
      <p>The <code>Parts</code> button asks MuseScore to open its Parts window. If the window appears, choose the part you want MuseScore to open. If nothing appears, click the main score in MuseScore first, then open Parts manually and choose a part.</p>
      <p>When MuseScore is focused on a Part tab, clicks in this viewer are matched against that Part tab rather than the full score. Playback also follows the active MuseScore tab, so playing from a Part tab plays that Part.</p>
      <p><code>Auto listen</code> is recommended. It watches which score or Part tab MuseScore is currently showing and updates this page automatically.</p>
      <p>If you turn off <code>Auto listen</code>, changing the Part in this page does not switch MuseScore to that Part. You must also switch MuseScore to the same Part tab yourself. Keep the manual selection here aligned with the active MuseScore tab, otherwise note selection may target the wrong note.</p>
      <div class="helpActions"><button id="closeHelpBtn">Close</button></div>
    </div>
  </div>
  <div id="detailPanel">
    <pre id="detailText">Click a note to show details.</pre>
    <button id="copyBtn">Copy</button>
  </div>
  <div id="pages"></div>
  <script>
    let notes = [];
    let selectedSeq = null;
    let debugMode = false;
    let currentMode = { type: 'main', partIdx: null };
    let partsNoticeActive = false;
    const statusEl = document.getElementById('status');
    const pagesEl = document.getElementById('pages');
    const modeSelect = document.getElementById('modeSelect');
    const modeDisplay = document.getElementById('modeDisplay');
    const autoListenEl = document.getElementById('autoListen');
    const modeWarning = document.getElementById('modeWarning');
    const partsNotice = document.getElementById('partsNotice');
    const helpModal = document.getElementById('helpModal');
    const detailPanel = document.getElementById('detailPanel');
    const detailText = document.getElementById('detailText');
    let lastDetail = '';

    function fileUrl(path) {
      return '/files/' + encodeURIComponent(path).replaceAll('%2F', '/');
    }

    async function loadData() {
      const res = await fetch('/api/data');
      const data = await res.json();
      notes = data.notes;
      debugMode = Boolean(data.debug);
      detailPanel.style.display = debugMode ? 'grid' : 'none';
      populateModeSelect(data.manifest.parts || inferParts());
      renderPages(data.manifest.pages);
      statusEl.textContent = `Loaded ${notes.length} notes`;
      setInterval(refreshStatus, 700);
    }

    function inferParts() {
      const parts = new Map();
      for (const note of notes) {
        if (note.partIdx !== undefined && note.partIdx !== null) {
          parts.set(Number(note.partIdx), note.partName || `Part ${Number(note.partIdx) + 1}`);
        }
      }
      return [...parts.entries()].sort((a, b) => a[0] - b[0]).map(([partIdx, name]) => ({ partIdx, name }));
    }

    function populateModeSelect(parts) {
      const previous = modeSelect.value || 'main';
      modeSelect.innerHTML = '';
      const mainOpt = document.createElement('option');
      mainOpt.value = 'main';
      mainOpt.textContent = 'Main score';
      modeSelect.appendChild(mainOpt);
      for (const part of parts) {
        const partIdx = Number(part.partIdx ?? part.index ?? modeSelect.options.length - 1);
        const opt = document.createElement('option');
        opt.value = `part:${partIdx}`;
        opt.textContent = part.name || part.partName || `Part ${partIdx + 1}`;
        modeSelect.appendChild(opt);
      }
      if ([...modeSelect.options].some(opt => opt.value === previous)) {
        modeSelect.value = previous;
      }
      setModeFromValue(modeSelect.value);
      updateModeControls();
    }

    function syncPartOptions(partNames) {
      if (!Array.isArray(partNames) || partNames.length === 0) return;
      for (let i = 0; i < partNames.length; i++) {
        let opt = modeSelect.querySelector(`option[value="part:${i}"]`);
        if (!opt) {
          opt = document.createElement('option');
          opt.value = `part:${i}`;
          modeSelect.appendChild(opt);
        }
        opt.textContent = partNames[i] || `Part ${i + 1}`;
      }
      updateModeControls();
    }

    function setModeFromValue(value) {
      if (value === 'main') {
        currentMode = { type: 'main', partIdx: null };
      } else if (value.startsWith('part:')) {
        currentMode = { type: 'part', partIdx: Number(value.slice(5)) };
      }
      updateModeControls();
    }

    function modeLabelFromValue(value) {
      for (const opt of modeSelect.options) {
        if (opt.value === value) return opt.textContent;
      }
      return value;
    }

    function currentModeValue() {
      return currentMode.type === 'main' ? 'main' : `part:${currentMode.partIdx}`;
    }

    function scoreModeValue(status) {
      if (status.scoreMode === 'part' && status.currentPartIdx !== null && status.currentPartIdx !== undefined) {
        return `part:${status.currentPartIdx}`;
      }
      if (status.scoreMode === 'main') return 'main';
      return 'unknown';
    }

    function updateModeControls() {
      const auto = autoListenEl.checked;
      modeSelect.style.display = auto ? 'none' : '';
      modeDisplay.style.display = auto ? '' : 'none';
      modeDisplay.textContent = modeLabelFromValue(currentModeValue());
      if (auto) modeWarning.style.display = 'none';
    }

    function updateModeSelectFromScore(status) {
      if (!autoListenEl.checked) return;
      if (status.scoreMode === 'part' && status.currentPartIdx !== null && status.currentPartIdx !== undefined) {
        const value = `part:${status.currentPartIdx}`;
        if (modeSelect.value !== value) {
          modeSelect.value = value;
          setModeFromValue(value);
        }
        if (partsNoticeActive) hidePartsNotice();
      } else if (status.scoreMode === 'main') {
        if (modeSelect.value !== 'main') {
          modeSelect.value = 'main';
          setModeFromValue('main');
        }
      }
    }

    function updateModeWarning(status) {
      if (autoListenEl.checked) {
        modeWarning.style.display = 'none';
        return;
      }
      const actual = scoreModeValue(status);
      if (actual !== 'unknown' && actual !== currentModeValue()) {
        modeWarning.style.display = '';
      } else {
        modeWarning.style.display = 'none';
      }
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
          box.title = `${note.omrId} #${note.scoreNoteIndex ?? 'unmapped'} ${note.name ?? ''}`;
          box.style.left = `${100 * x1 / page.width}%`;
          box.style.top = `${100 * y1 / page.height}%`;
          box.style.width = `${100 * (x2 - x1) / page.width}%`;
          box.style.height = `${100 * (y2 - y1) / page.height}%`;
          box.addEventListener('click', ev => {
            ev.stopPropagation();
            if (debugMode) showNoteDetail(note);
            selectNote(note.omrId);
          });
          div.appendChild(box);
        }
        pagesEl.appendChild(div);
      }
    }

    function compactNote(note) {
      return {
        omrId: note.omrId,
        scoreNoteIndex: note.scoreNoteIndex,
        partScoreNoteIndex: note.partScoreNoteIndex,
        partIdx: note.partIdx,
        pageIndex: note.pageIndex,
        staffIdx: note.staffIdx,
        voiceIdx: note.voiceIdx,
        measureIdx: note.measureIdx,
        beat: note.beat,
        name: note.name,
        pitch: note.pitch,
        chordNoteIndex: note.chordNoteIndex,
        chordNoteCount: note.chordNoteCount,
        musicXmlNoteIndex: note.musicXmlNoteIndex,
        bbox: note.bbox,
        center: note.center,
        debug: note.debug
      };
    }

    function showNoteDetail(note) {
      const detail = compactNote(note);
      lastDetail = JSON.stringify(detail, null, 2);
      detailText.textContent = lastDetail;
    }

    async function selectNote(omrId) {
      document.querySelectorAll('.note').forEach(n => n.classList.remove('pending', 'selected'));
      const el = document.querySelector(`[data-omr-id="${omrId}"]`);
      if (el) el.classList.add('pending');
      const params = new URLSearchParams({ omrId, mode: currentMode.type });
      if (currentMode.partIdx !== null && currentMode.partIdx !== undefined) params.set('partIdx', String(currentMode.partIdx));
      const res = await fetch('/api/select?' + params.toString());
      const data = await res.json();
      if (!data.ok) {
        alert(data.message || 'Selection failed');
        return;
      }
      selectedSeq = String(data.selector.sequence);
      statusEl.textContent = `Sent ${omrId}`;
    }

    async function playPause() {
      const res = await fetch('/api/play-pause', { method: 'GET' });
      const data = await res.json();
      if (!data.ok) {
        alert(data.message || 'Playback command failed');
        return;
      }
      statusEl.textContent = 'Sent play/pause';
    }

    async function openParts() {
      const res = await fetch('/api/open-parts', { method: 'GET' });
      const data = await res.json();
      if (!data.ok) {
        alert(data.message || 'Open Parts command failed');
        return;
      }
      showPartsNotice();
    }

    function showPartsNotice() {
      partsNoticeActive = true;
      partsNotice.style.display = 'block';
    }

    function hidePartsNotice() {
      partsNoticeActive = false;
      partsNotice.style.display = 'none';
    }

    function connectionSummary(seen) {
      if (seen === null || seen === undefined) {
        return { state: 'Disconnected', seenText: 'never seen' };
      }
      const seconds = Number(seen);
      if (Number.isNaN(seconds)) {
        return { state: 'Disconnected', seenText: 'seen unknown' };
      }
      if (seconds < 2) {
        return { state: 'Connected', seenText: `seen ${seconds}s ago` };
      }
      if (seconds <= 10) {
        return { state: 'Stale', seenText: `seen ${seconds}s ago` };
      }
      return { state: 'Disconnected', seenText: `seen ${seconds}s ago` };
    }

    async function refreshStatus() {
      const res = await fetch('/api/status');
      const data = await res.json();
      const seen = data.musescoreSeenSecondsAgo;
      syncPartOptions(data.partNames);
      updateModeSelectFromScore(data);
      updateModeWarning(data);
      const scoreLabel = data.scoreMode === 'part' ? `Part: ${data.currentPartName || data.currentPartIdx}` : data.scoreMode || 'unknown';
      const conn = connectionSummary(seen);
      statusEl.textContent = `MuseScore: ${conn.state}, ${conn.seenText}; ${scoreLabel}; notes scanned ${data.noteCount}`;
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
    modeSelect.addEventListener('change', () => setModeFromValue(modeSelect.value));
    autoListenEl.addEventListener('change', updateModeControls);
    document.getElementById('partsBtn').addEventListener('click', openParts);
    document.getElementById('playPauseBtn').addEventListener('click', playPause);
    document.getElementById('helpBtn').addEventListener('click', () => { helpModal.style.display = 'block'; });
    document.getElementById('closeHelpBtn').addEventListener('click', () => { helpModal.style.display = 'none'; });
    helpModal.addEventListener('click', ev => {
      if (ev.target === helpModal) helpModal.style.display = 'none';
    });
    document.getElementById('closeNoticeBtn').addEventListener('click', hidePartsNotice);
    document.getElementById('copyBtn').addEventListener('click', async () => {
      if (!lastDetail) return;
      try {
        await navigator.clipboard.writeText(lastDetail);
        statusEl.textContent = 'Copied note detail';
      } catch (e) {
        detailText.focus();
        document.execCommand('selectAll');
      }
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
    parser.add_argument("--musescore", help="Path to MuseScore4.exe for --score-elements id mapping")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Show clicked-note debug details in the web viewer")
    args = parser.parse_args()

    global STATE
    root_dir, manifest, notes = load_bundle(Path(args.bundle))
    musescore = find_musescore(args.musescore)
    attach_score_note_indices(root_dir, manifest, notes, musescore)
    STATE = BridgeState(root_dir, manifest, notes, debug=args.debug)
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

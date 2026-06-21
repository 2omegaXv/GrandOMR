"""
Hybrid OMR Pipeline for Orchestral Scores
==========================================
HOMR pipeline (GPU) for staff detection + per-staff recognition,
with OCR-based instrument name identification.

Usage:
    python pipeline.py <image_path> [-o output.musicxml] [--no-gpu]
    python pipeline.py <directory>  [-o output_dir]  [--no-gpu]
"""

import os
import sys
import argparse
import re
import time
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ── Constants ──
INSTRUMENT_ABBREVS = {
    # Flute
    "fl": "Flute", "fl.": "Flute", "flauto": "Flute", "flauti": "Flute",
    "flöte": "Flute", "flöten": "Flute", "flote": "Flute", "floten": "Flute",
    # Oboe
    "ob": "Oboe", "ob.": "Oboe", "oboe": "Oboe", "oboi": "Oboe", "oboen": "Oboe",
    # Clarinet — with German key suffix variants (Cl.Es=Eb, Cl.B=Bb, Cl.A=A, Ci.Es=OCR noise)
    "cl": "Clarinet", "cl.": "Clarinet", "clar": "Clarinet", "clarinetto": "Clarinet",
    "clarinette": "Clarinet", "clarinetten": "Clarinet", "klar": "Clarinet",
    "ci": "Clarinet",
    "cl.es": "Clarinet:Eb", "kl.es": "Clarinet:Eb", "ci.es": "Clarinet:Eb",
    "cl.b": "Clarinet:Bb", "kl.b": "Clarinet:Bb", "ci.b": "Clarinet:Bb",
    "cl.a": "Clarinet:A", "kl.a": "Clarinet:A", "ci.a": "Clarinet:A",
    # Bass Clarinet
    "baßclarinette": "Bass Clarinet", "bassclarinette": "Bass Clarinet",
    "baßklarinette": "Bass Clarinet", "bassklarinette": "Bass Clarinet",
    "bcl": "Bass Clarinet", "b.cl": "Bass Clarinet",
    "bkl": "Bass Clarinet", "bklar": "Bass Clarinet", "bkl.": "Bass Clarinet",
    "babclarinette": "Bass Clarinet", "babklarinette": "Bass Clarinet",
    # Bassoon
    "fg": "Bassoon", "fg.": "Bassoon", "fag": "Bassoon", "fagotto": "Bassoon",
    "fagotte": "Bassoon", "fagott": "Bassoon",
    # Contrabassoon
    "contrafagott": "Contrabassoon", "contrafag": "Contrabassoon", "cfg": "Contrabassoon",
    "kontrafagott": "Contrabassoon", "c.-fag": "Contrabassoon", "c.fag": "Contrabassoon",
    "c.-fag.": "Contrabassoon", "c.fag.": "Contrabassoon",
    "c-fag": "Contrabassoon", "k-fag": "Contrabassoon",
    # Horn
    "cor": "Horn", "cor.": "Horn", "hn": "Horn", "hn.": "Horn", "horn": "Horn",
    "corni": "Horn", "hörner": "Horn", "horner": "Horn", "hr": "Horn", "hr.": "Horn",
    "tenorhorn": "Horn",
    # Trumpet
    "tr": "Trumpet", "tr.": "Trumpet", "trp": "Trumpet", "tromba": "Trumpet",
    "trombe": "Trumpet", "trompete": "Trumpet", "trompeten": "Trumpet", "trpt": "Trumpet",
    "trp.": "Trumpet",
    # Trombone
    "trb": "Trombone", "trb.": "Trombone", "tbn": "Trombone", "tbn.": "Trombone",
    "trombone": "Trombone", "tromboni": "Trombone",
    "pos": "Trombone", "pos.": "Trombone", "posaune": "Trombone", "posaunen": "Trombone",
    # Tuba
    "tuba": "Tuba", "tb": "Tuba", "baßtuba": "Bass Tuba", "basstuba": "Bass Tuba",
    "baß-tuba": "Bass Tuba", "bass-tuba": "Bass Tuba", "babtuba": "Bass Tuba",
    "bab-tuba": "Bass Tuba",
    # Timpani
    "timp": "Timpani", "timp.": "Timpani", "timpani": "Timpani",
    "pk": "Timpani", "pk.": "Timpani", "pauken": "Timpani",
    # Percussion
    "gr.tr": "Bass Drum", "gr. tr.": "Bass Drum", "große trommel": "Bass Drum",
    "grosse trommel": "Bass Drum", "große tr": "Bass Drum",
    "trommel": "Bass Drum", "gr.trommel": "Bass Drum",
    # Violin
    "vl": "Violin", "vl.": "Violin", "vln": "Violin", "vln.": "Violin",
    "vi": "Violin", "vi.": "Violin",
    "violin": "Violin", "violino": "Violin", "violini": "Violin",
    "violinen": "Violin", "violine": "Violin",
    # Viola
    "vla": "Viola", "vla.": "Viola", "viola": "Viola", "viole": "Viola",
    "violen": "Viola", "br": "Viola", "br.": "Viola", "bratsche": "Viola",
    "bratschen": "Viola", "va": "Viola", "va.": "Viola",
    # Cello
    "vc": "Cello", "vc.": "Cello", "vcl": "Cello", "violonc": "Cello",
    "violoncello": "Cello", "violoncelli": "Cello", "cello": "Cello", "celli": "Cello",
    # Contrabass
    "cb": "Contrabass", "cb.": "Contrabass", "kb": "Contrabass", "kb.": "Contrabass",
    "contrabass": "Contrabass", "contrabasso": "Contrabass", "contrabassi": "Contrabass",
    "contrabässe": "Contrabass", "contrabasse": "Contrabass", "kontrabässe": "Contrabass",
    "kontrabasse": "Contrabass", "kontrabass": "Contrabass",
    "b.get": "Contrabass", "b.get.": "Contrabass", "b.gei": "Contrabass",
    "bassgeige": "Contrabass", "bassgeigen": "Contrabass",
    # Others
    "picc": "Piccolo", "picc.": "Piccolo", "piccolo": "Piccolo",
    "eh": "English Horn", "e.h": "English Horn", "e.h.": "English Horn",
    "c.a.": "English Horn", "cor anglais": "English Horn",
    "englisch horn": "English Horn", "englisches horn": "English Horn",
    "arpa": "Harp", "harp": "Harp", "harfe": "Harp",
    "piano": "Piano", "pf": "Piano", "pf.": "Piano",
    "cel": "Celesta", "cel.": "Celesta", "celesta": "Celesta",
}

INSTRUMENT_MIDI = {
    "Violin":       ("strings.violin",        40),
    "Viola":        ("strings.viola",          41),
    "Cello":        ("strings.cello",          42),
    "Contrabass":   ("strings.contrabass",     43),
    "Flute":        ("wind.flutes.flute",      73),
    "Piccolo":      ("wind.flutes.flute.piccolo", 72),
    "Oboe":         ("wind.reed.oboe",         68),
    "English Horn": ("wind.reed.english-horn", 69),
    "Clarinet":     ("wind.reed.clarinet",     71),
    "Bass Clarinet": ("wind.reed.clarinet.bass", 71),
    "Bassoon":      ("wind.reed.bassoon",      70),
    "Contrabassoon": ("wind.reed.contrabassoon", 70),
    "Horn":         ("brass.french-horn",      60),
    "Trumpet":      ("brass.trumpet",          56),
    "Trombone":     ("brass.trombone",          57),
    "Tuba":         ("brass.tuba",             58),
    "Bass Tuba":    ("brass.tuba",             58),
    "Timpani":      ("drum.timpani",           47),
    "Bass Drum":    ("drum.bass-drum",         116),
    "Harp":         ("pluck.harp",             46),
    "Piano":        ("keyboard.piano",          0),
    "Celesta":      ("keyboard.celesta",        8),
}

_KEY_SEMITONES = {
    "C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
    "Gb": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11,
}
_KEY_LETTER_INDEX = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
_TRANSPOSE_ALWAYS_DOWN = {"Horn", "English Horn"}
_EXTRA_OCTAVE = {"Bass Clarinet": -1}

DEFAULT_TRANSPOSE_KEY = {
    "Clarinet": "Bb",
    "Bass Clarinet": "Bb",
    "Horn": "F",
    "Trumpet": "Bb",
    "English Horn": "F",
}


@dataclass
class PluginTokenNote:
    page_index: int
    part_idx: int
    staff_idx: int
    token_index: int
    token_x: float
    token_y: float | None
    pitch: int | None
    token_dist: float | None
    homr_bbox: list[float]
    homr_center: list[float]
    homr_debug_id: int | None = None


@dataclass
class PluginPageData:
    image_path: str
    image_width: int
    image_height: int
    notes: list[PluginTokenNote]


def _compute_transpose(base: str, key: str):
    """Compute (diatonic, chromatic, octave_change) purely from key letter.
    Direction: Horn/English Horn always down; others nearest to unison."""
    semitones = _KEY_SEMITONES.get(key)
    if semitones is None or semitones == 0:
        return None
    letter_idx = _KEY_LETTER_INDEX[key[0]]
    go_down = base in _TRANSPOSE_ALWAYS_DOWN or semitones > 6
    if go_down:
        chromatic = semitones - 12
        diatonic = letter_idx - 7
    else:
        chromatic = semitones
        diatonic = letter_idx
    octave = _EXTRA_OCTAVE.get(base, 0)
    return (diatonic, chromatic, octave)


_KEY_NORMALIZE = {
    "a": "A", "b": "Bb", "bb": "Bb", "c": "C", "d": "D",
    "e": "E", "es": "Eb", "eb": "Eb", "f": "F", "g": "G",
    "ab": "Ab", "as": "Ab",
}


def _parse_instrument_key(name: str):
    """'Clarinet:A' → ('Clarinet', 'A'); 'Clarinet in A' → ('Clarinet', 'A');
    'Horn:Eb' → ('Horn', 'Eb'); 'Violin' → ('Violin', None)."""
    name = name.strip()
    # New colon format from VLM: "Clarinet:A"
    m = re.match(r'^(.+?):([A-Za-z]+)\s*$', name)
    if m:
        raw_key = m.group(2)
        key = _KEY_NORMALIZE.get(raw_key.lower(), raw_key)
        return m.group(1).strip(), key
    # Legacy "in X" format
    m = re.match(r'^(.+?)\s+in\s+([A-Za-z]+)\s*$', name)
    if m:
        raw_key = m.group(2)
        key = _KEY_NORMALIZE.get(raw_key.lower(), raw_key)
        return m.group(1).strip(), key
    return name, None


def _instrument_base(name: str) -> str:
    """Strip 'in X' suffix for INSTRUMENT_MIDI / ORCHESTRAL_ORDER lookups.
    For shared-staff names like 'Bass Drum/Cymbals', use the primary (first) instrument.
    Falls back to longest-prefix fuzzy match (e.g. 'Violin I' → 'Violin')."""
    base = _parse_instrument_key(name)[0]
    if "/" in base:
        base = base.split("/")[0].strip()
    if base not in INSTRUMENT_MIDI:
        match = max((k for k in INSTRUMENT_MIDI if base.startswith(k)), key=len, default=None)
        if match:
            base = match
    return base


_STEP_TO_MIDI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _extract_part_pitches(xml_string: str) -> List[List[int]]:
    """Extract MIDI pitch values for each part from a MusicXML string."""
    root = ET.fromstring(xml_string)
    result = []
    for part in root.findall("part"):
        midis = []
        for note in part.iter("note"):
            pitch = note.find("pitch")
            if pitch is None:
                continue
            step = pitch.find("step")
            octave = pitch.find("octave")
            alter = pitch.find("alter")
            if step is None or octave is None:
                continue
            midi = (int(octave.text) + 1) * 12 + _STEP_TO_MIDI.get(step.text, 0)
            if alter is not None:
                midi += int(float(alter.text))
            midis.append(midi)
        result.append(midis)
    return result


def _midi_from_pitch_element(pitch_el) -> int | None:
    if pitch_el is None:
        return None
    step = pitch_el.find("step")
    octave = pitch_el.find("octave")
    alter = pitch_el.find("alter")
    if step is None or octave is None:
        return None
    try:
        midi = (int(octave.text) + 1) * 12 + _STEP_TO_MIDI.get(step.text, 0)
        if alter is not None:
            midi += int(float(alter.text))
        return midi
    except (TypeError, ValueError):
        return None


def _pitch_name_to_midi(pitch: str) -> int | None:
    if not pitch or pitch in ("_", "PAD"):
        return None
    match = re.match(r"^([A-G])(-?\d+)$", pitch)
    if not match:
        return None
    return (int(match.group(2)) + 1) * 12 + _STEP_TO_MIDI[match.group(1)]


def _note_bbox_from_box(box, x_scale: float, y_scale: float) -> list[float]:
    x1 = min(box.top_left[0], box.bottom_left[0], box.top_right[0], box.bottom_right[0])
    x2 = max(box.top_left[0], box.bottom_left[0], box.top_right[0], box.bottom_right[0])
    y1 = min(box.top_left[1], box.bottom_left[1], box.top_right[1], box.bottom_right[1])
    y2 = max(box.top_left[1], box.bottom_left[1], box.top_right[1], box.bottom_right[1])
    return [
        round(float(x1) * x_scale, 2),
        round(float(y1) * y_scale, 2),
        round(float(x2) * x_scale, 2),
        round(float(y2) * y_scale, 2),
    ]


def _match_staff_token_notes_to_noteheads(
    symbols,
    staff,
    part_idx: int,
    page_index: int,
    image_width: int,
    image_height: int,
    homr_width: int,
    homr_height: int,
) -> list[PluginTokenNote]:
    import math

    unit = staff.average_unit_size
    region_x_min = staff.min_x - 2 * unit
    region_x_max = staff.max_x + 2 * unit
    region_w = region_x_max - region_x_min
    if region_w <= 0:
        return []

    canvas_w = 1280.0
    scale = canvas_w / region_w
    x_scale = image_width / homr_width
    y_scale = image_height / homr_height
    staff_notes = sorted(staff.get_notes(), key=lambda n: (n.center[0], n.center[1]))
    token_candidates = []

    for local_token_index, symbol_item in enumerate(symbols):
        if isinstance(symbol_item, tuple):
            token_index, sym = symbol_item
        else:
            token_index, sym = local_token_index, symbol_item
        if not sym.rhythm.startswith("note_"):
            continue
        if sym.coordinates is None:
            continue
        if math.isnan(sym.coordinates[0]):
            continue
        token_y = None
        if len(sym.coordinates) > 1 and not math.isnan(sym.coordinates[1]):
            token_y = float(sym.coordinates[1])
        token_candidates.append({
            "tokenIndex": token_index,
            "tokenX": float(sym.coordinates[0]),
            "tokenY": token_y,
            "pitch": _pitch_name_to_midi(sym.pitch),
        })

    if not token_candidates:
        return []

    result = []
    for note in staff_notes:
        note_canvas_x = (note.center[0] - region_x_min) * scale
        best_token = None
        best_dist = float("inf")
        for token in token_candidates:
            dist = abs(note_canvas_x - token["tokenX"])
            if dist < best_dist:
                best_dist = dist
                best_token = token
        if best_token is None:
            continue

        # Keep the tolerance loose. trOMR attention x is approximate; this is
        # only a local anchor from a HOMR notehead bbox to the nearest token.
        if best_dist > 90.0:
            continue
        center = [
            round(float(note.center[0]) * x_scale, 2),
            round(float(note.center[1]) * y_scale, 2),
        ]
        result.append(
            PluginTokenNote(
                page_index=page_index,
                part_idx=part_idx,
                staff_idx=part_idx,
                token_index=best_token["tokenIndex"],
                token_x=best_token["tokenX"],
                token_y=best_token["tokenY"],
                pitch=best_token["pitch"],
                token_dist=round(float(best_dist), 2),
                homr_bbox=_note_bbox_from_box(note.box, x_scale, y_scale),
                homr_center=center,
                homr_debug_id=getattr(note.box, "debug_id", None),
            )
        )

    return result


def _split_symbols_by_newline(symbols):
    segments = []
    current = []
    for token_index, sym in enumerate(symbols):
        if sym.rhythm == "newline":
            if current:
                segments.append(current)
                current = []
            continue
        current.append((token_index, sym))
    if current:
        segments.append(current)
    return segments


def _build_plugin_page_data(
    img_path: str,
    page_index: int,
    result_staffs,
    staffs_sorted,
    predictions_original_shape,
) -> PluginPageData:
    image = cv2.imread(img_path)
    if image is None:
        raise RuntimeError(f"Cannot read source image for plugin output: {img_path}")
    image_height, image_width = image.shape[:2]
    homr_height, homr_width = predictions_original_shape[:2]
    notes = []
    part_count = len(result_staffs)
    for part_idx, symbols in enumerate(result_staffs):
        segments = _split_symbols_by_newline(symbols)
        if not segments:
            continue
        for system_idx, segment in enumerate(segments):
            staff_linear_idx = system_idx * part_count + part_idx
            if staff_linear_idx >= len(staffs_sorted):
                continue
            notes.extend(
                _match_staff_token_notes_to_noteheads(
                    segment,
                    staffs_sorted[staff_linear_idx],
                    part_idx=part_idx,
                    page_index=page_index,
                    image_width=image_width,
                    image_height=image_height,
                    homr_width=homr_width,
                    homr_height=homr_height,
                )
            )
    return PluginPageData(
        image_path=str(img_path),
        image_width=image_width,
        image_height=image_height,
        notes=notes,
    )


def _build_plugin_page_data_for_system(
    img_path: str,
    page_index: int,
    result_staffs,
    sys_staves,
    predictions_original_shape,
) -> PluginPageData:
    image = cv2.imread(img_path)
    if image is None:
        raise RuntimeError(f"Cannot read source image for plugin output: {img_path}")
    image_height, image_width = image.shape[:2]
    homr_height, homr_width = predictions_original_shape[:2]
    notes = []
    for part_idx, symbols in enumerate(result_staffs):
        if part_idx >= len(sys_staves):
            continue
        segment = [(idx, sym) for idx, sym in enumerate(symbols) if sym.rhythm != "newline"]
        notes.extend(
            _match_staff_token_notes_to_noteheads(
                segment,
                sys_staves[part_idx],
                part_idx=part_idx,
                page_index=page_index,
                image_width=image_width,
                image_height=image_height,
                homr_width=homr_width,
                homr_height=homr_height,
            )
        )
    return PluginPageData(
        image_path=str(img_path),
        image_width=image_width,
        image_height=image_height,
        notes=notes,
    )


def _build_plugin_page_image_only(img_path: str, page_index: int) -> PluginPageData:
    image = cv2.imread(img_path)
    if image is None:
        raise RuntimeError(f"Cannot read source image for plugin output: {img_path}")
    image_height, image_width = image.shape[:2]
    return PluginPageData(
        image_path=str(img_path),
        image_width=image_width,
        image_height=image_height,
        notes=[],
    )


def _xml_note_records(root: ET.Element):
    records = []
    part_staff_offsets = {}
    next_staff_idx = 0
    for part_idx, part in enumerate(root.findall("part")):
        measures = part.findall("measure")
        max_staff = 1
        for measure in measures:
            for note in measure.findall("note"):
                try:
                    staff_num = int(note.findtext("staff", "1"))
                except ValueError:
                    staff_num = 1
                max_staff = max(max_staff, staff_num)
        part_staff_offsets[part_idx] = next_staff_idx
        next_staff_idx += max_staff

        divisions = 1
        for measure_idx, measure in enumerate(measures):
            attrs = measure.find("attributes")
            if attrs is not None:
                d_text = attrs.findtext("divisions")
                if d_text:
                    try:
                        divisions = int(d_text)
                    except ValueError:
                        pass
            current_time = 0
            last_note_start = 0
            same_pitch_index = {}
            for child in measure:
                if child.tag == "backup":
                    duration = int(child.findtext("duration", "0") or "0")
                    current_time = max(0, current_time - duration)
                    continue
                if child.tag == "forward":
                    duration = int(child.findtext("duration", "0") or "0")
                    current_time += duration
                    continue
                if child.tag != "note":
                    continue

                voice_text = child.findtext("voice", "1")
                try:
                    xml_voice = int(voice_text)
                except ValueError:
                    xml_voice = 1
                try:
                    staff_num = int(child.findtext("staff", "1"))
                except ValueError:
                    staff_num = 1
                local_voice = ((xml_voice - 1) % 4) + 1
                duration = int(child.findtext("duration", "0") or "0")
                is_chord = child.find("chord") is not None
                start = last_note_start if is_chord else current_time

                pitch = _midi_from_pitch_element(child.find("pitch"))
                if pitch is not None:
                    staff_idx = part_staff_offsets[part_idx] + staff_num - 1
                    voice_idx = local_voice - 1
                    same_key = (staff_idx, voice_idx, measure_idx, start, pitch)
                    note_index_same_pitch = same_pitch_index.get(same_key, 0)
                    same_pitch_index[same_key] = note_index_same_pitch + 1
                    beat = start / divisions if divisions else 0.0
                    records.append({
                        "element": child,
                        "partIdx": part_idx,
                        "staffIdx": staff_idx,
                        "voiceIdx": voice_idx,
                        "measureIdx": measure_idx,
                        "beat": float(beat),
                        "pitch": pitch,
                        "noteIndex": note_index_same_pitch,
                    })

                if not is_chord:
                    last_note_start = start
                    current_time += duration
    return records


def _group_xml_records_by_chord(records: list[dict]) -> dict[int, list[list[dict]]]:
    groups_by_part: dict[int, list[list[dict]]] = {}
    for rec in records:
        groups = groups_by_part.setdefault(rec["partIdx"], [])
        key = (rec["staffIdx"], rec["voiceIdx"], rec["measureIdx"], rec["beat"])
        if not groups or groups[-1][0].get("_groupKey") != key:
            rec["_groupKey"] = key
            groups.append([rec])
        else:
            rec["_groupKey"] = key
            groups[-1].append(rec)
    return groups_by_part


def _attach_note_ids_and_build_selectors(xml_string: str, plugin_pages: list[PluginPageData]):
    # Legacy safety path. Formal plugin output is expected to use writer callback
    # prebinding; without it we cannot reliably know which XML notes are clickable.
    root = ET.fromstring(xml_string)
    records = _xml_note_records(root)
    for xml_idx, rec in enumerate(records):
        note_id = f"grandomr-n{xml_idx + 1:06d}"
        rec["element"].set("id", note_id)
    xml_with_ids = ET.tostring(root, encoding="unicode", xml_declaration=True)
    return xml_with_ids, []


def _tag_symbols_for_plugin(result_staffs) -> None:
    for part_idx, symbols in enumerate(result_staffs):
        for token_index, sym in enumerate(symbols):
            setattr(sym, "_grandomr_part_idx", part_idx)
            setattr(sym, "_grandomr_token_index", token_index)


def _make_plugin_note_callback(writer_notes: list[dict]):
    def callback(model_note, xml_note) -> None:
        pitch = _pitch_name_to_midi(getattr(model_note, "pitch", ""))
        if pitch is None:
            return
        writer_notes.append({
            "partIdx": getattr(model_note, "_grandomr_part_idx", None),
            "tokenIndex": getattr(model_note, "_grandomr_token_index", None),
            "pitch": pitch,
            "rhythm": getattr(model_note, "rhythm", None),
        })
    return callback


def _group_bbox_notes_by_visual_chord(notes: list[PluginTokenNote]) -> list[list[PluginTokenNote]]:
    groups: list[list[PluginTokenNote]] = []
    sorted_notes = sorted(
        notes,
        key=lambda n: (n.page_index, n.part_idx, n.staff_idx, n.homr_center[0], n.homr_center[1]),
    )
    for note in sorted_notes:
        if groups:
            prev = groups[-1]
            prev_x = sum(item.homr_center[0] for item in prev) / len(prev)
            same_group = (
                note.page_index == prev[-1].page_index
                and note.part_idx == prev[-1].part_idx
                and note.staff_idx == prev[-1].staff_idx
                and abs(note.homr_center[0] - prev_x) <= 30.0
            )
            if same_group:
                prev.append(note)
                continue
        groups.append([note])
    return groups


def _choose_xml_records_for_bboxes(
    bbox_group: list[PluginTokenNote],
    xml_records: list[dict],
) -> list[tuple[PluginTokenNote, dict]]:
    if not bbox_group or not xml_records:
        return []

    selected_bboxes = list(bbox_group)
    if len(selected_bboxes) > len(xml_records):
        selected_bboxes = sorted(
            selected_bboxes,
            key=lambda n: (
                n.token_dist is None,
                n.token_dist if n.token_dist is not None else float("inf"),
                n.homr_center[1],
            ),
        )[:len(xml_records)]

    selected_bboxes.sort(key=lambda n: n.homr_center[1])
    xml_available = sorted(xml_records, key=lambda r: r.get("_xmlOrder", 0), reverse=True)

    pairs: list[tuple[PluginTokenNote, dict]] = []
    if len(selected_bboxes) == len(xml_available):
        return list(zip(selected_bboxes, xml_available))

    xml_available = sorted(xml_records, key=lambda r: r.get("_xmlOrder", 0))
    for bbox in selected_bboxes:
        if not xml_available:
            break
        best_idx = 0
        best_score = float("inf")
        for idx, rec in enumerate(xml_available):
            if bbox.pitch is not None:
                score = abs(bbox.pitch - rec["pitch"])
            else:
                score = idx
            if score < best_score:
                best_score = score
                best_idx = idx
        pairs.append((bbox, xml_available.pop(best_idx)))
    return pairs


def _prebind_plugin_notes_from_writer_map(
    xml_string: str,
    plugin_page: PluginPageData,
    writer_notes: list[dict],
    id_prefix: str,
):
    """Attach ids using writer callback token identity, then bind HOMR bboxes locally.

    HOMR bboxes choose their nearest trOMR token independently. The token's writer
    callback gives the XML chord group; bboxes at the same visual x are then
    assigned to notes inside that chord by vertical order.
    """
    root = ET.fromstring(xml_string)
    records = _xml_note_records(root)

    if len(writer_notes) != len(records):
        print(
            f"[Plugin] Warning: writer note count {len(writer_notes)} "
            f"!= XML pitched note count {len(records)}"
        )

    for xml_idx, rec in enumerate(records):
        rec["_xmlOrder"] = xml_idx
        if xml_idx >= len(writer_notes):
            continue
        writer = writer_notes[xml_idx]
        rec["_writerPartIdx"] = writer.get("partIdx")
        rec["_writerTokenIndex"] = writer.get("tokenIndex")
        rec["_writerPitch"] = writer.get("pitch")

    token_to_xml_group: dict[tuple[int, int], list[dict]] = {}
    for xml_groups in _group_xml_records_by_chord(records).values():
        for group in xml_groups:
            group_sorted = sorted(group, key=lambda r: r.get("_xmlOrder", 0))
            for rank, rec in enumerate(group_sorted):
                rec["_chordNoteRank"] = rank
                rec["_chordSize"] = len(group)
            for rec in group:
                part_idx = rec.get("_writerPartIdx")
                token_index = rec.get("_writerTokenIndex")
                if part_idx is None or token_index is None:
                    continue
                token_to_xml_group[(part_idx, token_index)] = group

    plugin_notes = []
    used_xml_records = set()
    for bbox_group in _group_bbox_notes_by_visual_chord(plugin_page.notes):
        votes: dict[int, dict] = {}
        for bbox_note in bbox_group:
            xml_group = token_to_xml_group.get((bbox_note.part_idx, bbox_note.token_index))
            if xml_group is None:
                continue
            vote = votes.setdefault(id(xml_group), {
                "group": xml_group,
                "count": 0,
                "dist": 0.0,
            })
            vote["count"] += 1
            vote["dist"] += bbox_note.token_dist if bbox_note.token_dist is not None else 9999.0
        if not votes:
            continue
        chosen = min(votes.values(), key=lambda v: (-v["count"], v["dist"]))
        xml_group = [
            rec for rec in chosen["group"]
            if id(rec) not in used_xml_records
        ]
        if not xml_group:
            continue
        for matched, rec in _choose_xml_records_for_bboxes(bbox_group, xml_group):
            note_id = f"{id_prefix}-n{len(plugin_notes) + 1:06d}"
            rec["element"].set("id", note_id)
            used_xml_records.add(id(rec))
            xml_idx = rec.get("_xmlOrder", 0)
            plugin_notes.append({
                "omrId": note_id,
                "pageIndex": matched.page_index,
                "bbox": matched.homr_bbox,
                "center": matched.homr_center,
                "musicXmlNoteIndex": xml_idx,
                "staffIdx": rec["staffIdx"],
                "partIdx": rec["partIdx"],
                "voiceIdx": rec["voiceIdx"],
                "pitch": rec["pitch"],
                "debug": {
                    "homrDebugId": matched.homr_debug_id,
                    "tromrTokenIndex": matched.token_index,
                    "tromrTokenX": matched.token_x,
                    "tromrTokenY": matched.token_y,
                    "tromrTokenDist": matched.token_dist,
                    "tokenPitch": matched.pitch,
                    "writerTokenIndex": rec.get("_writerTokenIndex"),
                    "writerPitch": rec.get("_writerPitch"),
                    "chordNoteRank": rec.get("_chordNoteRank"),
                    "xmlChordSize": rec.get("_chordSize"),
                    "bboxChordSize": len(bbox_group),
                },
            })
    xml_with_ids = ET.tostring(root, encoding="unicode", xml_declaration=True)
    return xml_with_ids, plugin_notes


def _finalize_prebound_plugin_notes(xml_string: str, plugin_pages: list[PluginPageData]):
    notes = []
    musicxml_index_by_id = {
        rec["element"].get("id"): idx
        for idx, rec in enumerate(_xml_note_records(ET.fromstring(xml_string)))
        if rec["element"].get("id")
    }
    seen = set()
    for page in plugin_pages:
        for note in page.notes:
            if not isinstance(note, dict):
                continue
            omr_id = note.get("omrId")
            if not omr_id or omr_id not in musicxml_index_by_id or omr_id in seen:
                continue
            seen.add(omr_id)
            item = dict(note)
            item["musicXmlNoteIndex"] = musicxml_index_by_id.get(omr_id, item.get("musicXmlNoteIndex"))
            notes.append(item)
    xml_with_ids = ET.tostring(ET.fromstring(xml_string), encoding="unicode", xml_declaration=True)
    return xml_with_ids, notes


def _build_tagged_musicxml(xml_string: str, notes: list[dict]) -> str:
    root = ET.fromstring(xml_string)
    clickable_ids = {note.get("omrId") for note in notes if note.get("omrId")}
    for rec in _xml_note_records(root):
        note_id = rec["element"].get("id")
        if not note_id or note_id not in clickable_ids:
            continue
        lyric = ET.Element("lyric", {"number": "99", "print-object": "no"})
        syllabic = ET.SubElement(lyric, "syllabic")
        syllabic.text = "single"
        text = ET.SubElement(lyric, "text")
        text.text = f"GOMR:{note_id}"
        rec["element"].append(lyric)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def write_plugin_output(
    plugin_output: str,
    musicxml_path: str,
    xml_string: str,
    plugin_pages: list[PluginPageData],
) -> str:
    out_dir = Path(plugin_output)
    pages_dir = out_dir / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    has_prebound = any(
        isinstance(note, dict)
        for page in plugin_pages
        for note in page.notes
    )
    if has_prebound:
        xml_with_ids, notes = _finalize_prebound_plugin_notes(xml_string, plugin_pages)
    else:
        xml_with_ids, notes = _attach_note_ids_and_build_selectors(xml_string, plugin_pages)
    tagged_xml = _build_tagged_musicxml(xml_with_ids, notes)
    score_path = out_dir / "score.musicxml"
    tagged_score_path = out_dir / "score.tagged.musicxml"
    score_path.write_text(xml_with_ids, encoding="utf-8")
    tagged_score_path.write_text(tagged_xml, encoding="utf-8")
    Path(musicxml_path).write_text(xml_with_ids, encoding="utf-8")

    page_entries = []
    for page_idx, page in enumerate(plugin_pages):
        page_name = f"page_{page_idx + 1:04d}{Path(page.image_path).suffix.lower() or '.png'}"
        target = pages_dir / page_name
        shutil.copy2(page.image_path, target)
        page_entries.append({
            "pageIndex": page_idx,
            "imagePath": str(Path("pages") / page_name).replace("\\", "/"),
            "width": page.image_width,
            "height": page.image_height,
        })

    manifest = {
        "schemaVersion": 1,
        "musicxmlPath": "score.musicxml",
        "taggedMusicxmlPath": "score.tagged.musicxml",
        "sourceMusicxmlPath": str(Path(musicxml_path).resolve()),
        "pages": page_entries,
        "notesPath": "notes.json",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "notes.json").write_text(
        json.dumps({"schemaVersion": 1, "notes": notes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[Plugin] Written plugin output: {out_dir}")
    return str(score_path)


def _match_override_to_detected(override: List[str], detected: List[str],
                                xml_string: str = None) -> List[str]:
    """When override has more names than detected staves (tacet instruments),
    find the best order-preserving subsequence of override matching detected.
    Uses pitch range verification when xml_string is provided."""
    from itertools import combinations
    M, N = len(override), len(detected)
    if M <= N:
        return override[:N]

    part_pitches = _extract_part_pitches(xml_string) if xml_string else None

    detected_bases = [_instrument_base(n) for n in detected]
    candidates = []
    for indices in combinations(range(M), N):
        subset_bases = [_instrument_base(override[i]) for i in indices]
        name_score = sum(1 for a, b in zip(subset_bases, detected_bases) if a == b)
        pitch_score = 0
        pitch_bad = 0
        if part_pitches and len(part_pitches) == N:
            for j, idx in enumerate(indices):
                base = _instrument_base(override[idx])
                rng = STANDARD_RANGES.get(base)
                midis = part_pitches[j]
                if rng and midis:
                    lo, hi = rng
                    frac = sum(1 for m in midis if lo - 12 <= m <= hi + 12) / len(midis)
                    pitch_score += frac
                    if frac < 0.3:
                        pitch_bad += 1
                elif midis:
                    pitch_score += 1.0
        candidates.append((name_score, pitch_score, pitch_bad, indices))

    if part_pitches and candidates:
        candidates.sort(key=lambda c: (c[2], -c[1], -c[0]))
        best_name, best_pitch, best_bad, best_indices = candidates[0]
    else:
        candidates.sort(key=lambda c: -c[0])
        best_name, best_pitch, best_bad, best_indices = candidates[0]

    if best_name >= N * 0.3 or (part_pitches and best_pitch >= N * 0.5):
        dropped = [override[i] for i in range(M) if i not in set(best_indices)]
        matched = [override[i] for i in best_indices]
        print(f"[Override] Matched {best_name}/{N} by name, pitch={best_pitch:.1f}, bad={best_bad}, tacet: {dropped}")
        return matched
    print(f"[Override] Poor match ({best_name}/{N}), using first {N} names")
    return override[:N]


def _inject_transpose(attrs_el, instrument_name: str):
    """Insert <transpose> child into an <attributes> element if the instrument transposes."""
    base, key = _parse_instrument_key(instrument_name)
    if base not in DEFAULT_TRANSPOSE_KEY:
        return
    if key is None:
        key = DEFAULT_TRANSPOSE_KEY[base]
    tr = _compute_transpose(base, key)
    if tr is None:
        return
    diatonic, chromatic, octave = tr
    te = ET.SubElement(attrs_el, "transpose")
    ET.SubElement(te, "diatonic").text = str(diatonic)
    ET.SubElement(te, "chromatic").text = str(chromatic)
    if octave != 0:
        ET.SubElement(te, "octave-change").text = str(octave)


# ══════════════════════════════════════════════════════════════════════════════
# OCR Instrument Names
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_instrument_name(raw: str) -> str:
    """Map OCR text to a standard instrument name."""
    text = raw.strip().rstrip(".").strip()
    text = re.sub(r'(?<=\s)[!|l]{1,3}\s*$',
                  lambda m: m.group().replace('!', 'I').replace('|', 'I').replace('l', 'I'), text)
    # OCR l↔1 confusion: "C1." → "Cl.", "F1." → "Fl."
    text = re.sub(r'\b([A-Za-z])1([.\s])', r'\1l\2', text)
    text = re.sub(r'\b([A-Za-z])1$', r'\1l', text)
    # OCR ) confusion with l: "C).Es" → "Cl.Es", "C).B" → "Cl.B"
    text = re.sub(r'\b([A-Za-z])\)\.', r'\1l.', text)
    # OCR sometimes strips spaces: "BaBclarinetteinA" → "BaBclarinette in A"
    text_spaced = re.sub(r'(?i)(in)([A-Z][a-z]*)\s*$', r' \1 \2', text)
    text_no_key = re.sub(r'\s+in\s+[A-Za-z]+\s*$', '', text_spaced, flags=re.IGNORECASE).strip()

    for t in [text_spaced, text_no_key, text]:
        t_lower = t.lower().strip()
        for abbrev, full in INSTRUMENT_ABBREVS.items():
            if t_lower == abbrev or t_lower == abbrev.rstrip("."):
                return full
        best_prefix = ("", "")
        for abbrev, full in INSTRUMENT_ABBREVS.items():
            clean = abbrev.rstrip(".")
            if t_lower.startswith(clean) and len(clean) > len(best_prefix[0]):
                best_prefix = (clean, full)
        if best_prefix[1]:
            return best_prefix[1]

    text_lower = text.lower()
    best_sub = ("", "")
    for abbrev, full in INSTRUMENT_ABBREVS.items():
        clean = abbrev.rstrip(".")
        if len(clean) >= 4 and clean in text_lower and len(clean) > len(best_sub[0]):
            best_sub = (clean, full)
    if best_sub[1]:
        return best_sub[1]
    return text.title() if text else "Unknown"


def _group_staves_by_brackets(sorted_staffs, brace_dots):
    """
    Use detected braces/brackets to group staves in the first system.
    A brace/bracket that is tall and near the left edge spans multiple staves
    belonging to the same instrument family (e.g. Violin I & II under one brace).

    Returns list of lists: each inner list is indices into sorted_staffs that
    share one instrument name group.
    """
    # Filter brace_dots to those near the staff left edge (x < staff_left + 30)
    # and tall enough to span at least one staff gap
    staff_left = min(s.min_x for s in sorted_staffs)
    avg_unit = float(np.median([s.average_unit_size for s in sorted_staffs]))
    min_brace_height = avg_unit * 3

    braces = []
    for bd in brace_dots:
        cx, cy = bd.center
        w, h = bd.size
        if cx < staff_left + 30 and h > min_brace_height and w < 40:
            y_top = cy - h / 2
            y_bot = cy + h / 2
            braces.append((y_top, y_bot, cx, h))

    # Sort braces by height (ascending) so smaller sub-brackets take priority
    braces.sort(key=lambda b: b[3])

    # Exclude the system bracket (spans >60% of total y range)
    if len(sorted_staffs) >= 2:
        total_y = sorted_staffs[-1].max_y - sorted_staffs[0].min_y
        braces = [b for b in braces if b[3] < total_y * 0.6]

    # Assign each staff to brace groups
    n = len(sorted_staffs)
    staff_group = list(range(n))  # default: each staff is its own group

    for y_top, y_bot, cx, h in braces:
        members = []
        for si, staff in enumerate(sorted_staffs):
            staff_cy = (staff.min_y + staff.max_y) / 2
            if y_top <= staff_cy <= y_bot:
                members.append(si)
        if len(members) >= 2:
            gid = members[0]
            for m in members:
                staff_group[m] = gid

    # Build groups preserving order
    groups = []
    seen = set()
    for si in range(n):
        gid = staff_group[si]
        if gid not in seen:
            seen.add(gid)
            group = [i for i in range(n) if staff_group[i] == gid]
            groups.append(group)

    return groups


# ── VLM-based instrument name recognition ──

# Pass 1: free-form identification — identify what instrument each staff is
_VLM_PROMPT_PASS1 = """This is a page from an orchestral music score.
{ocr_hint}
Look at each staff (a group of 5 horizontal lines) from top to bottom.
Use the stave y-positions above to anchor each staff's location. For each staff, identify which instrument it belongs to based on the label on the left margin.
Translate German abbreviations to English (Fl.=Flute, Ob.=Oboe, Kl./Cl.=Clarinet, Bcl./Bkl.=Bass Clarinet, Fg./Fag.=Bassoon, C-Fag.=Contrabassoon, Hr.=Horn, Trp.=Trumpet, Pos.=Trombone, Pk.=Timpani, Gr.Tr.=Bass Drum, Vl.=Violin, Va./Br.=Viola, Vc./Vcll.=Cello, B./Kb.=Contrabass, Ten.Hr.=Tenor Horn).
For transposing instruments include the key only if explicitly written (e.g. "Clarinet in A"). Do NOT assume default keys.
For each staff, write one line: "Staff N (y=...): InstrumentName — reason". No markdown, no bullet points."""

# Pass 2: format pass-1 result into exactly n lines using stave positions
_VLM_PROMPT_PASS2 = """This is a page from an orchestral music score.
{ocr_hint}
A preliminary scan identified the following (may have errors in instrument names):
{pass1_result}

Using the stave y-positions above and the preliminary scan, output the FINAL instrument list.
There are exactly {n} staves. Match each stave to an instrument by its y-position. Output exactly {n} lines.

Use formal English instrument names. Append :KEY ONLY for transposing instruments when the key is explicitly shown (e.g. "Kl.(A)"→"Clarinet:A", "Cl.Es."→"Clarinet:Eb", "Trp.B"→"Trumpet:Bb"). Do NOT use : for numbering (write "Trombone" not "Trombone:1"). Violin parts: use "Violin I" / "Violin II".
- If two different labels clearly point to the SAME single staff (stacked beside one staff), join: e.g. "Trombone/Tuba".
- "B" as a KEY means Bb; "B." or "Kb." as an INSTRUMENT means Contrabass.
- CRITICAL: "in X" key labels at different y-positions apply only to the nearest staves.
- German: Fl.=Flute, Ob.=Oboe, Kl./Cl.=Clarinet, Bcl./Bkl.=Bass Clarinet, Fg./Fag.=Bassoon, C-Fag./K-Fag.=Contrabassoon, Hr./Hrn.=Horn, Trp./Trpt.=Trumpet, Pos.=Trombone, Pk.=Timpani, Gr.Tr.=Bass Drum, Beck./Bck.=Cymbals, Tamt./T.-t.=Tam-tam, Trgl.=Triangle, Ten.Hr.=Tenor Horn, Hrf./Hfe.=Harp, Cel.=Celesta, Vl.=Violin, Va./Br.=Viola, Vc./Vcl./Vcll.=Cello, B./Kb./K-B./K.B.=Contrabass.
- Output ONLY instrument names. No explanations, no numbering, no extra text.
There are exactly {n} staves. Output exactly {n} lines."""


def _ocr_margin_labels(image, staff_left: float,
                       y_start: int = 0, y_end: int = None) -> str:
    """Quick OCR scan of the left margin, returns sorted raw labels as a hint string.
    Scans the full image for better detection, then filters by y_start/y_end."""
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    x_end = max(0, int(staff_left) - 5)
    if x_end < 20:
        return ""
    crop = image[:, :x_end]
    if crop.size == 0:
        return ""
    result, _ = ocr(crop)
    if not result:
        return ""
    if y_end is None:
        y_end = image.shape[0]
    items = []
    for r in result:
        t = r[1].strip()
        if len(t) < 1:
            continue
        if re.fullmatch(r'\d+', t):
            continue
        y_center = sum(p[1] for p in r[0]) / 4
        x_center = sum(p[0] for p in r[0]) / 4
        if not (y_start <= y_center <= y_end):
            continue
        items.append((y_center, x_center, t))
    items.sort()
    if not items:
        return ""

    labels = ", ".join(f'"{t}"(x={int(x)},y={int(y)})' for y, x, t in items)
    return (f"OCR labels on left margin (x=pixels from left, y=pixels from top, sorted top-to-bottom): {labels}\n"
            f"Labels with LARGER x are closer to the staves and are per-staff markers (e.g. 'in Es', 'in B'). "
            f"Labels with SMALLER x are group/bracket labels spanning multiple staves.\n"
            f"Use these as reference for instrument names and keys.\n")


def _vlm_read_instrument_names(image_pil, n_staves: int, ocr_hint: str = "") -> list:
    """Use VLM API (Qwen3-VL-235B) with two passes to read instrument names.

    Pass 1: free-form scan (no count constraint) — identify all visible instruments.
    Pass 2: use pass-1 result as context, then format into exactly n_staves lines.
    """
    import base64, io
    try:
        import openai
    except ImportError:
        print("[VLM] openai package not installed")
        return []

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    api_key, base_url = None, None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("API_KEY="):
                    api_key = line.split("=", 1)[1]
                elif line.startswith("BASE_URL="):
                    base_url = line.split("=", 1)[1]

    if not api_key or not base_url:
        print("[VLM] No API_KEY or BASE_URL in .env")
        return []

    buf = io.BytesIO()
    image_pil.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    img_content = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}

    client = openai.OpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/v1/")

    # ── Pass 1: free-form identification ──
    p1_prompt = _VLM_PROMPT_PASS1.format(ocr_hint=ocr_hint)
    r1 = client.chat.completions.create(
        model="Qwen3-VL-235B-A22B-Instruct",
        messages=[{"role": "user", "content": [img_content, {"type": "text", "text": p1_prompt}]}],
        max_tokens=600,
        temperature=0.0,
    )
    pass1_result = r1.choices[0].message.content.strip()
    print(f"[VLM] Pass 1 raw:\n{pass1_result}")

    # ── Pass 2: format into exactly n_staves lines ──
    p2_prompt = _VLM_PROMPT_PASS2.format(
        n=n_staves, pass1_result=pass1_result, ocr_hint=ocr_hint)
    r2 = client.chat.completions.create(
        model="Qwen3-VL-235B-A22B-Instruct",
        messages=[{"role": "user", "content": [img_content, {"type": "text", "text": p2_prompt}]}],
        max_tokens=500,
        temperature=0.0,
    )
    text = r2.choices[0].message.content.strip()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines


def _vlm_read_time_sig(pil_crop) -> "tuple | None":
    """Ask VLM to read a time signature from a cropped staff image.

    The crop should span the full system height from the double barline to just
    before the first note, so the time signature digits are clearly visible.
    Returns (beats, beat_type) as ints, or None if VLM can't determine it.
    """
    import base64, io, re as _re
    try:
        import openai
    except ImportError:
        return None

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    api_key, base_url = None, None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("API_KEY="):
                    api_key = line.split("=", 1)[1]
                elif line.startswith("BASE_URL="):
                    base_url = line.split("=", 1)[1]
    if not api_key or not base_url:
        return None

    buf = io.BytesIO()
    pil_crop.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        "This image shows an orchestral score region containing a double barline. "
        "What is the time signature printed after the double barline? "
        "Reply with ONLY the fraction, e.g. '5/4', '3/2', '4/4', '6/8', '12/8'. "
        "If no time signature is visible, reply with exactly 'no'. "
        "No other text."
    )
    client = openai.OpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/v1/")
    try:
        r = client.chat.completions.create(
            model="Qwen3-VL-235B-A22B-Instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            max_tokens=20,
            temperature=0.0,
        )
        raw = r.choices[0].message.content.strip()
        print(f"[VLM-TimeSig] raw: {raw!r}")
        m = _re.search(r'(\d+)\s*/\s*(\d+)', raw)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception as e:
        print(f"[VLM-TimeSig] failed: {e}")
    return None


def _is_junk_ocr(t: str) -> bool:
    """Filter out OCR fragments that are numbers, punctuation, or too short."""
    t = t.strip()
    if len(t) < 2:
        return True
    cleaned = re.sub(r'[\d.:;,\-\s　-〿＀-｠]+', '', t)
    return len(cleaned) == 0


def _ocr_crop(ocr, image, y_start, y_end, x_end):
    """Run OCR on a crop region and return filtered text results with y-positions."""
    crop = image[y_start:y_end, 0:x_end]
    if crop.size == 0:
        return []
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if np.mean(gray_crop < 180) < 0.003:
        return []
    result, _ = ocr(crop)
    if not result:
        return []
    valid = []
    for r in result:
        t = r[1].strip()
        if _is_junk_ocr(t):
            continue
        bbox = r[0]
        y_center = sum(p[1] for p in bbox) / 4
        valid.append((y_center, t, r[2]))
    return valid


def _detect_names_for_system(sys_staves, image, brace_dots=None, use_vlm=True, coord_scale=1.0) -> List[str]:
    """Detect instrument names for a single system's staves.

    Works for any system (first, second, reduced, etc.) using the same
    OCR-hint → VLM (3 retries) → RapidOCR fallback path.
    """
    n = len(sys_staves)
    if n == 0:
        return []

    sorted_staves = sorted(sys_staves, key=lambda s: s.min_y)
    avg_unit = float(np.median([s.average_unit_size for s in sorted_staves]))
    staff_left = min(s.min_x for s in sorted_staves)

    if brace_dots:
        bracket_groups = _group_staves_by_brackets(sorted_staves, brace_dots)
    else:
        bracket_groups = [[i] for i in range(n)]

    cs = coord_scale
    print(f"[OCR] {n} staves, {len(bracket_groups)} groups, staff_left={int(staff_left)} (scale={cs:.2f})")

    # Quick OCR scan for margin labels (used as VLM hint)
    ocr_hint = ""
    if use_vlm:
        try:
            y0 = max(0, int((sorted_staves[0].min_y - avg_unit * 3) * cs))
            y1 = int((sorted_staves[-1].max_y + avg_unit * 3) * cs)
            ocr_hint = _ocr_margin_labels(image, staff_left * cs, y_start=y0, y_end=y1)
            if ocr_hint:
                print(f"[OCR→VLM] {ocr_hint.splitlines()[0][:120]}")
        except Exception as e:
            print(f"[OCR→VLM] Failed: {e}")

    if not ocr_hint:
        print(f"[OCR] No instrument labels detected, skipping VLM")
        return []

    # Try VLM (up to 3 retries)
    if use_vlm:
        try:
            from PIL import Image as PILImage
            img_y0 = max(0, int((sorted_staves[0].min_y - avg_unit * 2) * cs))
            img_y1 = int((sorted_staves[-1].max_y + avg_unit * 2) * cs)
            first_note_xs = [nn.center[0] for s in sorted_staves for nn in s.get_notes()]
            x_right = min(image.shape[1], int(min(first_note_xs) * cs)) if first_note_xs else image.shape[1]
            sys_crop = image[img_y0:img_y1, :x_right]
            pil_img = PILImage.fromarray(
                sys_crop if sys_crop.ndim == 3 else cv2.cvtColor(sys_crop, cv2.COLOR_GRAY2RGB))

            # Build per-stave OCR-label assignment for the VLM hint
            # Parse OCR hint to extract (label, y) pairs
            import re as _re
            ocr_pairs = _re.findall(r'"([^"]+)"\(x=\d+,y=(\d+)\)', ocr_hint)
            ocr_label_ys = [(txt, int(y)) for txt, y in ocr_pairs]

            stave_lines = []
            for si, stv in enumerate(sorted_staves):
                sy = int((stv.min_y + stv.max_y) / 2 * cs)
                if ocr_label_ys:
                    nearest = min(ocr_label_ys, key=lambda lbl: abs(lbl[1] - sy))
                    stave_lines.append(
                        f"  Stave {si+1} (y={sy}): nearest label '{nearest[0]}' (dist={abs(nearest[1]-sy)}px)")
                else:
                    stave_lines.append(f"  Stave {si+1} (y={sy})")
            stave_hint = (
                f"HOMR stave layout ({n} staves, crop starts at full-image y={img_y0}):\n"
                + "\n".join(stave_lines) + "\n"
                + "Use the nearest label for each stave. If multiple consecutive staves share the same nearest label, they all belong to that instrument.\n")
            vlm_hint = stave_hint + ocr_hint

            for attempt in range(3):
                vlm_lines = _vlm_read_instrument_names(pil_img, n_staves=n, ocr_hint=vlm_hint)
                print(f"[VLM] attempt {attempt+1}: {len(vlm_lines)} names: {vlm_lines}")

                if len(vlm_lines) == n:
                    known_instruments = set(INSTRUMENT_MIDI.keys())
                    valid = sum(1 for nm in vlm_lines if _instrument_base(nm) in known_instruments)
                    if valid >= n * 0.5:
                        print(f"[VLM] Direct match: {valid}/{n} known instruments")
                        return vlm_lines

                if vlm_lines:
                    labels = _match_vlm_names_to_staves(vlm_lines, bracket_groups, n)
                    if labels:
                        for i, name in enumerate(labels):
                            print(f"  Staff {i}: {name}")
                        return labels
                print(f"[VLM] attempt {attempt+1} could not match ({len(vlm_lines)} lines, need {n}), retrying...")

            print("[VLM] All retries exhausted, falling back to RapidOCR")
        except Exception as e:
            print(f"[VLM] Failed: {e}, falling back to RapidOCR")

    return _rapidocr_instrument_names(sorted_staves, image, bracket_groups, n, avg_unit, staff_left)


def ocr_instrument_names_from_staves(homr_staffs, image, brace_dots=None, use_vlm=True,
                                     coord_scale=1.0) -> List[str]:
    """Identify instrument names for the first system's staves."""
    if not homr_staffs:
        return []
    sorted_staffs = sorted(homr_staffs, key=lambda s: s.min_y)
    if brace_dots is not None:
        systems = _detect_system_breaks(sorted_staffs, brace_dots)
    else:
        systems = [list(sorted_staffs)]
    return _detect_names_for_system(systems[0], image, brace_dots, use_vlm, coord_scale=coord_scale)


def _match_vlm_names_to_staves(vlm_names, bracket_groups, n_parts):
    """Match VLM-detected names to staves using bracket group structure."""

    # Step 1: Absorb roman numeral lines into nearby instrument names
    # Pattern: "I.", "VI.", "II." → all become "Violin"
    # Rule: a roman numeral next to a known instrument name gets that name
    is_roman = [bool(re.match(r'^[IVX]+\.?$', n.rstrip('.').strip())) for n in vlm_names]
    resolved = list(vlm_names)

    for i, name in enumerate(vlm_names):
        if is_roman[i]:
            # Look at neighbors for a known instrument
            for di in [1, -1, 2, -2]:
                ni = i + di
                if 0 <= ni < len(vlm_names) and not is_roman[ni]:
                    resolved[i] = vlm_names[ni]
                    break

    # Step 2: Try exact count matches
    if len(resolved) == n_parts:
        return resolved
    if len(resolved) == len(bracket_groups):
        labels = [None] * n_parts
        for name, group in zip(resolved, bracket_groups):
            for si in group:
                labels[si] = name
        return labels

    # Step 3: Collapse consecutive duplicates → match bracket groups
    collapsed = []
    for name in resolved:
        if not collapsed or name != collapsed[-1]:
            collapsed.append(name)
    if len(collapsed) == len(bracket_groups):
        labels = [None] * n_parts
        for name, group in zip(collapsed, bracket_groups):
            for si in group:
                labels[si] = name
        return labels

    return None


def _rapidocr_instrument_names(first_system, image, bracket_groups, n_parts, avg_unit, staff_left):
    """Original RapidOCR-based instrument name detection (fallback)."""
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()

    x_end = max(0, int(staff_left) - 5)
    staff_names = {}

    # ── Pass 1: OCR each bracket group ──
    if x_end >= 20:
        for gi, group in enumerate(bracket_groups):
            first_staff = first_system[group[0]]
            last_staff = first_system[group[-1]]

            y_margin = 1.5 * avg_unit if len(group) > 1 else 3.0 * avg_unit
            y_start = max(0, int(first_staff.min_y - y_margin))
            y_end = min(image.shape[0], int(last_staff.max_y + y_margin))

            valid = _ocr_crop(ocr, image, y_start, y_end, x_end)
            if not valid:
                continue

            # For multi-staff groups: try per-staff name assignment
            if len(group) > 1 and len(valid) >= 2:
                valid.sort(key=lambda x: x[0])
                unique_names = []
                for _, t, _ in valid:
                    name = _normalize_instrument_name(t)
                    if not unique_names or name != unique_names[-1]:
                        unique_names.append(name)
                if len(unique_names) == len(group):
                    for si, name in zip(group, unique_names):
                        staff_names[si] = name
                    print(f"  Group {group}: per-staff → {unique_names}")
                    continue

            best_text = max(valid, key=lambda r: (len(r[1]), r[2]))[1]
            name = _normalize_instrument_name(best_text)
            for si in group:
                staff_names[si] = name
            print(f"  Group {group}: OCR='{best_text}' → {name}")

    # ── Pass 2: retry unlabeled or poorly-labeled staves with expanded crop ──
    known_instruments = set(INSTRUMENT_MIDI.keys())
    unlabeled = [i for i in range(n_parts) if i not in staff_names or staff_names.get(i, "") not in known_instruments]
    if unlabeled and x_end >= 20:
        consec_groups = []
        current = [unlabeled[0]]
        for i in range(1, len(unlabeled)):
            if unlabeled[i] == unlabeled[i-1] + 1:
                current.append(unlabeled[i])
            else:
                consec_groups.append(current)
                current = [unlabeled[i]]
        consec_groups.append(current)

        for cg in consec_groups:
            first_staff = first_system[cg[0]]
            last_staff = first_system[cg[-1]]
            y_margin = 5.0 * avg_unit
            y_start = max(0, int(first_staff.min_y - y_margin))
            y_end = min(image.shape[0], int(last_staff.max_y + y_margin))

            valid = _ocr_crop(ocr, image, y_start, y_end, x_end)
            if not valid:
                continue

            # Prefer texts that normalize to known instrument names
            candidates = [(v, _normalize_instrument_name(v[1])) for v in valid]
            known = [(v, nm) for v, nm in candidates if nm in known_instruments]
            if known:
                best_text = max(known, key=lambda x: (len(x[0][1]), x[0][2]))[0][1]
                name = max(known, key=lambda x: (len(x[0][1]), x[0][2]))[1]
            else:
                best_text = max(valid, key=lambda r: (len(r[1]), r[2]))[1]
                name = _normalize_instrument_name(best_text)
            for si in cg:
                staff_names[si] = name
            print(f"  Group {cg} (pass2): OCR='{best_text}' → {name}")

    labels = [staff_names.get(si, f"Part {si+1}") for si in range(n_parts)]
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Tremolo detection via template matching
# ══════════════════════════════════════════════════════════════════════════════

def _load_tremolo_templates(template_dir):
    """Load tremolo_tight_*.png templates from directory."""
    import glob as _glob
    paths = sorted(_glob.glob(os.path.join(template_dir, "tremolo_tight_*.png")))
    templates = []
    for p in paths:
        idx = int(os.path.basename(p).split("_")[-1].split(".")[0])
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates.append({"idx": idx, "img": img, "h": img.shape[0], "w": img.shape[1]})
    return templates


def detect_tremolo(full_image, template_dir, threshold=0.75):
    """Detect tremolo marks via template matching on the full-res image.
    Returns list of (cx, cy, w, h, score) in full-res pixel coordinates."""
    templates = _load_tremolo_templates(template_dir)
    if not templates:
        return []

    if len(full_image.shape) == 3:
        gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = full_image

    scales = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    all_dets = []
    for tmpl in templates:
        for scale in scales:
            tw = max(5, int(tmpl["w"] * scale))
            th = max(5, int(tmpl["h"] * scale))
            if th > gray.shape[0] or tw > gray.shape[1]:
                continue
            resized = cv2.resize(tmpl["img"], (tw, th), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            locs = np.where(result >= threshold)
            for py, px in zip(*locs):
                all_dets.append((px, py, tw, th, float(result[py, px])))

    # NMS
    if not all_dets:
        return []
    boxes = np.array([[d[0], d[1], d[0]+d[2], d[1]+d[3]] for d in all_dets])
    scores = np.array([d[4] for d in all_dets])
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0)
        order = order[np.where(iou <= 0.3)[0] + 1]
    dets = [all_dets[i] for i in keep]

    results = []
    for d in dets:
        cx = d[0] + d[2] / 2.0
        cy = d[1] + d[3] / 2.0
        results.append((cx, cy, d[2], d[3], d[4]))
    return results


def match_tremolo_to_noteheads(tremolo_dets, noteheads, staffs_sorted, coord_scale):
    """Match each tremolo detection to nearest notehead within a 3x3 grid.
    tremolo_dets: list of (cx, cy, w, h, score) in full-res coords.
    noteheads: list of BoundingEllipse in HOMR-resized coords.
    coord_scale: full_res / homr_res ratio.
    Returns list of (staff_index, notehead_cx_homr, notehead_cy_homr, tremolo_score)."""
    results = []
    for tcx, tcy, tw, th, tscore in tremolo_dets:
        # 3x3 grid around tremolo box: notehead center must be within 1.5 * box dimension
        search_rx = tw * 1.5
        search_ry = th * 1.5

        best_nh = None
        best_dist = float("inf")
        for nh in noteheads:
            # notehead center is in HOMR space, scale to full-res
            nhx = nh.center[0] * coord_scale
            nhy = nh.center[1] * coord_scale
            dx = abs(nhx - tcx)
            dy = abs(nhy - tcy)
            if dx > search_rx or dy > search_ry:
                continue
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_nh = nh

        if best_nh is None:
            continue

        # Find which staff this notehead belongs to
        nh_cy = best_nh.center[1]  # HOMR space
        best_si, best_sdist = -1, float("inf")
        for si, staff in enumerate(staffs_sorted):
            staff_cy = (staff.min_y + staff.max_y) / 2
            sdist = abs(nh_cy - staff_cy)
            if sdist < best_sdist:
                best_sdist = sdist
                best_si = si

        if best_si >= 0:
            staff = staffs_sorted[best_si]
            staff_margin = 4 * staff.average_unit_size
            if nh_cy < staff.min_y - staff_margin or nh_cy > staff.max_y + staff_margin:
                continue
            results.append((best_si, best_nh.center[0], best_nh.center[1], tscore))
    return results


def inject_tremolo(result_staffs, matched_tremolo, staffs_sorted, part_count: int | None = None):
    """Inject tremolo articulation into the nearest note EncodedSymbol.
    matched_tremolo: list of (staff_index, nh_cx_homr, nh_cy_homr, score).
    Uses notehead position in HOMR space → canvas space → match to EncodedSymbol."""
    from collections import defaultdict
    import math

    by_staff = defaultdict(list)
    for si, nhx, nhy, score in matched_tremolo:
        by_staff[si].append((nhx, nhy, score))

    n_injected = 0
    if part_count is None:
        part_count = len(result_staffs)

    for si, det_list in by_staff.items():
        part_idx = si % part_count
        system_idx = si // part_count
        if part_idx >= len(result_staffs):
            continue
        staff = staffs_sorted[si]
        segments = _split_symbols_by_newline(result_staffs[part_idx])
        if system_idx >= len(segments):
            continue
        symbols = segments[system_idx]

        unit = staff.average_unit_size
        region_x_min = staff.min_x - 2 * unit
        region_x_max = staff.max_x + 2 * unit
        region_w = region_x_max - region_x_min

        canvas_w = 1280.0
        scale = canvas_w / region_w

        for nhx, nhy, score in det_list:
            canvas_x = (nhx - region_x_min) * scale

            best_idx, best_dist = -1, float("inf")
            for idx, symbol_item in enumerate(symbols):
                if isinstance(symbol_item, tuple):
                    _token_index, sym = symbol_item
                else:
                    sym = symbol_item
                if not sym.rhythm.startswith("note_"):
                    continue
                if sym.coordinates is None:
                    continue
                if math.isnan(sym.coordinates[0]):
                    continue
                dist = abs(sym.coordinates[0] - canvas_x)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            tolerance = 60.0
            if best_idx >= 0 and best_dist < tolerance:
                sym = symbols[best_idx]
                if "tremolo" not in sym.articulation:
                    if sym.articulation == "." or sym.articulation == "":
                        sym.articulation = "tremolo"
                    else:
                        sym.articulation = sym.articulation + "_tremolo"
                    n_injected += 1

    return n_injected


# ══════════════════════════════════════════════════════════════════════════════
# Dynamics detection via YOLO (ft-15ep on DeepScoresV2)
# ══════════════════════════════════════════════════════════════════════════════

_YOLO_DYN_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "runs_dynamics", "dynamics_finetune_15ep", "weights", "best.pt",
)

# f/p: precision-first (conf at P=0.95); s: recall-first (F1-optimal)
_YOLO_DYN_CONF = {"dynamicF": 0.65, "dynamicP": 0.65, "dynamicS": 0.31}
_YOLO_DYN_CONF_DEFAULT = 0.25

# single-letter per YOLO class; compound symbols assembled by _group_glyphs
_YOLO_CLASS_TO_LETTER = {
    "dynamicF": "f", "dynamicP": "p", "dynamicS": "s",
    "dynamicM": "m", "dynamicR": "r", "dynamicZ": "z",
}

# valid MusicXML <dynamics> child element names, normalised to f/p only:
#   ff/sf/mf/fz/rf → f   |   pp/mp → p
#   ≥3 letters (fff, ppp, sfz, rfz, sffz…) → absent → discarded
#   mixed f+p (fp, pf, sfp…) → absent → discarded
_COMPOUND_DYN = {
    "f":  "f",
    "ff": "f",
    "sf": "f",
    "mf": "f",
    "fz": "f",
    "rf": "f",
    "p":  "p",
    "pp": "p",
    "mp": "p",
}

_yolo_dyn_model = None

# expression words that are NOT dynamics (OCR filter)
_EXPR_WORDS = {
    "cresc", "decresc", "dim", "dimin", "poco", "molto", "sempre",
    "dolce", "legg", "leggiero", "arco", "pizz", "spicc", "sul",
    "piu", "meno", "assai", "subito", "marc", "espress", "tranq",
    "riten", "rit", "accel", "rubato", "ten", "sosт", "calando",
    "morendo", "smorzando", "perdendosi", "col", "con", "div", "unis",
}


def _get_yolo_dyn_model():
    global _yolo_dyn_model
    if _yolo_dyn_model is None:
        from ultralytics import YOLO
        _yolo_dyn_model = YOLO(_YOLO_DYN_WEIGHTS)
    return _yolo_dyn_model


def _group_glyphs(raw):
    """Group adjacent glyphs into compound dynamics.

    raw: list of (letter, cx, cy, bw, bh)
    Returns list of (combined_str, cx, cy).
    """
    if not raw:
        return []
    raw = sorted(raw, key=lambda d: (d[2], d[1]))  # sort by y then x
    used = [False] * len(raw)
    groups = []
    for i, (li, cxi, cyi, bwi, bhi) in enumerate(raw):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(raw)):
            if used[j]:
                continue
            lj, cxj, cyj, bwj, bhj = raw[j]
            if abs(cyj - cyi) > max(bhi, bhj) * 0.6:
                break  # sorted by y, no more close rows
            if abs(cxj - cxi) < (bwi + bwj) * 1.1:
                group.append(j)
                used[j] = True
        group.sort(key=lambda k: raw[k][1])  # sort by x
        combined = "".join(raw[k][0] for k in group)
        gcx = sum(raw[k][1] for k in group) / len(group)
        gcy = sum(raw[k][2] for k in group) / len(group)
        groups.append((combined, gcx, gcy))
    return groups


def _ocr_expression_regions(img_bgr):
    """Return list of (x1,y1,x2,y2) for OCR-detected expression text regions.

    Splits image into horizontal strips so small text is not downscaled.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    except ImportError:
        return []

    h = img_bgr.shape[0]
    strip_h, overlap = 400, 50
    regions = []
    y = 0
    while y < h:
        y2 = min(y + strip_h, h)
        if y2 - y < 20:  # RapidOCR rounds dims to nearest multiple of 32; a strip this
            break        # short rounds to 0 after scaling (e.g. h=8 → scaled 6 → 0)
        strip = img_bgr[y:y2, :]
        result, _ = ocr(strip)
        if result:
            for line in result:
                box, text, conf = line
                words = re.split(r"[\s.,:;]+", text.lower())
                if any(w in _EXPR_WORDS for w in words if w):
                    xs = [p[0] for p in box]
                    ys = [p[1] + y for p in box]
                    pad = 20
                    regions.append((min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad))
        y += strip_h - overlap
    return regions


def detect_dynamics(staffs_sorted, img_path, bar_line_boxes, homr_shape):
    """Detect dynamics markings via YOLO, map to (staff_idx, measure_1based, note_idx, dyn_type)."""
    full_image = cv2.imread(img_path)
    if full_image is None:
        return []
    full_h, full_w = full_image.shape[:2]
    homr_h, homr_w = homr_shape[:2]
    coord_scale = full_w / homr_w

    expr_regions = _ocr_expression_regions(full_image)

    model = _get_yolo_dyn_model()
    yolo_res = model(img_path, imgsz=1280, verbose=False)[0]

    # Pass 1: collect valid single-glyph detections
    raw_conf = []  # (letter, cx, cy, bw, bh, conf)
    for box in yolo_res.boxes:
        cls_name = yolo_res.names[int(box.cls)]
        letter = _YOLO_CLASS_TO_LETTER.get(cls_name)
        if letter is None:
            continue
        conf = float(box.conf)
        if conf < _YOLO_DYN_CONF.get(cls_name, _YOLO_DYN_CONF_DEFAULT):
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if any(ex1 <= cx <= ex2 and ey1 <= cy <= ey2 for ex1, ey1, ex2, ey2 in expr_regions):
            continue
        raw_conf.append((letter, cx, cy, x2 - x1, y2 - y1, conf))

    # Spatial NMS: remove near-duplicate same-letter detections (keep highest conf)
    raw_conf.sort(key=lambda d: -d[5])
    used = [False] * len(raw_conf)
    raw = []
    for i, (li, cxi, cyi, bwi, bhi, _) in enumerate(raw_conf):
        if used[i]:
            continue
        raw.append((li, cxi, cyi, bwi, bhi))
        for j in range(i + 1, len(raw_conf)):
            if not used[j] and raw_conf[j][0] == li:
                if abs(raw_conf[j][1] - cxi) < bwi * 0.8 and abs(raw_conf[j][2] - cyi) < bhi * 0.8:
                    used[j] = True

    # Pass 2: group adjacent glyphs, map to MusicXML
    results = []
    for combined, cx, cy in _group_glyphs(raw):
        dyn_type = _COMPOUND_DYN.get(combined)
        if dyn_type is None:
            continue
        cx_homr = cx / coord_scale
        cy_homr = cy / coord_scale

        # Find the nearest staff above this detection
        si = None
        best_gap = float("inf")
        for i, staff in enumerate(staffs_sorted):
            if cy_homr <= staff.max_y:
                continue
            gap = cy_homr - staff.max_y
            if gap < best_gap:
                best_gap = gap
                si = i
        if si is None:
            continue

        staff = staffs_sorted[si]
        unit = staff.average_unit_size
        staff_barlines = sorted(
            bl.center[0] for bl in bar_line_boxes
            if staff.min_y - unit <= bl.center[1] <= staff.max_y + unit
        )
        measure = 1
        for bi, bx in enumerate(staff_barlines):
            if cx_homr > bx:
                measure = bi + 2

        measure_edges = [staff.min_x] + staff_barlines + [staff.max_x]
        note_xs = sorted(n.center[0] for n in staff.get_notes())
        m_lo = measure_edges[measure - 1] if measure - 1 < len(measure_edges) else staff.min_x
        m_hi = measure_edges[measure] if measure < len(measure_edges) else staff.max_x
        notes_in_m = [nx for nx in note_xs if m_lo <= nx <= m_hi]
        note_idx = 0
        if notes_in_m:
            note_idx = min(range(len(notes_in_m)), key=lambda i: abs(notes_in_m[i] - cx_homr))

        results.append((si, measure, note_idx, dyn_type))

    return results



def detect_double_barlines(staff, bar_line_boxes):
    """Find double barline x-positions for a given staff from raw barline boxes.
    bar_line_boxes: list of RotatedBoundingBox from HOMR barline detection.
    Returns list of x-coordinates (HOMR space) for each double barline found.
    Detects both: (a) two separate close boxes, (b) single wide merged box."""
    unit = staff.average_unit_size
    margin = unit * 2
    staff_bls = []
    for bl in bar_line_boxes:
        cy = bl.center[1]
        cx = bl.center[0]
        if (staff.min_y - margin <= cy <= staff.max_y + margin
                and staff.min_x <= cx <= staff.max_x):
            staff_bls.append(bl)

    if not staff_bls:
        return []
    staff_bls.sort(key=lambda b: b.center[0])
    double_barline_xs = []
    used = set()

    # (a) Two separate close barline boxes
    for i in range(len(staff_bls) - 1):
        if i in used:
            continue
        b1 = staff_bls[i]
        b2 = staff_bls[i + 1]
        dx = abs(b2.center[0] - b1.center[0])
        if dx < unit * 1.5:
            right_x = max(b1.center[0] + b1.size[0] / 2,
                          b2.center[0] + b2.size[0] / 2)
            double_barline_xs.append(right_x)
            used.add(i)
            used.add(i + 1)

    # (b) Single wide barline (merged double barline)
    # Normal single barlines are 3-5px; double barlines merged are 8-12px
    width_threshold = max(7, unit * 0.9)
    for i, bl in enumerate(staff_bls):
        if i in used:
            continue
        if bl.size[0] >= width_threshold:
            right_x = bl.center[0] + bl.size[0] / 2
            double_barline_xs.append(right_x)
            used.add(i)

    return sorted(double_barline_xs)


# ══════════════════════════════════════════════════════════════════════════════
# HOMR pipeline (all GPU)
# ══════════════════════════════════════════════════════════════════════════════


def _detect_system_breaks(staffs_sorted, brace_dots):
    """Split sorted staves into system groups using large bracket detection.
    Each system starts with a tall bracket/brace on the left that spans all
    its staves.  We find these tall bounding-boxes and assign staves to them."""
    if len(staffs_sorted) <= 1:
        return [list(staffs_sorted)]

    avg_staff_h = float(np.median([s.max_y - s.min_y for s in staffs_sorted]))
    min_bracket_h = avg_staff_h * 4

    candidates = []
    for bd in brace_dots:
        h = bd.size[1]
        w = max(bd.size[0], 1)
        if h > min_bracket_h and h / w > 5:
            y_top = bd.center[1] - h / 2
            y_bot = bd.center[1] + h / 2
            candidates.append((y_top, y_bot))

    if not candidates:
        return [list(staffs_sorted)]

    candidates.sort(key=lambda b: b[0])

    merged = [list(candidates[0])]
    for top, bot in candidates[1:]:
        if top < merged[-1][1]:
            merged[-1][0] = min(merged[-1][0], top)
            merged[-1][1] = max(merged[-1][1], bot)
        else:
            merged.append([top, bot])

    if len(merged) <= 1:
        return [list(staffs_sorted)]

    systems = [[] for _ in merged]
    margin = avg_staff_h
    for staff in staffs_sorted:
        cy = (staff.min_y + staff.max_y) / 2
        assigned = False
        for bi, (top, bot) in enumerate(merged):
            if top - margin <= cy <= bot + margin:
                systems[bi].append(staff)
                assigned = True
                break
        if not assigned:
            best = min(range(len(merged)),
                       key=lambda i: abs(cy - (merged[i][0] + merged[i][1]) / 2))
            systems[best].append(staff)

    systems = [s for s in systems if s]
    return systems if systems else [list(staffs_sorted)]


_VLM_EXTRA_SYSTEM_PROMPT = """This is a CROPPED region from an orchestral music score, showing the LEFT MARGIN of one system (行).
Read each instrument name or abbreviation from top to bottom.
Map each to one of these standard names: {master_names}
There are exactly {n} staves in this region. Output exactly {n} lines, one standard name per staff, from top to bottom. No numbering, no extra text.
German abbreviations: Fl.=Flute, Ob.=Oboe, Kl./Cl.=Clarinet, Fg./Fag.=Bassoon, C-Fag.=Contrabassoon, Hr./Hrn.=Horn, Trp.=Trumpet, Pos.=Trombone, Pk.=Timpani, Gr.Tr.=Bass Drum, Hrf./Hfe.=Harp, Cel.=Celesta, Vl.=Violin, Va./Br.=Viola, Vc.=Cello, B./Kb.=Contrabass."""


def _ocr_extra_system_names(sys_staves, image, master_names, use_vlm=True):
    """Detect instrument names for a non-first system via VLM on left-margin crop."""
    n = len(sys_staves)
    if n == 0:
        return []
    if not use_vlm:
        return [f"Part {i + 1}" for i in range(n)]

    fallback_names = [f"Part {i + 1}" for i in range(n)]
    try:
        import base64, io
        from PIL import Image as PILImage
        import openai

        top = max(0, int(sys_staves[0].min_y - 80))
        bottom = min(image.shape[0], int(sys_staves[-1].max_y + 80))
        left_end = int(min(s.min_x for s in sys_staves))
        crop = image[top:bottom, 0:left_end]
        if crop.size == 0:
            raise ValueError("empty crop")

        pil_crop = PILImage.fromarray(
            crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB))

        buf = io.BytesIO()
        pil_crop.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        api_key, base_url = None, None
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("API_KEY="):
                        api_key = line.split("=", 1)[1]
                    elif line.startswith("BASE_URL="):
                        base_url = line.split("=", 1)[1]

        if not api_key or not base_url:
            raise ValueError("no VLM credentials")

        unique_master = sorted(set(master_names))
        prompt = _VLM_EXTRA_SYSTEM_PROMPT.format(
            master_names=", ".join(unique_master), n=n)

        client = openai.OpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/v1/")
        response = client.chat.completions.create(
            model="Qwen3-VL-235B-A22B-Instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            max_tokens=300, temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        print(f"[VLM-extra] System names ({len(lines)}): {lines}")

        master_bases = set(_instrument_base(nm) for nm in master_names)
        if len(lines) == n:
            valid = sum(1 for nm in lines if _instrument_base(nm) in master_bases)
            if valid >= n * 0.5:
                return lines
        # VLM count mismatch — try expanding via grand staff pair detection
        if len(lines) == n - 1 and len(lines) > 0:
            gaps = []
            for gi in range(1, n):
                gaps.append(sys_staves[gi].min_y - sys_staves[gi - 1].max_y)
            min_gap_idx = int(np.argmin(gaps))
            expanded = lines[:min_gap_idx + 1] + [lines[min_gap_idx]] + lines[min_gap_idx + 1:]
            valid = sum(1 for nm in expanded if _instrument_base(nm) in master_bases)
            if valid >= n * 0.5:
                print(f"[VLM-extra] Expanded via grand staff at staves {min_gap_idx}/{min_gap_idx + 1}: {expanded}")
                return expanded
        if len(lines) > 0:
            valid_lines = [nm for nm in lines if _instrument_base(nm) in master_bases]
            if len(valid_lines) >= n * 0.5:
                result = list(lines[:n]) if len(lines) >= n else lines + [lines[-1]] * (n - len(lines))
                print(f"[VLM-extra] Best-effort mapping: {result}")
                return result
        print(f"[VLM-extra] Could not match {len(lines)} names to {n} staves")
    except Exception as e:
        print(f"[VLM-extra] Failed: {e}")

    try:
        avg_unit = float(np.median([s.average_unit_size for s in sys_staves]))
        staff_left = min(s.min_x for s in sys_staves)
        bracket_groups = [[i] for i in range(n)]
        ocr_names = _rapidocr_instrument_names(
            sys_staves, image, bracket_groups, n, avg_unit, staff_left)
        master_bases = set(_instrument_base(nm) for nm in master_names)
        valid = sum(1 for nm in ocr_names if _instrument_base(nm) in master_bases)
        if valid >= max(1, n * 0.4):
            print(f"[OCR-extra] System names ({valid}/{n} matched master): {ocr_names}")
            return ocr_names
        print(f"[OCR-extra] Weak match ({valid}/{n}), using generic names")
    except Exception as e:
        print(f"[OCR-extra] Failed: {e}")

    return fallback_names



def run_homr_pipeline(img_path: str, use_gpu: bool = True, use_vlm: bool = True,
                      tremolo_templates: str = None,
                      part_names_override: List[str] = None,
                      collect_plugin_data: bool = False,
                      page_index: int = 0):
    """
    Run HOMR's complete pipeline with automatic system grouping override
    for orchestral scores. Returns list of (MusicXML string, part names) tuples.
    Normally returns a single-element list; multi-element for pages with
    systems of different staff counts (e.g. reduced orchestration).
    """
    from homr.main import (
        load_and_preprocess_predictions, predict_symbols,
        download_weights,
    )
    from homr.staff_parsing import parse_staffs
    from homr.staff_detection import detect_staff, break_wide_fragments
    from homr.note_detection import add_notes_to_staffs, combine_noteheads_with_stems
    from homr.bar_line_detection import detect_bar_lines, prepare_bar_line_image
    from homr.brace_dot_detection import (
        find_braces_brackets_and_grand_staff_lines, prepare_brace_dot_image,
    )
    from homr.bounding_boxes import create_rotated_bounding_boxes
    from homr.model import MultiStaff
    from grandomr_music_xml_generator import generate_xml, XmlGeneratorArguments
    from homr.transformer.configs import Config
    from homr.title_detection import detect_title

    download_weights(use_gpu)

    t0 = time.time()
    print("[HOMR] Preprocessing + segmentation...")
    predictions, debug = load_and_preprocess_predictions(img_path, False, False, use_gpu)
    t1 = time.time()
    print(f"[HOMR] Segmentation done ({t1-t0:.1f}s)")

    symbols = predict_symbols(debug, predictions)
    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)

    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    if not noteheads_with_stems:
        raise RuntimeError("No noteheads found in image")

    avg_nh = float(np.median([n.notehead.size[1] for n in noteheads_with_stems]))
    all_noteheads = [n.notehead for n in noteheads_with_stems]
    all_stems = [n.stem for n in noteheads_with_stems if n.stem is not None]
    bar_lines_or_rests = [
        l for l in symbols.bar_lines
        if not l.is_overlapping_with_any(all_noteheads)
        and not l.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, avg_nh)

    staffs = detect_staff(
        debug, predictions.staff, symbols.staff_fragments,
        symbols.clefs_keys, bar_line_boxes,
    )
    if not staffs:
        raise RuntimeError("No staves detected")

    title_future = detect_title(debug, staffs[0])

    add_notes_to_staffs(
        staffs, noteheads_with_stems, predictions.symbols, predictions.notehead,
    )

    # Filter out narrow staff fragments that would crash TrOMR resize
    if len(staffs) > 1:
        widths = [s.max_x - s.min_x for s in staffs]
        median_w = float(np.median(widths))
        min_w = median_w * 0.2
        good_staffs = [s for s in staffs if (s.max_x - s.min_x) >= min_w]
        if len(good_staffs) < len(staffs):
            print(f"[HOMR] Filtered {len(staffs) - len(good_staffs)} narrow staff fragment(s)")
            staffs = good_staffs

    t2 = time.time()
    print(f"[HOMR] Staff detection done: {len(staffs)} staves ({t2-t1:.1f}s)")

    # ── Detect braces/brackets for grouping ──
    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    brace_dots = create_rotated_bounding_boxes(
        brace_dot_img, skip_merging=True, max_size=(100, -1),
    )

    # ── OCR instrument names using bracket groups ──
    # Use full-res image for VLM crop (HOMR downscales internally; full-res has sharper text)
    _vlm_image = cv2.imread(img_path)
    if _vlm_image is not None and predictions.original.shape[1] > 0:
        _vlm_scale = _vlm_image.shape[1] / predictions.original.shape[1]
    else:
        _vlm_image = predictions.original
        _vlm_scale = 1.0
    part_names = ocr_instrument_names_from_staves(
        staffs, _vlm_image, brace_dots, use_vlm=use_vlm, coord_scale=_vlm_scale)

    # If OCR/VLM returned nothing, use override as part_names for correct grouping
    if not part_names and part_names_override is not None:
        part_names = list(part_names_override)
        print(f"[Override] No labels on this page, reusing: {len(part_names)} instruments")

    # ── Determine parts-per-system and group staves ──
    n_parts = len(part_names)
    staffs_sorted = sorted(staffs, key=lambda s: s.min_y)
    system_groups = _detect_system_breaks(staffs_sorted, brace_dots)
    sys_sizes = [len(g) for g in system_groups]

    # Merge adjacent small systems that together equal the first system's staff count
    if len(system_groups) > 1:
        target = len(system_groups[0])
        merged = [system_groups[0]]
        i = 1
        while i < len(system_groups):
            acc = list(system_groups[i])
            i += 1
            while len(acc) < target and i < len(system_groups):
                acc.extend(system_groups[i])
                i += 1
            if len(acc) == target:
                merged.append(acc)
            else:
                merged.append(acc)
        if [len(g) for g in merged] != sys_sizes:
            print(f"[HOMR] Merged split systems: {sys_sizes} → {[len(g) for g in merged]}")
            system_groups = merged
            sys_sizes = [len(g) for g in system_groups]

    if not part_names and sys_sizes:
        part_names = [f"Part {i + 1}" for i in range(sys_sizes[0])]
        n_parts = len(part_names)
        print(f"[HOMR] No part names after OCR/VLM/override; using generic names: {n_parts} parts")

    print(f"[HOMR] Detected {len(system_groups)} system(s): {sys_sizes} staves each")

    # Remove duplicate staves within each system (same y-range detected twice by SegNet)
    deduped_groups = []
    for grp in system_groups:
        seen, deduped = [], []
        for s in grp:
            key = (round(s.min_y / 5), round(s.max_y / 5))  # 5px tolerance
            if key not in seen:
                seen.append(key)
                deduped.append(s)
            else:
                print(f"[HOMR] Removed duplicate stave at y={int(s.min_y)}-{int(s.max_y)}")
        deduped_groups.append(deduped)
    if [len(g) for g in deduped_groups] != sys_sizes:
        system_groups = deduped_groups
        sys_sizes = [len(g) for g in system_groups]
        print(f"[HOMR] After dedup: {sys_sizes} staves each")

    first_sys_count = sys_sizes[0]
    all_same = all(sz == first_sys_count for sz in sys_sizes)
    multi_system_mode = (n_parts > 0 and first_sys_count == n_parts
                         and not all_same)

    if multi_system_mode:
        # ── MULTI-SYSTEM with different staff counts ──
        print(f"[HOMR] Multi-system page: {sys_sizes}")
        median_w = float(np.median([s.max_x - s.min_x for s in staffs_sorted]))
        for gi, grp in enumerate(system_groups):
            parts = []
            for si, s in enumerate(grp):
                w = int(s.max_x - s.min_x)
                gap = int(s.min_y - grp[si-1].max_y) if si > 0 else 0
                parts.append(f"s{si}:y={int(s.min_y)}-{int(s.max_y)} w={w}px gap={gap}px")
            print(f"  sys{gi}: " + ", ".join(parts))

        transformer_config = Config()
        transformer_config.use_gpu_inference = use_gpu
        try:
            title = title_future.result(60)
        except Exception:
            title = Path(img_path).stem

        tremolo_dets = []
        coord_scale = 1.0
        if tremolo_templates and os.path.isdir(tremolo_templates):
            full_image = cv2.imread(img_path)
            homr_h, homr_w = predictions.original.shape[:2]
            coord_scale = full_image.shape[1] / homr_w
            tremolo_dets = detect_tremolo(full_image, tremolo_templates, threshold=0.75)

        results = []
        for sys_idx, sys_staves in enumerate(system_groups):
            t_sys = time.time()
            print(f"\n[HOMR] Processing system {sys_idx + 1}/{len(system_groups)} "
                  f"({len(sys_staves)} staves)")

            sys_multi = [MultiStaff(sys_staves, [])]
            sys_result = parse_staffs(
                debug, sys_multi, predictions.preprocessed,
                selected_staff=-1, config=transformer_config,
            )
            n_sym = sum(len(s) for s in sys_result)
            print(f"[HOMR] System {sys_idx + 1} TrOMR: {len(sys_result)} parts, "
                  f"{n_sym} symbols ({time.time() - t_sys:.1f}s)")

            if tremolo_dets:
                matched = match_tremolo_to_noteheads(
                    tremolo_dets, all_noteheads, sys_staves, coord_scale)
                if matched:
                    n_inj = inject_tremolo(sys_result, matched, sys_staves, part_count=len(sys_result))
                    print(f"[Tremolo] System {sys_idx + 1}: {n_inj} injected")

            n_ks = correct_key_signatures(
                sys_result, sys_staves, predictions.original,
                bar_line_boxes, full_res_image=full_image_ks)
            if n_ks:
                print(f"[KeySig] System {sys_idx + 1}: {n_ks} corrected")

            writer_notes = []
            if collect_plugin_data:
                _tag_symbols_for_plugin(sys_result)

            if sys_idx == 0:
                sys_names = part_names
            else:
                sys_names = _detect_names_for_system(
                    sys_staves, _vlm_image, brace_dots, use_vlm=use_vlm, coord_scale=_vlm_scale)
                ref_names = part_names_override if part_names_override else part_names
                if not sys_names and ref_names:
                    sys_names = list(ref_names)
                    print(f"[Override] System {sys_idx+1}: no labels, reusing: {len(sys_names)} instruments")
                elif ref_names and len(ref_names) > len(sys_names):
                    sys_names = _match_override_to_detected(ref_names, sys_names)

            xml_args = XmlGeneratorArguments(
                note_callback=_make_plugin_note_callback(writer_notes)
                if collect_plugin_data else None
            )
            xml_root = generate_xml(xml_args, sys_result, title)
            xml_string = xml_root.to_string()

            sys_dynamics = detect_dynamics(sys_staves, img_path, bar_line_boxes, predictions.original.shape)
            if sys_dynamics:
                xml_string = _inject_dynamics(xml_string, sys_dynamics)
                print(f"[Dynamics] System {sys_idx + 1}: {len(sys_dynamics)} marking(s)")

            plugin_page = None
            if collect_plugin_data:
                plugin_page = _build_plugin_page_data_for_system(
                    img_path=img_path,
                    page_index=page_index,
                    result_staffs=sys_result,
                    sys_staves=sys_staves,
                    predictions_original_shape=predictions.original.shape,
                )
                id_prefix = f"grandomr-p{page_index + 1:04d}s{sys_idx + 1:03d}"
                xml_string, prebound_notes = _prebind_plugin_notes_from_writer_map(
                    xml_string,
                    plugin_page,
                    writer_notes,
                    id_prefix=id_prefix,
                )
                plugin_page.notes = prebound_notes
                print(f"[Plugin] System {sys_idx + 1}: mapped {len(prebound_notes)} note box(es)")

            results.append((xml_string, sys_names, plugin_page))

        if collect_plugin_data:
            if not any(plugin_page is not None for _xml, _names, plugin_page in results):
                plugin_page = _build_plugin_page_image_only(img_path, page_index)
                print("[Plugin] Multi-system page detected; no note bbox mapping was produced")
                if results:
                    xml_string, sys_names, _ = results[0]
                    results[0] = (xml_string, sys_names, plugin_page)
        return results

    # ── Normal path: uniform system sizes ──
    if n_parts > 0 and len(staffs_sorted) >= n_parts and all_same:
        multi_staffs = [MultiStaff(g, []) for g in system_groups]
        print(f"[HOMR] Grouped: {n_parts} parts × {len(system_groups)} systems")
    elif n_parts > 0 and all_same and first_sys_count != n_parts:
        multi_staffs = [MultiStaff(g, []) for g in system_groups]
        n_parts = first_sys_count
        if len(part_names) > n_parts:
            part_names = part_names[:n_parts]
        else:
            part_names = part_names + [f"Part {i + 1}" for i in range(len(part_names), n_parts)]
        print(f"[HOMR] Adjusted: {n_parts} parts × {len(system_groups)} systems")
    else:
        brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
        brace_dot = create_rotated_bounding_boxes(
            brace_dot_img, skip_merging=True, max_size=(100, -1),
        )
        multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)
        print(f"[HOMR] Auto-grouped: {[len(ms.staffs) for ms in multi_staffs]}")
        n_parts = len(multi_staffs[0].staffs) if multi_staffs else 0

    t3 = time.time()
    transformer_config = Config()
    transformer_config.use_gpu_inference = use_gpu

    result_staffs = parse_staffs(
        debug, multi_staffs, predictions.preprocessed,
        selected_staff=-1, config=transformer_config,
    )

    try:
        title = title_future.result(60)
    except Exception:
        title = Path(img_path).stem

    t4 = time.time()
    n_result = len(result_staffs)
    n_symbols = sum(len(s) for s in result_staffs)
    print(f"[HOMR] TrOMR done: {n_result} parts, {n_symbols} symbols ({t4 - t3:.1f}s)")

    if tremolo_templates and os.path.isdir(tremolo_templates):
        t_tr = time.time()
        full_image = cv2.imread(img_path)
        homr_h, homr_w = predictions.original.shape[:2]
        full_h, full_w = full_image.shape[:2]
        coord_scale = full_w / homr_w

        tremolo_dets = detect_tremolo(full_image, tremolo_templates, threshold=0.75)
        if tremolo_dets:
            matched = match_tremolo_to_noteheads(
                tremolo_dets, all_noteheads, staffs_sorted, coord_scale,
            )
            n_inj = inject_tremolo(result_staffs, matched, staffs_sorted, part_count=len(result_staffs))
            print(f"[Tremolo] {len(tremolo_dets)} detections, {len(matched)} matched to noteheads, "
                  f"{n_inj} injected ({time.time() - t_tr:.1f}s)")
        else:
            print(f"[Tremolo] 0 detections ({time.time() - t_tr:.1f}s)")

    t_ks = time.time()
    full_image_ks = cv2.imread(img_path)
    n_ks = correct_key_signatures(result_staffs, staffs_sorted, predictions.original,
                                  bar_line_boxes, full_res_image=full_image_ks)
    if n_ks:
        print(f"[KeySig] Corrected {n_ks} key signature(s) ({time.time() - t_ks:.1f}s)")

    writer_notes = []
    if collect_plugin_data:
        _tag_symbols_for_plugin(result_staffs)

    xml_args = XmlGeneratorArguments(
        note_callback=_make_plugin_note_callback(writer_notes)
        if collect_plugin_data else None
    )
    xml_root = generate_xml(xml_args, result_staffs, title)
    xml_string = xml_root.to_string()

    xml_string = _retry_tromr_post_double(
        staffs_sorted, bar_line_boxes, multi_staffs,
        debug, predictions.preprocessed, xml_string, transformer_config,
        full_res_image=_vlm_image, homr_to_full_scale=_vlm_scale, use_vlm=use_vlm,
    )

    t_dyn = time.time()
    dynamics = detect_dynamics(staffs_sorted, img_path, bar_line_boxes, predictions.original.shape)
    if dynamics:
        xml_string = _inject_dynamics(xml_string, dynamics)
        print(f"[Dynamics] Injected {len(dynamics)} marking(s) ({time.time() - t_dyn:.1f}s)")

    plugin_page = None
    if collect_plugin_data:
        plugin_page = _build_plugin_page_data(
            img_path=img_path,
            page_index=page_index,
            result_staffs=result_staffs,
            staffs_sorted=staffs_sorted,
            predictions_original_shape=predictions.original.shape,
        )
        id_prefix = f"grandomr-p{page_index + 1:04d}s001"
        xml_string, prebound_notes = _prebind_plugin_notes_from_writer_map(
            xml_string,
            plugin_page,
            writer_notes,
            id_prefix=id_prefix,
        )
        plugin_page.notes = prebound_notes
        print(f"[Plugin] Mapped {len(prebound_notes)} HOMR note box(es) to XML note(s)")

    return [(xml_string, part_names, plugin_page)]


# ══════════════════════════════════════════════════════════════════════════════
# Post-process MusicXML
# ══════════════════════════════════════════════════════════════════════════════

def _inject_dynamics(xml_string: str, dynamics_list) -> str:
    """Insert <direction> elements for detected dynamics into MusicXML."""
    if not dynamics_list:
        return xml_string
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return xml_string

    parts = root.findall("part")

    seen = set()
    bound_notes = set()
    for staff_idx, measure_num, note_idx, dyn_type in dynamics_list:
        key = (staff_idx, measure_num, dyn_type)
        if key in seen:
            continue
        seen.add(key)

        if staff_idx >= len(parts):
            continue
        part = parts[staff_idx]
        measures = part.findall("measure")
        if measure_num < 1 or measure_num > len(measures):
            continue
        measure = measures[measure_num - 1]

        sounding = [ch for ch in measure if ch.tag == "note" and ch.find("rest") is None]
        if not sounding:
            continue
        target = sounding[min(note_idx, len(sounding) - 1)]
        note_key = (staff_idx, measure_num, id(target))
        if note_key in bound_notes:
            continue
        bound_notes.add(note_key)

        direction = ET.Element("direction", attrib={"placement": "below"})
        dir_type = ET.SubElement(direction, "direction-type")
        dynamics_el = ET.SubElement(dir_type, "dynamics", attrib={"default-y": "-80"})
        ET.SubElement(dynamics_el, dyn_type)

        children = list(measure)
        measure.insert(children.index(target), direction)

    return ET.tostring(root, encoding="unicode", xml_declaration=False)


def _display_name(name: str) -> str:
    """Convert internal 'Clarinet:A' format to display 'Clarinet in A'."""
    base, key = _parse_instrument_key(name)
    if key:
        return f"{base} in {key}"
    return base


def _inject_part_names(xml_string: str, part_names: List[str]) -> str:
    """Replace generic part names and instrument encoding in the MusicXML."""
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return xml_string

    part_list = root.find("part-list")
    if part_list is None:
        return xml_string

    score_parts = part_list.findall("score-part")
    parts = root.findall("part")
    for idx, (sp, name) in enumerate(zip(score_parts, part_names)):
        display = _display_name(name)
        pn = sp.find("part-name")
        if pn is not None:
            pn.text = display
        else:
            ET.SubElement(sp, "part-name").text = display

        base = _instrument_base(name)
        sound, midi_prog = INSTRUMENT_MIDI.get(base, ("keyboard.piano", 1))

        si = sp.find("score-instrument")
        if si is not None:
            iname = si.find("instrument-name")
            if iname is not None:
                iname.text = display
            isound = si.find("instrument-sound")
            if isound is not None:
                isound.text = sound

        mi = sp.find("midi-instrument")
        if mi is not None:
            mp = mi.find("midi-program")
            if mp is not None:
                mp.text = str(midi_prog)

        if idx < len(parts):
            first_m = parts[idx].find("measure")
            if first_m is not None:
                attrs = first_m.find("attributes")
                if attrs is None:
                    attrs = ET.SubElement(first_m, "attributes")
                if attrs.find("transpose") is None:
                    _inject_transpose(attrs, name)

    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _cross_part_post_process(xml_string: str, prev_ts=None, prev_bar_ended: bool = False,
                             prev_ks=None, prev_ks_changed: bool = False):
    """
    Post-process MusicXML using cross-part consistency constraints.

    Guiding principle: all parts in an orchestral score share one timeline.
    TrOMR recognizes each staff independently, so we use multi-part consensus
    to correct individual errors.

    Layer 0 — Time sig + key sig via double-barline segmentation and cross-page context
    Layer 1 — Metadata alignment: time sig, key sig (majority vote per measure)
    Layer 2 — Structural alignment: unify measure count across parts
    Layer 3 — Content repair: fix measures whose duration != time signature

    prev_ks: list of (part_name, chromatic_transpose, fifths) from previous system.
             Only parts whose (name, transpose) exactly match an entry are updated.
    prev_ks_changed: True if previous system had any double barline → trust HOMR's key.

    Returns (xml_string, last_ts, this_bar_ended, last_ks, this_ks_changed).
    last_ts:          time sig in effect at end of this section.
    this_bar_ended:   True if the LAST measure ends with a double barline (time sig scope).
    last_ks:          key sig contexts at end of this section (same format as prev_ks).
    this_ks_changed:  True if any double barline appears (key sig scope).
    """
    from collections import Counter

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return xml_string, prev_ts, False, prev_ks, False, None

    parts = root.findall("part")
    tremolo_n = _remove_invalid_two_note_tremolos(root)
    if tremolo_n:
        print(f"[PostProcess] Tremolo: removed {tremolo_n} invalid mark(s)")
    if len(parts) < 2:
        return xml_string, prev_ts, False, prev_ks, False, None

    fixes = []

    # ── Layer 0: Time signature via double-barline context ──
    all_mnums = sorted({
        int(m.get("number", ""))
        for part in parts for m in part.findall("measure")
        if m.get("number", "").lstrip("-").isdigit()
    })
    last_ts = prev_ts
    last_ks = prev_ks
    this_bar_ended = False
    this_ks_changed = False
    if all_mnums:
        m_start, m_end = all_mnums[0], all_mnums[-1]
        _all_dbls = _double_bar_measures(root)
        first_dbl = _all_dbls[0] if _all_dbls else None

        # Strip phantom measures from the END of the page.  Two cases:
        # Case A — near a page-ending double barline: strip measures after the
        #   barline (only trigger when last dbl is within 2 of the tail).
        # Case B — no nearby double barline: strip 2+ consecutive sparse trailing
        #   measures (< 1/3 parts with notes) as HOMR recognition artifacts.
        def _parts_with_notes_fn(mnum):
            return sum(
                1 for p in parts
                if any(m.get("number") == str(mnum) and m.findall("note")
                       for m in p.findall("measure"))
            )
        _last_dbl = _all_dbls[-1] if _all_dbls else None
        _phantom_stripped_b = []
        if len(all_mnums) > 1:
            _sparse_tail = []
            while len(all_mnums) > 1 and _parts_with_notes_fn(all_mnums[-1]) < max(1, len(parts) // 3):
                _sparse_tail.append(all_mnums.pop())
            if len(_sparse_tail) >= 2:
                _phantom_stripped_b = _sparse_tail
            else:
                all_mnums.extend(reversed(_sparse_tail))
        if _phantom_stripped_b:
            _s = set(_phantom_stripped_b)
            m_end = all_mnums[-1]
            _r = f"m{min(_phantom_stripped_b)}-{max(_phantom_stripped_b)}" if len(_phantom_stripped_b) > 1 else f"m{_phantom_stripped_b[0]}"
            fixes.append(f"stripped phantom {_r}")
            for _p in parts:
                for _pm in list(_p.findall("measure")):
                    try:
                        if int(_pm.get("number", "")) in _s:
                            _p.remove(_pm)
                    except (ValueError, TypeError):
                        pass
        if _last_dbl is not None and _last_dbl >= m_end - 2 and len(all_mnums) > 1:
            # When _last_dbl == m_end and m_end-1 is also a double barline,
            # m_end is a phantom's opening barline — strip from m_end downward.
            # Otherwise strip measures strictly after _last_dbl.
            _all_dbls_set = set(_all_dbls)
            _strip_from = (
                _last_dbl
                if _last_dbl == m_end and (m_end - 1) in _all_dbls_set
                else _last_dbl + 1
            )
            _stripped = []
            while len(all_mnums) > 1 and all_mnums[-1] >= _strip_from:
                _stripped.append(all_mnums.pop())
            if _stripped:
                # Only strip measures where fewer than 1/3 of parts have any notes.
                # Measures with real content (≥1/3 parts non-empty) are put back.
                _n_parts = max(1, len(parts))
                _phantom_set = set()
                _restored = []
                for _mnum in _stripped:
                    _parts_with_notes = sum(
                        1 for _p in parts
                        if any(_nd.find("rest") is None
                               for _m in _p.findall("measure")
                               if _m.get("number") == str(_mnum)
                               for _nd in _m.findall("note"))
                    )
                    if _parts_with_notes < _n_parts / 3:
                        _phantom_set.add(_mnum)
                    else:
                        _restored.append(_mnum)
                if _restored:
                    all_mnums.extend(_restored)
                    all_mnums.sort()
                    print(f"[Phantom] Kept m{sorted(_restored)}: ≥1/3 parts have notes")
                if _phantom_set:
                    _phantom_list = sorted(_phantom_set)
                    _r = (f"m{min(_phantom_list)}-{max(_phantom_list)}"
                          if len(_phantom_list) > 1 else f"m{_phantom_list[0]}")
                    fixes.append(f"stripped phantom {_r}")
                    for _p in parts:
                        for _pm in list(_p.findall("measure")):
                            try:
                                if int(_pm.get("number", "")) in _phantom_set:
                                    _p.remove(_pm)
                            except (ValueError, TypeError):
                                pass
                m_end = all_mnums[-1]

        seg1_end = first_dbl if first_dbl is not None else m_end

        def _homr_ts(from_measure, to_measure):
            """Majority-vote time sig across all parts in [from_measure, to_measure].
            Returns (ts, confidence) where confidence = fraction of votes for the winner
            (1.0 = unanimous, ~0.5 = bare majority). Returns (None, 0.0) when no valid
            time sig is found. Skips beat-type=1 (whole-note mis-emit by HOMR)."""
            from collections import Counter as _Counter
            votes: _Counter = _Counter()
            for part in root.findall("part"):
                for m in part.findall("measure"):
                    try:
                        mnum = int(m.get("number", ""))
                    except (ValueError, TypeError):
                        continue
                    if not (from_measure <= mnum <= to_measure):
                        continue
                    t = m.find(".//time")
                    if t is not None:
                        try:
                            bt = int(t.findtext("beat-type", "4"))
                            if bt == 1:
                                continue  # whole-note beat is a mis-emit; skip
                            votes[(int(t.findtext("beats", "4")), bt)] += 1
                        except ValueError:
                            pass
            if not votes:
                return None, 0.0
            best_ts, cnt = votes.most_common(1)[0]
            return best_ts, cnt / sum(votes.values())

        # Segment before (or without) first double barline
        seg1_was_fallback = False
        if prev_ts is None or prev_bar_ended:
            # No prior context, OR new section starts after a double barline:
            # Trust HOMR's time sig. At a system start TrOMR correctly reads the
            # denominator token and HOMR derives the numerator from note durations,
            # which is more reliable than our purely note-based inference.
            _h, _h_conf = _homr_ts(m_start, seg1_end)
            seg1_ts = _h or (4, 4)
            if _h is None:
                # HOMR had only invalid time sigs (e.g. beat-type=1); apply the fallback.
                seg1_was_fallback = True
                _apply_ts_to_segment(root, seg1_ts, m_start, seg1_end)
                fixes.append(f"m{m_start}-{seg1_end}: fallback {seg1_ts[0]}/{seg1_ts[1]}")
            else:
                fixes.append(f"m{m_start}-{seg1_end}: HOMR {seg1_ts[0]}/{seg1_ts[1]} (conf={_h_conf:.0%})")
            # Don't call _apply_ts_to_segment when HOMR gave a valid ts: keep it in place
        else:
            # Continuation: inherit the previous time sig as-is (no normalization so
            # that 3/2 stays 3/2 instead of being coerced to 6/4).
            # Safety check: if HOMR's denominator differs from the inherited one, a
            # new time sig is printed in this system → trust HOMR instead of inheriting.
            homr_seg1, homr_conf = _homr_ts(m_start, seg1_end)
            if homr_seg1 is not None and homr_seg1 != prev_ts and (
                homr_seg1[1] != prev_ts[1]          # denominator changed
                or homr_conf >= 0.90                # or same denom, numerator changed, very high agreement
            ):
                # A new printed time sig exists; trust HOMR.
                # Denominator change is a hard signal (TrOMR reads the token directly).
                # Same-denominator numerator change requires near-unanimous agreement to
                # avoid acting on truncation artifacts (e.g. 3/2 misread as 2/2 due to
                # missing rests), but is necessary for real changes like 3/2 → 2/2.
                seg1_ts = homr_seg1
                fixes.append(f"m{m_start}-{seg1_end}: HOMR {seg1_ts[0]}/{seg1_ts[1]} (was {prev_ts[0]}/{prev_ts[1]}, conf={homr_conf:.0%})")
            else:
                seg1_ts = prev_ts
                _apply_ts_to_segment(root, seg1_ts, m_start, seg1_end)
                fixes.append(f"m{m_start}-{seg1_end}: inherit {seg1_ts[0]}/{seg1_ts[1]}")
        last_ts = seg1_ts

        # Process each inter-double-barline segment separately so that multiple
        # section changes within one page are handled independently.
        if _all_dbls and _all_dbls[0] < m_end:
            _prev_seg_ts = seg1_ts
            for _i, _dbl in enumerate(_all_dbls):
                _seg_start = _dbl + 1
                _seg_end = _all_dbls[_i + 1] if _i + 1 < len(_all_dbls) else m_end
                if _seg_start > m_end:
                    break
                _seg_ts_h, _ = _homr_ts(_seg_start, _seg_end)
                if _seg_ts_h is not None:
                    _seg_ts = _seg_ts_h
                elif _i == 0 and seg1_was_fallback:
                    _seg_ts = seg1_ts
                elif _prev_seg_ts[1] == 4:
                    _seg_ts = _infer_ts_from_measures(root, _seg_start, _seg_end) or _prev_seg_ts
                else:
                    _seg_ts = _prev_seg_ts
                _apply_ts_to_segment(root, _seg_ts, _seg_start, _seg_end)
                last_ts = _seg_ts
                fixes.append(f"m{_seg_start}-{_seg_end}: HOMR/inferred {_seg_ts[0]}/{_seg_ts[1]}")
                _prev_seg_ts = _seg_ts

        # this_bar_ended: True only if the LAST measure ends with a double barline.
        # Mid-system double barlines don't set this because the post-double time sig
        # is already inferred and stored in last_ts, so the next system can inherit it.
        this_bar_ended = (_last_dbl is not None and _last_dbl == m_end)
        # this_ks_changed: True whenever ANY double barline appears (key sig scope).
        # The next system shows the new key at its start and should trust HOMR's key.
        this_ks_changed = bool(_all_dbls)
        # ks_fixup_range: if the double barline is mid-system, the post-double section
        # inherits the WRONG (old) key. We record this range so the caller can apply
        # the correct key once the next system is processed and HOMR's new key is known.
        ks_fixup_range = (
            (first_dbl + 1, m_end)
            if (len(_all_dbls) == 1 and first_dbl is not None and first_dbl < m_end)
            else None
        )
        if fixes:
            print(f"[TimeSig] " + "; ".join(fixes))

    # ── Layer 0b: Key signature via double-barline context ──
    # Rule: if no double barline in previous system (prev_ks_changed=False), inherit the
    # previous page/system's key sig for all matching (part_name, chromatic_transpose) pairs.
    # After a double barline, TrOMR fails at natural-sign key changes mid-system, but
    # correctly reads the key at the START of the next system — so trust HOMR's key then.
    ks_fixup_range = ks_fixup_range if all_mnums else None
    if all_mnums:
        ks_fixes = []
        if prev_ks is not None and not prev_ks_changed:
            # Continuation: inherit prev key sigs for matching parts over the whole system.
            _apply_ks_contexts(root, prev_ks, m_start, m_end)
            ks_fixes.append(f"m{m_start}-{m_end}: inherit {len(prev_ks)} parts")
        # Always snapshot the current (possibly inherited) key sigs for the next system.
        last_ks = _get_ks_contexts(root)
        if ks_fixes:
            print(f"[KeySig] " + "; ".join(ks_fixes))

    # ── Layer 1: Metadata alignment ──

    measures_by_num = {}
    for part in parts:
        for measure in part.findall("measure"):
            mn = measure.get("number")
            if mn not in measures_by_num:
                measures_by_num[mn] = []
            measures_by_num[mn].append(measure)

    for mn, measures in measures_by_num.items():
        # Time signature consensus
        time_sigs = []
        for m in measures:
            t = m.find(".//time")
            if t is not None:
                time_sigs.append((t.findtext("beats", ""), t.findtext("beat-type", "")))
        if time_sigs:
            (maj_beats, maj_bt), cnt = Counter(time_sigs).most_common(1)[0]
            if cnt < len(time_sigs):
                fixes.append(f"m{mn}: time sig → {maj_beats}/{maj_bt} (was split {Counter(time_sigs)})")
            for m in measures:
                t = m.find(".//time")
                if t is not None:
                    b, bt = t.find("beats"), t.find("beat-type")
                    if b is not None: b.text = maj_beats
                    if bt is not None: bt.text = maj_bt

    # ── Layer 2: Structural alignment (unify measure count) ──

    measure_counts = [len(p.findall("measure")) for p in parts]
    target_measures = Counter(measure_counts).most_common(1)[0][0]

    for pi, part in enumerate(parts):
        measures = part.findall("measure")
        n = len(measures)
        if n < target_measures:
            last = measures[-1] if measures else None
            for mi in range(n + 1, target_measures + 1):
                new_m = ET.SubElement(part, "measure", number=str(mi))
                rest = ET.SubElement(new_m, "note")
                ET.SubElement(rest, "rest")
                dur = ET.SubElement(rest, "duration")
                if last is not None:
                    divs = last.findtext(".//divisions", "1")
                else:
                    divs = "1"
                dur.text = str(int(divs) * 4)
                ET.SubElement(rest, "type").text = "whole"
            if n < target_measures:
                fixes.append(f"P{pi+1}: added {target_measures - n} empty measures ({n}→{target_measures})")
        elif n > target_measures:
            for m in measures[target_measures:]:
                part.remove(m)
            fixes.append(f"P{pi+1}: removed {n - target_measures} extra measures ({n}→{target_measures})")

    # ── Layer 3: Content repair (fix measure durations) ──

    # Pre-pass: chord dot/type consistency — simultaneous notes must share
    # the same duration, so if a majority have a dot the rest should too.
    for part in parts:
        pid = part.get("id", "?")
        for measure in part.findall("measure"):
            mn = measure.get("number", "?")
            children = list(measure)
            i = 0
            while i < len(children):
                if children[i].tag != "note":
                    i += 1
                    continue
                group = [children[i]]
                j = i + 1
                while j < len(children) and children[j].tag == "note" and children[j].find("chord") is not None:
                    group.append(children[j])
                    j += 1
                i = j
                if len(group) < 2:
                    continue
                dots = [n.find("dot") is not None for n in group]
                if len(set(dots)) <= 1:
                    continue
                majority_dot = sum(dots) > len(dots) / 2
                types = [n.findtext("type", "") for n in group]
                if len(set(types)) > 1:
                    continue
                lead_dur = group[0].findtext("duration", "")
                for k, note in enumerate(group):
                    has_dot = dots[k]
                    if has_dot == majority_dot:
                        continue
                    if majority_dot and not has_dot:
                        ET.SubElement(note, "dot")
                        if lead_dur:
                            dur_el = note.find("duration")
                            if dur_el is not None:
                                dur_el.text = lead_dur
                        fixes.append(f"{pid} m{mn}: chord dot added to {types[k]}")
                    elif not majority_dot and has_dot:
                        note.remove(note.find("dot"))
                        if lead_dur:
                            dur_el = note.find("duration")
                            if dur_el is not None:
                                dur_el.text = lead_dur
                        fixes.append(f"{pid} m{mn}: chord dot removed from {types[k]}")

    _TYPE_TO_QL = {
        "breve": 8.0, "whole": 4.0, "half": 2.0, "quarter": 1.0,
        "eighth": 0.5, "16th": 0.25, "32nd": 0.125, "64th": 0.0625,
    }
    _QL_TO_TYPE = {v: k for k, v in _TYPE_TO_QL.items()}
    _SORTED_QLS = sorted(_TYPE_TO_QL.values(), reverse=True)
    _ADJACENT_TYPES = {
        "breve": ["whole"], "whole": ["breve", "half"],
        "half": ["whole", "quarter"], "quarter": ["half", "eighth"],
        "eighth": ["quarter", "16th"], "16th": ["eighth", "32nd"],
        "32nd": ["16th", "64th"], "64th": ["32nd"],
    }

    def _note_expected_dur(note_el, divs):
        ntype = note_el.findtext("type", "")
        ql = _TYPE_TO_QL.get(ntype, 0)
        if ql == 0:
            return 0
        if note_el.find("dot") is not None:
            ql *= 1.5
        return round(ql * divs)

    def _measure_pos_tracking(measure):
        pos, max_pos = 0, 0
        for child in measure:
            if child.tag == "note":
                if child.find("chord") is not None:
                    continue
                try:
                    pos += int(child.findtext("duration", "0"))
                except (ValueError, TypeError):
                    pass
                max_pos = max(max_pos, pos)
            elif child.tag == "backup":
                try:
                    pos -= int(child.findtext("duration", "0"))
                except (ValueError, TypeError):
                    pass
            elif child.tag == "forward":
                try:
                    pos += int(child.findtext("duration", "0"))
                except (ValueError, TypeError):
                    pass
        return max_pos

    def _scale_all_durations(measure, ratio):
        for child in measure:
            if child.tag in ("note", "backup", "forward"):
                dur_el = child.find("duration")
                if dur_el is not None:
                    try:
                        old = int(dur_el.text)
                        dur_el.text = str(max(1, round(old * ratio)))
                    except (ValueError, TypeError):
                        pass

    for pi, part in enumerate(parts):
        pid = part.get("id", f"P{pi+1}")
        current_divs = 1
        current_beats = 4
        current_bt = 4

        for measure in part.findall("measure"):
            mn = measure.get("number", "?")

            t = measure.find(".//time")
            if t is not None:
                try:
                    current_beats = int(t.findtext("beats", "4"))
                    current_bt = int(t.findtext("beat-type", "4"))
                except ValueError:
                    pass
            d = measure.find(".//divisions")
            if d is not None:
                try:
                    current_divs = int(d.text)
                except (ValueError, TypeError):
                    pass

            expected_dur = round(current_beats * current_divs * 4.0 / current_bt)

            # Strip extra dots (TrOMR sometimes outputs double dots)
            for note in (c for c in measure if c.tag == "note"):
                dots = note.findall("dot")
                if len(dots) > 1:
                    for d in dots[1:]:
                        note.remove(d)

            actual_dur = _measure_pos_tracking(measure)

            if actual_dur == 0 or actual_dur == expected_dur:
                continue

            notes = [c for c in measure if c.tag == "note"]
            if not notes:
                continue

            # ── Fix A: whole-rest with wrong duration ──
            non_chord = [n for n in notes if n.find("chord") is None]
            if (len(non_chord) == 1 and non_chord[0].find("rest") is not None
                    and non_chord[0].findtext("type", "") in ("whole", "breve", "")):
                dur_el = non_chord[0].find("duration")
                if dur_el is not None:
                    dur_el.text = str(expected_dur)
                    fixes.append(f"{pid} m{mn}: whole rest dur {actual_dur}→{expected_dur}")
                    continue

            # ── Fix B: uniform scale if all notes are off by same ratio ──
            # This happens when divisions changed but durations weren't updated.
            # Check: does every note's dur/expected_dur give the same ratio?
            ratios = []
            for note in notes:
                exp_d = _note_expected_dur(note, current_divs)
                if exp_d <= 0:
                    continue
                dur_el = note.find("duration")
                if dur_el is None:
                    continue
                try:
                    cur_d = int(dur_el.text)
                except (ValueError, TypeError):
                    continue
                ratios.append(cur_d / exp_d)

            if ratios and len(set(round(r, 3) for r in ratios)) == 1 and abs(ratios[0] - 1.0) > 0.01:
                scale = 1.0 / ratios[0]
                _scale_all_durations(measure, scale)
                new_dur = _measure_pos_tracking(measure)
                if new_dur == expected_dur:
                    fixes.append(f"{pid} m{mn}: uniform scale ×{scale:.3f} ({actual_dur}→{expected_dur})")
                    continue
                else:
                    _scale_all_durations(measure, 1.0 / scale)

            # ── Fix B': align notes to types + recalculate backups ──
            # For each note, set duration = expected from type.
            # Then recalculate each backup as the sum of non-chord durations
            # since the previous backup (or measure start).
            note_changes = []
            for note in notes:
                exp_d = _note_expected_dur(note, current_divs)
                if exp_d <= 0:
                    continue
                dur_el = note.find("duration")
                if dur_el is None:
                    continue
                try:
                    cur_d = int(dur_el.text)
                except (ValueError, TypeError):
                    continue
                if cur_d != exp_d:
                    note_changes.append((dur_el, exp_d, cur_d))

            if note_changes:
                for dur_el, exp_d, _ in note_changes:
                    dur_el.text = str(exp_d)
                seg_dur = 0
                for child in measure:
                    if child.tag == "note":
                        if child.find("chord") is None:
                            try:
                                seg_dur += int(child.findtext("duration", "0"))
                            except (ValueError, TypeError):
                                pass
                    elif child.tag == "backup":
                        b_el = child.find("duration")
                        if b_el is not None:
                            b_el.text = str(seg_dur)
                        seg_dur = 0
                    elif child.tag == "forward":
                        try:
                            seg_dur += int(child.findtext("duration", "0"))
                        except (ValueError, TypeError):
                            pass
                new_dur = _measure_pos_tracking(measure)
                if new_dur == expected_dur:
                    fixes.append(f"{pid} m{mn}: aligned {len(note_changes)} notes + backups ({actual_dur}→{expected_dur})")
                    continue
                else:
                    for dur_el, _, old_d in note_changes:
                        dur_el.text = str(old_d)
                    seg_dur2 = 0
                    for child in measure:
                        if child.tag == "note":
                            if child.find("chord") is None:
                                try:
                                    seg_dur2 += int(child.findtext("duration", "0"))
                                except (ValueError, TypeError):
                                    pass
                        elif child.tag == "backup":
                            b_el = child.find("duration")
                            if b_el is not None:
                                b_el.text = str(seg_dur2)
                            seg_dur2 = 0
                        elif child.tag == "forward":
                            try:
                                seg_dur2 += int(child.findtext("duration", "0"))
                            except (ValueError, TypeError):
                                pass

            # ── Fix C: single-note type change (adjacent types only) ──
            actual_dur = _measure_pos_tracking(measure)
            diff = actual_dur - expected_dur
            if diff == 0:
                continue

            best_fix = None
            for note in notes:
                if note.find("chord") is not None:
                    continue
                dur_el = note.find("duration")
                if dur_el is None:
                    continue
                try:
                    cur_d = int(dur_el.text)
                except (ValueError, TypeError):
                    continue
                cur_type = note.findtext("type", "")
                cur_dot = note.find("dot") is not None
                target_d = cur_d - diff
                if target_d <= 0:
                    continue
                target_ql = target_d / current_divs

                candidates = _ADJACENT_TYPES.get(cur_type, [])
                for cand_type in candidates:
                    cand_ql = _TYPE_TO_QL[cand_type]
                    for dotted in (False, True):
                        ql = cand_ql * 1.5 if dotted else cand_ql
                        if abs(target_ql - ql) < 0.001:
                            cost = abs(cur_d - target_d)
                            if best_fix is None or cost < best_fix[0]:
                                best_fix = (cost, note, dur_el, target_d, cand_type, dotted)
                if not cur_dot:
                    cand_ql = _TYPE_TO_QL.get(cur_type, 0) * 1.5
                    if abs(target_ql - cand_ql) < 0.001:
                        cost = abs(cur_d - target_d)
                        if best_fix is None or cost < best_fix[0]:
                            best_fix = (cost, note, dur_el, target_d, cur_type, True)
                if cur_dot:
                    cand_ql = _TYPE_TO_QL.get(cur_type, 0)
                    if abs(target_ql - cand_ql) < 0.001:
                        cost = abs(cur_d - target_d)
                        if best_fix is None or cost < best_fix[0]:
                            best_fix = (cost, note, dur_el, target_d, cur_type, False)

            if best_fix is not None:
                _, note, dur_el, new_d, new_type, new_dot = best_fix
                old_type = note.findtext("type", "?")
                old_dot = "." if note.find("dot") is not None else ""
                old_d = dur_el.text
                dur_el.text = str(new_d)
                type_el = note.find("type")
                if type_el is not None:
                    type_el.text = new_type
                if new_dot and note.find("dot") is None:
                    ET.SubElement(note, "dot")
                elif not new_dot and note.find("dot") is not None:
                    note.remove(note.find("dot"))
                fixes.append(f"{pid} m{mn}: {old_type}{old_dot}({old_d}) → {new_type}{'.' if new_dot else ''}({new_d})")
                continue

            # ── Fix C': shrink/remove a rest to fix overshoot ──
            # If the measure is too long, find a rest whose duration can be
            # reduced (to a standard value) or removed to make the total exact.
            actual_dur = _measure_pos_tracking(measure)
            diff = actual_dur - expected_dur
            if diff > 0:
                best_rest_fix = None
                for note in notes:
                    if note.find("chord") is not None:
                        continue
                    if note.find("rest") is None:
                        continue
                    dur_el = note.find("duration")
                    if dur_el is None:
                        continue
                    try:
                        cur_d = int(dur_el.text)
                    except (ValueError, TypeError):
                        continue
                    new_d = cur_d - diff
                    if new_d < 0:
                        continue
                    if new_d == 0:
                        # removing rest entirely — prefer shrinking over removal
                        cost = cur_d * 10
                        if best_rest_fix is None or cost < best_rest_fix[0]:
                            best_rest_fix = (cost, note, dur_el, new_d, "remove")
                        continue
                    new_ql = new_d / current_divs
                    # accept if it maps to any standard note value (with or without dot)
                    matched = False
                    for std_ql in _SORTED_QLS:
                        if abs(new_ql - std_ql) < 0.001 or abs(new_ql - std_ql * 1.5) < 0.001:
                            matched = True
                            break
                    if matched:
                        cost = diff
                        if best_rest_fix is None or cost < best_rest_fix[0]:
                            best_rest_fix = (cost, note, dur_el, new_d, "shrink")

                if best_rest_fix is not None:
                    _, note, dur_el, new_d, action = best_rest_fix
                    old_type = note.findtext("type", "?")
                    old_d = dur_el.text
                    if action == "remove" and new_d == 0:
                        measure.remove(note)
                        fixes.append(f"{pid} m{mn}: removed rest {old_type}({old_d})")
                    else:
                        dur_el.text = str(new_d)
                        new_ql = new_d / current_divs
                        for std_ql in _SORTED_QLS:
                            if abs(new_ql - std_ql) < 0.001:
                                new_type = _QL_TO_TYPE[std_ql]
                                new_dot = False
                                break
                            if abs(new_ql - std_ql * 1.5) < 0.001:
                                new_type = _QL_TO_TYPE[std_ql]
                                new_dot = True
                                break
                        else:
                            new_type = old_type
                            new_dot = note.find("dot") is not None
                        type_el = note.find("type")
                        if type_el is not None:
                            type_el.text = new_type
                        if new_dot and note.find("dot") is None:
                            ET.SubElement(note, "dot")
                        elif not new_dot and note.find("dot") is not None:
                            note.remove(note.find("dot"))
                        ndot = "." if new_dot else ""
                        fixes.append(f"{pid} m{mn}: rest {old_type}({old_d}) → {new_type}{ndot}({new_d})")
                    continue

            # ── Fix T: triplet detection ──
            # If 3 consecutive equal-duration notes exist whose individual
            # duration equals the overshoot, convert them to a triplet.
            actual_dur = _measure_pos_tracking(measure)
            diff = actual_dur - expected_dur
            if diff > 0:
                voice_notes = {}
                for note in notes:
                    if note.find("chord") is not None:
                        continue
                    v = note.findtext("voice", "1")
                    voice_notes.setdefault(v, []).append(note)

                triplet_fixed = False
                for v, vnotes in voice_notes.items():
                    v_total = sum(int(n.findtext("duration", "0")) for n in vnotes)
                    v_diff = v_total - expected_dur
                    if v_diff <= 0:
                        continue
                    for i in range(len(vnotes) - 2):
                        d0 = int(vnotes[i].findtext("duration", "0"))
                        d1 = int(vnotes[i + 1].findtext("duration", "0"))
                        d2 = int(vnotes[i + 2].findtext("duration", "0"))
                        if d0 == d1 == d2 == v_diff and d0 > 0:
                            new_d = round(d0 * 2 / 3)
                            for j in range(3):
                                n = vnotes[i + j]
                                n.find("duration").text = str(new_d)
                                if n.find("time-modification") is None:
                                    tm = ET.SubElement(n, "time-modification")
                                    ET.SubElement(tm, "actual-notes").text = "3"
                                    ET.SubElement(tm, "normal-notes").text = "2"
                                notations = n.find("notations")
                                if notations is None:
                                    notations = ET.SubElement(n, "notations")
                                if j == 0:
                                    ET.SubElement(notations, "tuplet",
                                                  type="start", bracket="yes", number="1")
                                elif j == 2:
                                    ET.SubElement(notations, "tuplet",
                                                  type="stop", number="1")
                            triplet_fixed = True
                            fixes.append(f"{pid} m{mn}: triplet v{v} ({d0}→{new_d})×3")
                            break
                    if triplet_fixed:
                        break
                if triplet_fixed:
                    continue

            # ── Fix D: fallback — uniform scale + rounding correction ──
            actual_dur = _measure_pos_tracking(measure)
            if actual_dur > 0 and actual_dur != expected_dur:
                ratio = expected_dur / actual_dur
                if 0.3 < ratio < 3.0:
                    _scale_all_durations(measure, ratio)
                    new_dur = _measure_pos_tracking(measure)
                    if new_dur != expected_dur:
                        residual = new_dur - expected_dur
                        longest = None
                        for child in measure:
                            if child.tag == "note" and child.find("chord") is None:
                                d_el = child.find("duration")
                                if d_el is not None:
                                    try:
                                        v = int(d_el.text)
                                    except (ValueError, TypeError):
                                        continue
                                    if v - residual > 0 and (longest is None or v > longest[1]):
                                        longest = (d_el, v)
                        if longest:
                            longest[0].text = str(longest[1] - residual)
                    fixes.append(f"{pid} m{mn}: scaled ×{ratio:.2f} ({actual_dur}→{expected_dur})")

    # ── Fix V: per-voice duration correction for multi-voice measures ──
    vfix_n = _fix_voice_durations(root)
    if vfix_n:
        fixes.append(f"V-fix: corrected {vfix_n} multi-voice measures")

    # ── Fix Triplet: mark unmarked triplets ──
    trip_n = _mark_unmarked_triplets(root)
    if trip_n:
        fixes.append(f"Triplet: marked {trip_n} notes")

    # ── Fix Overflow: remove excess rests when notated types exceed bar ──
    overflow_n = _fix_notated_overflow(root)
    if overflow_n:
        fixes.append(f"Overflow: removed {overflow_n} excess rests")

    # ── Fix E: type-duration alignment (final pass) ──
    fixes.extend(_fix_type_duration_alignment(root, label="E-"))

    tremolo_n2 = _remove_invalid_two_note_tremolos(root)
    if tremolo_n2:
        fixes.append(f"Tremolo: removed {tremolo_n2} invalid mark(s)")

    if fixes:
        print(f"[PostProcess] {len(fixes)} fixes:")
        for f in fixes:
            print(f"  {f}")
    else:
        print("[PostProcess] No fixes needed")

    return ET.tostring(root, encoding="unicode", xml_declaration=True), last_ts, this_bar_ended, last_ks, this_ks_changed, ks_fixup_range


def _remove_invalid_two_note_tremolos(root):
    """Remove two-note tremolo pairs that cannot be rendered locally.

    A measured two-note tremolo should connect two adjacent chord events in the
    same part, measure, staff, and voice. Cross-measure pairs or pairs with
    intervening rhythmic events create long beams in MuseScore and are usually
    false positives from template matching.
    """
    removed = 0

    def chord_event_key(note):
        staff = note.findtext("staff", "1")
        voice = note.findtext("voice", "1")
        return staff, voice

    def note_has_pitch(note):
        return note.find("pitch") is not None

    def event_key(event):
        first = event[0]
        return chord_event_key(first)

    for part in root.findall("part"):
        open_start = {}
        for measure in part.findall("measure"):
            events = []
            current = []
            for child in measure:
                if child.tag != "note":
                    continue
                if child.find("chord") is None:
                    if current:
                        events.append(current)
                    current = [child]
                else:
                    if current:
                        current.append(child)
                    else:
                        current = [child]
            if current:
                events.append(current)

            measure_starts = {}
            measure_stops = {}
            for idx, event in enumerate(events):
                for note in event:
                    trem = note.find("./notations/ornaments/tremolo")
                    if trem is None:
                        continue
                    trem_type = trem.attrib.get("type")
                    if trem_type == "start":
                        measure_starts.setdefault(event_key(event), []).append((idx, note))
                    elif trem_type == "stop":
                        measure_stops.setdefault(event_key(event), []).append((idx, note))

            valid_start_notes = set()
            valid_stop_notes = set()
            for key, starts in measure_starts.items():
                stops = measure_stops.get(key, [])
                used_stops = set()
                for start_idx, start_note in starts:
                    match = None
                    for stop_pos, (stop_idx, stop_note) in enumerate(stops):
                        if stop_pos in used_stops:
                            continue
                        if stop_idx <= start_idx:
                            continue
                        between = events[start_idx + 1:stop_idx]
                        if any(any(note_has_pitch(n) or n.find("rest") is not None for n in ev) for ev in between):
                            continue
                        match = (stop_pos, stop_note)
                        break
                    if match is not None:
                        used_stops.add(match[0])
                        valid_start_notes.add(id(start_note))
                        valid_stop_notes.add(id(match[1]))

            # Any unmatched start from an earlier measure is invalid once a new
            # measure begins; two-note tremolo must not cross measure boundaries.
            for prev_note in open_start.values():
                removed += _remove_tremolo_element(prev_note)
            open_start = {}

            for idx, event in enumerate(events):
                for note in event:
                    trem = note.find("./notations/ornaments/tremolo")
                    if trem is None:
                        continue
                    trem_type = trem.attrib.get("type")
                    if trem_type == "start":
                        if id(note) not in valid_start_notes:
                            open_start[chord_event_key(note)] = note
                    elif trem_type == "stop":
                        if id(note) not in valid_stop_notes:
                            removed += _remove_tremolo_element(note)

            for key, note in list(open_start.items()):
                removed += _remove_tremolo_element(note)
                del open_start[key]

        for note in open_start.values():
            removed += _remove_tremolo_element(note)

    return removed


def _remove_tremolo_element(note):
    notations = note.find("notations")
    if notations is None:
        return 0
    ornaments = notations.find("ornaments")
    if ornaments is None:
        return 0
    tremolo = ornaments.find("tremolo")
    if tremolo is None:
        return 0
    ornaments.remove(tremolo)
    if len(ornaments) == 0:
        notations.remove(ornaments)
    if len(notations) == 0:
        note.remove(notations)
    return 1


def _fix_voice_durations(root):
    """Fix per-voice duration mismatches in multi-voice measures.

    Scales notes of each wrong voice to fit expected duration,
    then restructures all multi-voice measures into contiguous voice
    blocks to ensure correct position tracking.
    Returns number of measures fixed.
    """
    parts = root.findall("part")
    count = 0

    for part in parts:
        cur_divs = 1
        cur_beats = 4
        cur_bt = 4

        for measure in part.findall("measure"):
            t = measure.find(".//time")
            if t is not None:
                try:
                    cur_beats = int(t.findtext("beats", "4"))
                    cur_bt = int(t.findtext("beat-type", "4"))
                except ValueError:
                    pass
            d = measure.find(".//divisions")
            if d is not None:
                try:
                    cur_divs = int(d.text)
                except (ValueError, TypeError):
                    pass

            has_backup = any(ch.tag == "backup" for ch in measure)

            expected_dur = round(cur_beats * cur_divs * 4.0 / cur_bt)

            voice_durs = {}
            for ch in measure:
                if ch.tag == "note" and ch.find("chord") is None:
                    v = ch.findtext("voice", "1")
                    voice_durs.setdefault(v, 0)
                    voice_durs[v] += int(ch.findtext("duration", "0"))

            wrong = {v: tot for v, tot in voice_durs.items()
                     if tot != expected_dur and tot > 0}

            if wrong:
                # Try triplet fix first for each wrong voice
                for v, total in list(wrong.items()):
                    v_diff = total - expected_dur
                    if v_diff <= 0:
                        continue
                    vnotes = [ch for ch in measure
                              if ch.tag == "note" and ch.find("chord") is None
                              and ch.findtext("voice", "1") == v]
                    for i in range(len(vnotes) - 2):
                        d0 = int(vnotes[i].findtext("duration", "0"))
                        d1 = int(vnotes[i + 1].findtext("duration", "0"))
                        d2 = int(vnotes[i + 2].findtext("duration", "0"))
                        if d0 == d1 == d2 == v_diff and d0 > 0:
                            new_d = round(d0 * 2 / 3)
                            for j in range(3):
                                n = vnotes[i + j]
                                n.find("duration").text = str(new_d)
                                if n.find("time-modification") is None:
                                    tm = ET.SubElement(n, "time-modification")
                                    ET.SubElement(tm, "actual-notes").text = "3"
                                    ET.SubElement(tm, "normal-notes").text = "2"
                                notations = n.find("notations")
                                if notations is None:
                                    notations = ET.SubElement(n, "notations")
                                if j == 0:
                                    ET.SubElement(notations, "tuplet",
                                                  type="start", bracket="yes", number="1")
                                elif j == 2:
                                    ET.SubElement(notations, "tuplet",
                                                  type="stop", number="1")
                            del wrong[v]
                            break

                # Scale remaining wrong voices
                for v, total in wrong.items():
                    deficit = expected_dur - total
                    if deficit == 0:
                        continue
                    ratio = expected_dur / total
                    if not (0.3 < ratio < 3.0):
                        continue

                    vnotes = [ch for ch in measure
                              if ch.tag == "note" and ch.find("chord") is None
                              and ch.findtext("voice", "1") == v]

                    # Try smart fix: upgrade one note to fill deficit
                    fixed = False
                    if deficit > 0:
                        std_durs = [192, 144, 96, 72, 48, 36, 24, 18, 12, 9, 6]
                        std_map = {192: ("whole", False), 144: ("half", True),
                                   96: ("half", False), 72: ("quarter", True),
                                   48: ("quarter", False), 36: ("eighth", True),
                                   24: ("eighth", False), 18: ("16th", True),
                                   12: ("16th", False), 9: ("32nd", True),
                                   6: ("32nd", False)}
                        for note in vnotes:
                            dur_el = note.find("duration")
                            if dur_el is None:
                                continue
                            cur_d = int(dur_el.text or "0")
                            target_d = cur_d + deficit
                            if target_d in std_map:
                                new_type, new_dot = std_map[target_d]
                                dur_el.text = str(target_d)
                                type_el = note.find("type")
                                if type_el is not None:
                                    type_el.text = new_type
                                dot_el = note.find("dot")
                                if new_dot and dot_el is None:
                                    ET.SubElement(note, "dot")
                                elif not new_dot and dot_el is not None:
                                    note.remove(dot_el)
                                # Sync chord notes
                                for ch2 in measure:
                                    if ch2.tag == "note" and ch2.find("chord") is not None:
                                        if ch2.findtext("voice", "1") == v:
                                            cd = ch2.find("duration")
                                            if cd is not None and cd.text == str(cur_d):
                                                cd.text = str(target_d)
                                fixed = True
                                break

                    if not fixed:
                        # Fallback: proportional scaling
                        for ch in measure:
                            if ch.tag == "note" and ch.findtext("voice", "1") == v:
                                dur_el = ch.find("duration")
                                if dur_el is not None:
                                    dur_el.text = str(max(1, round(int(dur_el.text) * ratio)))

                        new_total = sum(
                            int(ch.findtext("duration", "0"))
                            for ch in measure
                            if ch.tag == "note" and ch.find("chord") is None
                            and ch.findtext("voice", "1") == v
                        )
                        residual = new_total - expected_dur
                        if residual != 0:
                            longest_el, longest_val = None, 0
                            for ch in measure:
                                if (ch.tag == "note" and ch.find("chord") is None
                                        and ch.findtext("voice", "1") == v):
                                    dv = int(ch.findtext("duration", "0"))
                                    if dv - residual > 0 and dv > longest_val:
                                        longest_el = ch.find("duration")
                                        longest_val = dv
                            if longest_el is not None:
                                longest_el.text = str(longest_val - residual)

            if not has_backup:
                if wrong:
                    count += 1
                continue

            # Restructure: collect elements per voice, rebuild contiguously
            preamble = []  # attributes, etc. before any note
            voice_elems = {}  # voice_num -> [elements]
            pending_dirs = []  # directions waiting to be assigned
            cur_voice = None
            seen_note = False

            for ch in measure:
                if ch.tag in ("attributes",):
                    if not seen_note:
                        preamble.append(ch)
                    # attributes mid-measure: attach to current voice
                    elif cur_voice is not None:
                        voice_elems.setdefault(cur_voice, []).append(ch)
                elif ch.tag == "note":
                    seen_note = True
                    v = ch.findtext("voice", "1")
                    cur_voice = v
                    elems = voice_elems.setdefault(v, [])
                    elems.extend(pending_dirs)
                    pending_dirs.clear()
                    elems.append(ch)
                elif ch.tag == "direction":
                    pending_dirs.append(ch)
                elif ch.tag in ("backup", "forward"):
                    pass  # drop old backups/forwards

            if not voice_elems:
                continue

            # Rebuild measure: preamble, then each voice with backup between
            for ch in list(measure):
                measure.remove(ch)
            for ch in preamble:
                measure.append(ch)

            sorted_voices = sorted(voice_elems.keys())
            for vi, v in enumerate(sorted_voices):
                if vi > 0:
                    backup = ET.SubElement(measure, "backup")
                    ET.SubElement(backup, "duration").text = str(expected_dur)
                for ch in voice_elems[v]:
                    measure.append(ch)
            for ch in pending_dirs:
                measure.append(ch)

            count += 1

    return count


_TRIPLET_TYPE_QL = {
    "whole": 4.0, "half": 2.0, "quarter": 1.0,
    "eighth": 0.5, "16th": 0.25, "32nd": 0.125,
}


def _mark_unmarked_triplets(root):
    """Detect notes with triplet durations and mark them.

    Two passes:
    1. Exact match: each note's duration == expected * 2/3 exactly.
    2. Group match: runs of same-type notes whose total == N * expected * 2/3,
       even if individual durations are imprecise. Corrects durations.
    Returns the number of notes marked.
    """
    parts = root.findall("part")
    total_marked = 0

    for part in parts:
        cur_divs = 1

        for measure in part.findall("measure"):
            d = measure.find(".//divisions")
            if d is not None:
                try:
                    cur_divs = int(d.text)
                except (ValueError, TypeError):
                    pass

            voice_notes = {}
            for ch in measure:
                if ch.tag != "note":
                    continue
                if ch.find("time-modification") is not None:
                    continue
                v = ch.findtext("voice", "1")
                is_chord = ch.find("chord") is not None
                voice_notes.setdefault(v, []).append((ch, is_chord))

            for v, entries in voice_notes.items():
                main_notes = [(n, idx) for idx, (n, is_chord) in enumerate(entries)
                              if not is_chord]

                # Pass 1: exact match (divisions must allow clean triplets)
                i = 0
                while i < len(main_notes):
                    n, _ = main_notes[i]
                    typ = n.findtext("type", "")
                    ql = _TRIPLET_TYPE_QL.get(typ, 0)
                    if ql <= 0:
                        i += 1
                        continue
                    expected_d = round(ql * cur_divs)
                    if expected_d * 2 % 3 != 0:
                        i += 1
                        continue
                    dur = int(n.findtext("duration", "0"))
                    triplet_d = expected_d * 2 // 3
                    if dur != triplet_d or triplet_d <= 0:
                        i += 1
                        continue

                    run = 1
                    while i + run < len(main_notes):
                        nn, _ = main_notes[i + run]
                        ntyp = nn.findtext("type", "")
                        nql = _TRIPLET_TYPE_QL.get(ntyp, 0)
                        if nql <= 0:
                            break
                        nexp = round(nql * cur_divs)
                        if nexp * 2 % 3 != 0:
                            break
                        ndur = int(nn.findtext("duration", "0"))
                        ntrip = nexp * 2 // 3
                        if ndur != ntrip:
                            break
                        run += 1

                    groups = run // 3
                    if groups > 0:
                        total_marked += _apply_triplet_markup(
                            main_notes, entries, i, groups)
                    i += groups * 3 if groups > 0 else 1

                # Pass 2: group-total match (for imprecise durations)
                i = 0
                while i < len(main_notes):
                    n, _ = main_notes[i]
                    if n.find("time-modification") is not None:
                        i += 1
                        continue
                    typ = n.findtext("type", "")
                    ql = _TRIPLET_TYPE_QL.get(typ, 0)
                    if ql <= 0:
                        i += 1
                        continue
                    expected_d = round(ql * cur_divs)
                    if expected_d < 2:
                        i += 1
                        continue

                    run = 1
                    while i + run < len(main_notes):
                        nn, _ = main_notes[i + run]
                        if nn.find("time-modification") is not None:
                            break
                        ntyp = nn.findtext("type", "")
                        if ntyp != typ:
                            break
                        run += 1

                    groups = run // 3
                    if groups > 0:
                        run_len = groups * 3
                        run_total = sum(
                            int(main_notes[i + j][0].findtext("duration", "0"))
                            for j in range(run_len))
                        expected_triplet_total = round(run_len * expected_d * 2 / 3)
                        if abs(run_total - expected_triplet_total) <= groups:
                            if run_total == expected_triplet_total:
                                _fix_triplet_durations(
                                    main_notes, entries, i, groups, expected_d)
                            total_marked += _apply_triplet_markup(
                                main_notes, entries, i, groups)
                            i += run_len
                            continue
                    i += 1

    return total_marked


def _fix_triplet_durations(main_notes, entries, start, groups, expected_d):
    """Correct note durations to proper triplet values for imprecise groups."""
    group_total = expected_d * 2
    base_d = group_total // 3
    extra = group_total % 3
    for g in range(groups):
        for j in range(3):
            mn_note, mn_idx = main_notes[start + g * 3 + j]
            correct_d = base_d + (1 if j < extra else 0)
            d_elem = mn_note.find("duration")
            if d_elem is not None:
                d_elem.text = str(correct_d)
            for ci in range(mn_idx + 1, len(entries)):
                cn, cn_chord = entries[ci]
                if not cn_chord:
                    break
                cd = cn.find("duration")
                if cd is not None:
                    cd.text = str(correct_d)


def _apply_triplet_markup(main_notes, entries, start, groups):
    """Add time-modification and tuplet notation to groups of 3 notes."""
    marked = 0
    for g in range(groups):
        for j in range(3):
            mn_note, mn_idx = main_notes[start + g * 3 + j]
            all_notes = [mn_note]
            for ci in range(mn_idx + 1, len(entries)):
                cn, cn_chord = entries[ci]
                if not cn_chord:
                    break
                all_notes.append(cn)
            for note in all_notes:
                if note.find("time-modification") is None:
                    tm = ET.SubElement(note, "time-modification")
                    ET.SubElement(tm, "actual-notes").text = "3"
                    ET.SubElement(tm, "normal-notes").text = "2"
                    marked += 1
            notations = mn_note.find("notations")
            if notations is None:
                notations = ET.SubElement(mn_note, "notations")
            if j == 0:
                ET.SubElement(notations, "tuplet",
                              type="start", bracket="yes", number="1")
            elif j == 2:
                ET.SubElement(notations, "tuplet",
                              type="stop", number="1")
    return marked


# ── Notated overflow fix: remove excess rests when types exceed bar ──

def _fix_notated_overflow(root):
    """Remove excess rests when type-based voice total exceeds bar duration.

    E.g., whole(4 beats) + rest-eighth(0.5) + rest-quarter(1) in 4/4
    = 5.5 beats > 4 → remove the rests and set whole duration to fill bar.
    """
    _TYPE_BEATS = {
        "breve": 8.0, "whole": 4.0, "half": 2.0, "quarter": 1.0,
        "eighth": 0.5, "16th": 0.25, "32nd": 0.125, "64th": 0.0625,
    }
    removed = 0
    for part in (root.findall(".//part") or root.findall("part")):
        divs = 1
        ts_beats, ts_btype = 4, 4
        for measure in part.findall("measure"):
            att = measure.find("attributes")
            if att is not None:
                d = att.findtext("divisions")
                if d:
                    try:
                        divs = int(d)
                    except ValueError:
                        pass
                t = att.find("time")
                if t is not None:
                    try:
                        ts_beats = int(t.findtext("beats", "4"))
                        ts_btype = int(t.findtext("beat-type", "4"))
                    except ValueError:
                        pass

            expected_beats = ts_beats * (4.0 / ts_btype)

            voices = {}
            for child in measure:
                if child.tag != "note" or child.find("grace") is not None:
                    continue
                if child.find("chord") is not None:
                    continue
                v = child.findtext("voice", "1")
                typ = child.findtext("type", "")
                ndots = len(child.findall("dot"))
                trip = child.find("time-modification") is not None
                is_rest = child.find("rest") is not None
                tb = _TYPE_BEATS.get(typ, 0)
                for _ in range(ndots):
                    tb *= 1.5
                if trip:
                    tb *= 2.0 / 3.0
                if v not in voices:
                    voices[v] = []
                voices[v].append((child, tb, is_rest))

            for v, entries in voices.items():
                total = sum(e[1] for e in entries)
                if total <= expected_beats + 0.01:
                    continue
                overflow = total - expected_beats
                rests = [e for e in entries if e[2]]
                rests.sort(key=lambda e: -e[1])
                to_remove = []
                remaining = overflow
                for entry in rests:
                    if remaining <= 0.01:
                        break
                    if entry[1] <= remaining + 0.01:
                        to_remove.append(entry[0])
                        remaining -= entry[1]
                if abs(remaining) > 0.01:
                    continue
                for el in to_remove:
                    measure.remove(el)
                    removed += 1
                for child in measure:
                    if child.tag != "note" or child.find("grace") is not None:
                        continue
                    if child.findtext("voice", "1") != v:
                        continue
                    exp_d = _td_note_expected(child, divs)
                    if exp_d > 0:
                        d_el = child.find("duration")
                        if d_el is not None:
                            d_el.text = str(exp_d)
                _td_recalc_backups(measure)
    return removed


# ── Module-level type-duration alignment (used after merge too) ──

_TD_TYPE_TO_QL = {
    "breve": 8.0, "whole": 4.0, "half": 2.0, "quarter": 1.0,
    "eighth": 0.5, "16th": 0.25, "32nd": 0.125, "64th": 0.0625,
}
_TD_QL_TO_TYPE = {v: k for k, v in _TD_TYPE_TO_QL.items()}
_TD_SORTED_QLS = sorted(_TD_TYPE_TO_QL.values(), reverse=True)


def _td_note_expected(note_el, divs):
    ntype = note_el.findtext("type", "")
    ql = _TD_TYPE_TO_QL.get(ntype, 0)
    if ql == 0:
        return 0
    ndots = len(note_el.findall("dot"))
    for _ in range(ndots):
        ql *= 1.5
    return round(ql * divs)


def _td_measure_max_pos(measure):
    pos, mx = 0, 0
    for child in measure:
        if child.tag == "note":
            if child.find("chord") is not None:
                continue
            try:
                pos += int(child.findtext("duration", "0"))
            except (ValueError, TypeError):
                pass
            mx = max(mx, pos)
        elif child.tag == "backup":
            try:
                pos -= int(child.findtext("duration", "0"))
            except (ValueError, TypeError):
                pass
        elif child.tag == "forward":
            try:
                pos += int(child.findtext("duration", "0"))
            except (ValueError, TypeError):
                pass
    return mx


def _td_recalc_backups(measure):
    seg = 0
    for child in measure:
        if child.tag == "note":
            if child.find("chord") is None:
                try:
                    seg += int(child.findtext("duration", "0"))
                except (ValueError, TypeError):
                    pass
        elif child.tag == "backup":
            b_el = child.find("duration")
            if b_el is not None:
                b_el.text = str(seg)
            seg = 0
        elif child.tag == "forward":
            try:
                seg += int(child.findtext("duration", "0"))
            except (ValueError, TypeError):
                pass


def _fix_type_duration_alignment(root, label=""):
    """Fix type-duration mismatches.  Phase 1: retype (safe). Phase 2: realign durations."""
    parts = root.findall(".//part") or root.findall("part")
    fixes = []

    # Phase 1: update types to match durations (no duration changes)
    for pi, part in enumerate(parts):
        pid = part.get("id", f"P{pi+1}")
        e_divs = 1
        for measure in part.findall("measure"):
            mn = measure.get("number", "?")
            dv = measure.find(".//divisions")
            if dv is not None:
                try:
                    e_divs = int(dv.text)
                except (ValueError, TypeError):
                    pass
            if e_divs <= 0:
                continue
            n_fixed = 0
            for note in measure.findall("note"):
                type_el = note.find("type")
                dur_el = note.find("duration")
                if type_el is None or dur_el is None:
                    continue
                ntype = type_el.text
                if ntype not in _TD_TYPE_TO_QL:
                    continue
                try:
                    dur = int(dur_el.text)
                except (ValueError, TypeError):
                    continue
                exp = _td_note_expected(note, e_divs)
                if dur == exp:
                    continue
                ql = dur / e_divs
                for sq in _TD_SORTED_QLS:
                    if abs(ql - sq) < 0.001:
                        type_el.text = _TD_QL_TO_TYPE[sq]
                        for d in note.findall("dot"):
                            note.remove(d)
                        n_fixed += 1
                        break
                    if abs(ql - sq * 1.5) < 0.001:
                        type_el.text = _TD_QL_TO_TYPE[sq]
                        for d in note.findall("dot"):
                            note.remove(d)
                        ET.SubElement(note, "dot")
                        n_fixed += 1
                        break
            if n_fixed:
                fixes.append(f"{pid} m{mn}: {label}retyped {n_fixed} notes")

    # Phase 2: for remaining mismatches, change durations to match types + compensate
    for pi, part in enumerate(parts):
        pid = part.get("id", f"P{pi+1}")
        e_divs = 1
        e_beats, e_bt = 4, 4
        for measure in part.findall("measure"):
            mn = measure.get("number", "?")
            t = measure.find(".//time")
            if t is not None:
                try:
                    e_beats = int(t.findtext("beats", "4"))
                    e_bt = int(t.findtext("beat-type", "4"))
                except ValueError:
                    pass
            dv = measure.find(".//divisions")
            if dv is not None:
                try:
                    e_divs = int(dv.text)
                except (ValueError, TypeError):
                    pass
            if e_divs <= 0:
                continue
            expected_dur = round(e_beats * e_divs * 4.0 / e_bt)
            notes = [c for c in measure if c.tag == "note"]
            if not notes:
                continue
            mismatches = []
            for note in notes:
                exp_d = _td_note_expected(note, e_divs)
                if exp_d <= 0:
                    continue
                dur_el = note.find("duration")
                if dur_el is None:
                    continue
                try:
                    cur_d = int(dur_el.text)
                except (ValueError, TypeError):
                    continue
                if cur_d != exp_d:
                    mismatches.append((note, dur_el, cur_d, exp_d))
            if not mismatches:
                continue
            old_state = {}
            for child in measure:
                if child.tag in ("note", "backup", "forward"):
                    d_el = child.find("duration")
                    if d_el is not None:
                        old_state[d_el] = d_el.text
            adjusted_els = set()
            for note, dur_el, _, exp_d in mismatches:
                dur_el.text = str(exp_d)
                adjusted_els.add(id(dur_el))
                for ch in measure:
                    if ch.tag == "note" and ch.find("chord") is not None:
                        cd = ch.find("duration")
                        if cd is not None and id(cd) in {id(d) for _, d, _, _ in mismatches}:
                            continue
                        prev_main = None
                        for ch2 in measure:
                            if ch2 is ch:
                                break
                            if ch2.tag == "note" and ch2.find("chord") is None:
                                prev_main = ch2
                        if prev_main is note and cd is not None:
                            cd.text = str(exp_d)
                            adjusted_els.add(id(cd))
            _td_recalc_backups(measure)
            new_dur = _td_measure_max_pos(measure)
            if new_dur == expected_dur:
                fixes.append(f"{pid} m{mn}: {label}realigned {len(mismatches)} notes")
                continue
            residual = new_dur - expected_dur
            # Try compensation: prefer rests, then non-adjusted notes, avoid adjusted notes
            candidates = []
            for child in measure:
                if child.tag == "note" and child.find("chord") is None:
                    d_el = child.find("duration")
                    if d_el is None:
                        continue
                    try:
                        v = int(d_el.text)
                    except (ValueError, TypeError):
                        continue
                    if v - residual <= 0:
                        continue
                    is_rest = child.find("rest") is not None
                    is_adjusted = id(d_el) in adjusted_els
                    # Priority: rest=0 (best), non-adjusted=1, adjusted=2 (worst)
                    priority = 0 if is_rest else (2 if is_adjusted else 1)
                    candidates.append((priority, -v, id(d_el), d_el, v))
            candidates.sort()
            if candidates:
                _, _, _, best_el, best_v = candidates[0]
                best_el.text = str(best_v - residual)
                _td_recalc_backups(measure)
                fixes.append(f"{pid} m{mn}: {label}realigned {len(mismatches)} notes + compensated")
            else:
                for d_el, oval in old_state.items():
                    d_el.text = oval
    return fixes

STANDARD_RANGES = {
    "Flute": (60, 96), "Piccolo": (74, 108),
    "Oboe": (58, 91), "English Horn": (52, 81),
    "Clarinet": (50, 91), "Bass Clarinet": (38, 77),
    "Bassoon": (34, 72), "Contrabassoon": (22, 53),
    "Horn": (34, 77), "Trumpet": (54, 82),
    "Trombone": (40, 72), "Tuba": (28, 58), "Bass Tuba": (24, 58),
    "Timpani": (40, 55), "Bass Drum": (35, 59),
    "Violin": (55, 103), "Viola": (48, 91),
    "Cello": (36, 76), "Contrabass": (28, 67),
    "Harp": (24, 103), "Piano": (21, 108), "Celesta": (60, 108),
}


def quality_check(musicxml_path: str, save_pianoroll: bool = True):
    """Analyze MusicXML quality: note counts, ranges, duration errors, piano roll."""
    import music21
    s = music21.converter.parse(musicxml_path)

    base = os.path.splitext(musicxml_path)[0]
    print(f"\n{'='*70}")
    print(f"QUALITY REPORT: {os.path.basename(musicxml_path)}")
    print(f"{'='*70}")
    print(f"Parts: {len(s.parts)}\n")

    issues = []
    part_info = []
    for i, part in enumerate(s.parts):
        name = part.partName or f"Part {i+1}"
        notes = list(part.recurse().notes)
        all_midi = []
        for n in notes:
            if hasattr(n, 'pitch'):
                all_midi.append(n.pitch.midi)
            elif hasattr(n, 'pitches'):
                all_midi.extend(p.midi for p in n.pitches)

        n_measures = len(part.getElementsByClass('Measure'))
        if len(notes) == 0 and n_measures > 0:
            issues.append(f"P{i+1} {name}: EMPTY (0/{n_measures} measures — likely tremolo or other notation TrOMR cannot recognize)")

        if all_midi and name in STANDARD_RANGES:
            lo, hi = STANDARD_RANGES[name]
            below = sum(1 for p in all_midi if p < lo - 5)
            above = sum(1 for p in all_midi if p > hi + 5)
            if below:
                issues.append(f"P{i+1} {name}: {below} notes below range")
            if above:
                issues.append(f"P{i+1} {name}: {above} notes above range")

        if all_midi and len(set(all_midi)) == 1 and len(all_midi) > 3:
            pname = music21.pitch.Pitch(all_midi[0]).nameWithOctave
            issues.append(f"P{i+1} {name}: all {len(all_midi)} pitches = {pname}")

        for m in part.getElementsByClass('Measure'):
            ts = m.getContextByClass('TimeSignature')
            if ts:
                expected = ts.barDuration.quarterLength
                actual = m.duration.quarterLength
                if abs(actual - expected) > 0.5 and actual > 0:
                    issues.append(f"P{i+1} {name} m{m.number}: dur {actual} != {expected}")

        lo_n = music21.pitch.Pitch(min(all_midi)).nameWithOctave if all_midi else "-"
        hi_n = music21.pitch.Pitch(max(all_midi)).nameWithOctave if all_midi else "-"
        part_info.append((i+1, name, len(notes), lo_n, hi_n))

    for pi, name, nc, lo, hi in part_info:
        print(f"  P{pi:2d} {name:20s}  notes={nc:3d}  range={lo}-{hi}")

    print()
    if issues:
        print(f"ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues found.")

    # Piano roll
    if save_pianoroll:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        fig, ax = plt.subplots(figsize=(20, 12))
        colors = plt.cm.tab20(np.linspace(0, 1, len(s.parts)))

        all_pitches = []
        for i, part in enumerate(s.parts):
            for note in part.recurse().notes:
                onset = float(note.offset + note.getOffsetInHierarchy(part))
                dur = max(float(note.quarterLength), 0.1)
                plist = []
                if hasattr(note, 'pitch'):
                    plist = [note.pitch.midi]
                elif hasattr(note, 'pitches'):
                    plist = [p.midi for p in note.pitches]
                for midi in plist:
                    all_pitches.append(midi)
                    ax.add_patch(mpatches.Rectangle(
                        (onset, midi - 0.4), dur, 0.8,
                        facecolor=colors[i], alpha=0.7, edgecolor='black', linewidth=0.3))

        if all_pitches:
            ax.set_ylim(min(all_pitches) - 5, max(all_pitches) + 5)
        ax.autoscale_view(scalex=True, scaley=False)
        ax.set_xlabel("Beat offset (quarter notes)")
        ax.set_ylabel("MIDI pitch")
        ax.set_title(os.path.basename(musicxml_path))
        yticks = range(int(ax.get_ylim()[0]) // 6 * 6, int(ax.get_ylim()[1]) + 1, 6)
        ax.set_yticks(list(yticks))
        ax.set_yticklabels([music21.pitch.Pitch(m).nameWithOctave for m in yticks])
        ax.grid(True, alpha=0.3)
        labels = [f"P{i+1} {part.partName or ''}" for i, part in enumerate(s.parts)]
        ax.legend(handles=[mpatches.Patch(color=colors[i], label=labels[i])
                           for i in range(len(s.parts))],
                  loc='upper right', fontsize=7, ncol=2)
        plt.tight_layout()
        png_path = base + "_pianoroll.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"\nPiano roll: {png_path}")

    print(f"{'='*70}")
    return issues


# ══════════════════════════════════════════════════════════════════════════════
# Multi-page merge
# ══════════════════════════════════════════════════════════════════════════════

ORCHESTRAL_ORDER = [
    "Piccolo", "Flute", "Oboe", "English Horn",
    "Clarinet", "Bass Clarinet", "Bassoon", "Contrabassoon",
    "Horn", "Trumpet", "Trombone", "Tuba", "Bass Tuba",
    "Timpani", "Bass Drum",
    "Harp", "Celesta", "Piano",
    "Violin", "Viola", "Cello", "Contrabass",
]


def _dominant_time_sig(xml_string: str):
    """Return (beats, beat_type) of the last explicit time sig in xml_string, or None."""
    root = ET.fromstring(xml_string)
    last = None
    for t in root.iter("time"):
        try:
            last = (int(t.findtext("beats", "4")), int(t.findtext("beat-type", "4")))
        except ValueError:
            pass
    return last


def _replace_all_time_sigs(xml_string: str, beats: int, beat_type: str) -> str:
    """Replace every <time>…</time> block in xml_string with the given values."""
    import re
    return re.sub(
        r'<time>.*?</time>',
        f'<time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>',
        xml_string, flags=re.DOTALL,
    )


def _normalize_ts(ts: tuple) -> tuple:
    """Normalize half-note denominator to quarter-note equivalent: (3,2) → (6,4)."""
    beats, beat_type = ts
    return (beats * 2, 4) if beat_type == 2 else (beats, beat_type)


def _first_double_bar_measure(root) -> "int | None":
    """Return the first measure number with a double barline (majority vote across parts)."""
    from collections import Counter
    votes: Counter = Counter()
    n_parts = max(1, len(root.findall("part")))
    for part in root.findall("part"):
        seen: set = set()
        for m in part.findall("measure"):
            try:
                mnum = int(m.get("number", ""))
            except (ValueError, TypeError):
                continue
            if mnum in seen:
                continue
            seen.add(mnum)
            for bl in m.findall(".//barline"):
                if bl.findtext("bar-style", "") in ("light-light", "heavy-heavy", "light-heavy"):
                    votes[mnum] += 1
    threshold = max(1, n_parts // 2)
    candidates = sorted(m for m, c in votes.items() if c >= threshold)
    return candidates[0] if candidates else None


def _double_bar_measures(root) -> list:
    """Return all measure numbers with a double barline (majority vote across parts)."""
    from collections import Counter
    votes: Counter = Counter()
    n_parts = max(1, len(root.findall("part")))
    for part in root.findall("part"):
        seen: set = set()
        for m in part.findall("measure"):
            try:
                mnum = int(m.get("number", ""))
            except (ValueError, TypeError):
                continue
            if mnum in seen:
                continue
            seen.add(mnum)
            for bl in m.findall(".//barline"):
                if bl.findtext("bar-style", "") in ("light-light", "heavy-heavy"):
                    votes[mnum] += 1
    threshold = max(1, n_parts // 2)
    return sorted(m for m, c in votes.items() if c >= threshold)


def _infer_ts_from_measures(root, from_m: int, to_m: int) -> "tuple | None":
    """Vote for time sig from non-whole-rest measures in measure range [from_m, to_m]."""
    from collections import Counter as _Counter
    ql_to_ts = {2.0: (2, 4), 3.0: (3, 4), 4.0: (4, 4), 6.0: (6, 4)}
    qls = []
    for part in root.findall("part"):
        divs = 1
        for m in part.findall("measure"):
            try:
                mnum = int(m.get("number", ""))
            except (ValueError, TypeError):
                continue
            # Update divs from ALL measures (not just in-range ones), so that
            # divisions declared in m1 are correctly applied to m3+ measures.
            d_el = m.findtext(".//divisions")
            if d_el:
                try:
                    divs = int(d_el)
                except ValueError:
                    pass
            if not (from_m <= mnum <= to_m):
                continue
            notes = m.findall("note")
            if not any(n.find("pitch") is not None for n in notes):
                continue  # all-rest measure: skip
            note_sum = 0
            for n in notes:
                if n.find("chord") is not None:
                    continue
                try:
                    note_sum += int(n.findtext("duration", "0"))
                except ValueError:
                    pass
            if note_sum > 0:
                qls.append(note_sum / divs)
    if not qls:
        return None
    votes = _Counter(round(ql * 2) / 2 for ql in qls)
    best = votes.most_common(1)[0][0]
    return ql_to_ts.get(best, (round(best), 4))


def _retry_tromr_post_double(staffs, bar_line_boxes, multi_staffs,
                              debug, preprocessed_image, xml_string,
                              transformer_config,
                              full_res_image=None, homr_to_full_scale=1.0,
                              use_vlm=False):
    """Re-run TrOMR on the post-double-barline region when TrOMR missed a time sig change.

    If a mid-system double barline exists but the time sig before and after is the same
    (indicating TrOMR failed to read the new printed time sig), crop each staff's grid to
    start just before the double barline and re-run parse_staffs on the cropped region.
    If the retry produces a different time sig, inject it into the post-double measures.
    """
    from collections import Counter as _C
    import statistics as _stats
    from homr.staff_detection import Staff as _Staff
    from homr.model import MultiStaff as _MultiStaff
    from homr.staff_parsing import parse_staffs as _parse_staffs
    from homr.music_xml_generator import generate_xml as _gen_xml, XmlGeneratorArguments as _XmlArgs

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return xml_string

    first_dbl = _first_double_bar_measure(root)
    if first_dbl is None:
        return xml_string

    try:
        m_end = max(
            int(m.get("number", "0"))
            for part in root.findall("part")
            for m in part.findall("measure")
            if (m.get("number", "") or "").isdigit()
        )
    except ValueError:
        return xml_string

    if first_dbl >= m_end - 1:
        return xml_string  # Only a single phantom measure after barline — skip retry

    def _majority_ts(from_m, to_m):
        v = _C()
        for part in root.findall("part"):
            for m in part.findall("measure"):
                try:
                    mn = int(m.get("number", ""))
                except (ValueError, TypeError):
                    continue
                if not (from_m <= mn <= to_m):
                    continue
                t = m.find(".//time")
                if t is None:
                    continue
                try:
                    bt = int(t.findtext("beat-type", "4"))
                    if bt == 1:
                        continue
                    v[(int(t.findtext("beats", "4")), bt)] += 1
                except ValueError:
                    pass
        return v.most_common(1)[0][0] if v else None

    ts_before = _majority_ts(1, first_dbl)

    # If TrOMR already has a clear majority time sig in the post-double segment
    # (>50% of part-votes agree), trust it — no VLM/retry needed.
    def _majority_ts_conf(from_m, to_m):
        v = _C()
        for part in root.findall("part"):
            for m in part.findall("measure"):
                try:
                    mn = int(m.get("number", ""))
                except (ValueError, TypeError):
                    continue
                if not (from_m <= mn <= to_m):
                    continue
                t = m.find(".//time")
                if t is None:
                    continue
                try:
                    bt = int(t.findtext("beat-type", "4"))
                    if bt == 1:
                        continue
                    v[(int(t.findtext("beats", "4")), bt)] += 1
                except ValueError:
                    pass
        if not v:
            return None, 0.0
        best, cnt = v.most_common(1)[0]
        return best, cnt / sum(v.values())

    # Trim phantom trailing measures before voting: skip 2+ consecutive sparse
    # tail measures (< 1/3 of parts have notes) so they don't dilute confidence.
    _all_parts_r = root.findall("part")
    _n_parts_r = len(_all_parts_r)
    _thresh_r = max(1, _n_parts_r // 3)
    _eff_m_end = m_end
    _sparse_run = 0
    _probe = m_end
    while _probe > first_dbl + 1:
        _pwn = sum(
            1 for _p in _all_parts_r
            if any(
                (m.get("number", "") or "").isdigit()
                and int(m.get("number", "0")) == _probe
                and m.findall("note")
                for m in _p.findall("measure")
            )
        )
        if _pwn >= _thresh_r:
            break
        _sparse_run += 1
        _probe -= 1
    if _sparse_run >= 2:
        _eff_m_end = _probe
        print(f"[VLM-TimeSig] phantom trim: m_end {m_end}→{_eff_m_end} ({_sparse_run} sparse tail)")

    ts_post, ts_post_conf = _majority_ts_conf(first_dbl + 1, _eff_m_end)
    if ts_post is not None and ts_post_conf > 0.75:
        print(f"[VLM-TimeSig] post-double majority {ts_post[0]}/{ts_post[1]} "
              f"(conf={ts_post_conf:.0%}) — trusting TrOMR, skipping retry")
        return xml_string

    # Locate consensus double barline x in HOMR coordinate space
    all_xs = []
    for s in staffs:
        xs = detect_double_barlines(s, bar_line_boxes)
        if xs:
            all_xs.extend(xs)
    if not all_xs:
        return xml_string

    # Cluster xs (gap > 100 px) and pick the first cluster that has notes on both sides.
    _xs_s = sorted(all_xs)
    _clusters, _cur = [], [_xs_s[0]]
    for _x in _xs_s[1:]:
        if _x - _cur[-1] > 100:
            _clusters.append(_cur); _cur = [_x]
        else:
            _cur.append(_x)
    _clusters.append(_cur)
    _all_note_xs = [sym.center[0]
                    for s in staffs for sym in s.symbols
                    if hasattr(sym, "center")]
    dbl_x = _stats.median(all_xs)  # fallback
    for _cl in _clusters:
        _cx = _stats.median(_cl)
        if any(_nx < _cx for _nx in _all_note_xs) and any(_nx > _cx for _nx in _all_note_xs):
            dbl_x = _cx
            break

    # ── VLM: primary method — ask first before trusting TrOMR ──
    # If VLM identifies a numeric ts → apply it.
    # If VLM says no ts visible → no change (overrides TrOMR's reading).
    # If VLM errors → fall through to TrOMR path.
    if use_vlm and full_res_image is not None:
        try:
            from PIL import Image as _PILImage

            # Crop from double barline to first note after it (HOMR coords → full-res).
            # Note: s.symbols contains homr.model.Note objects; use .center[0], not .notehead
            note_xs = [
                sym.center[0]
                for s in staffs
                for sym in s.symbols
                if hasattr(sym, "center") and sym.center[0] > dbl_x
            ]
            first_note_x = min(note_xs) if note_xs else dbl_x + 150

            x1 = max(0, int(dbl_x * homr_to_full_scale))
            x2 = min(full_res_image.shape[1], int(first_note_x * homr_to_full_scale) + 10)

            # Sample up to 5 staves, query VLM on each, take majority vote.
            import random as _random
            _staffs_sorted = sorted(staffs, key=lambda s: s.min_y)
            _sample = (_staffs_sorted if len(_staffs_sorted) <= 5
                       else _random.sample(_staffs_sorted, 5))
            _votes: dict = {}
            for _st in _sample:
                _y1 = max(0, int(_st.min_y * homr_to_full_scale) - 10)
                _y2 = min(full_res_image.shape[0],
                          int(_st.max_y * homr_to_full_scale) + 10)
                if x2 <= x1 or _y2 <= _y1:
                    continue
                _crop = full_res_image[_y1:_y2, x1:x2]
                _pil = _PILImage.fromarray(
                    _crop[:, :, ::-1] if _crop.ndim == 3 else _crop)
                _r = _vlm_read_time_sig(_pil)
                _key = _r if _r is not None else "none"
                _votes[_key] = _votes.get(_key, 0) + 1
            print(f"[VLM-TimeSig] votes: {_votes}")
            # Majority winner; treat "none" as a valid vote
            _winner = max(_votes, key=_votes.__getitem__)
            vlm_ts = None if _winner == "none" else _winner

            if vlm_ts is not None:
                # VLM identified a numeric ts
                if vlm_ts != ts_before:
                    print(f"[VLM-TimeSig] {ts_before[0] if ts_before else '?'}"
                          f"/{ts_before[1] if ts_before else '?'}"
                          f" → {vlm_ts[0]}/{vlm_ts[1]} at m{first_dbl + 1}")
                    _apply_ts_to_segment(root, vlm_ts, first_dbl + 1, m_end)
                    return ET.tostring(root, encoding="unicode", xml_declaration=False)
                else:
                    print(f"[VLM-TimeSig] No change (VLM confirmed {vlm_ts[0]}/{vlm_ts[1]}) — restoring ts_before")
                    if ts_before is not None:
                        _apply_ts_to_segment(root, ts_before, first_dbl + 1, m_end)
                        return ET.tostring(root, encoding="unicode", xml_declaration=False)
                    return xml_string
            else:
                # VLM sees no numeric ts symbol → same ts as before the barline.
                # Explicitly write ts_before to post-double measures so that any
                # incorrect TrOMR reading (e.g. false 3/4 on p185) is overridden.
                print(f"[VLM-TimeSig] No change (no ts symbol visible) — restoring ts_before")
                if ts_before is not None:
                    _apply_ts_to_segment(root, ts_before, first_dbl + 1, m_end)
                    return ET.tostring(root, encoding="unicode", xml_declaration=False)
                else:
                    # ts_before unknown (e.g. pickup bar has no <time> element).
                    # VLM confirmed no ts change → clear HOMR's wrong reading so
                    # _cross_part_post_process falls back to the uniform default.
                    for _part in root.findall("part"):
                        for _m in _part.findall("measure"):
                            try:
                                _mn = int(_m.get("number", ""))
                            except (ValueError, TypeError):
                                continue
                            if not (first_dbl + 1 <= _mn <= m_end):
                                continue
                            _attrs = _m.find("attributes")
                            if _attrs is not None:
                                for _t in list(_attrs.findall("time")):
                                    _attrs.remove(_t)
                    return ET.tostring(root, encoding="unicode", xml_declaration=False)
        except Exception as e:
            print(f"[VLM-TimeSig] Error: {e} — falling back to TrOMR")

    # ── TrOMR fallback (VLM unavailable or errored) ──
    ts_after = _majority_ts(first_dbl + 1, m_end)
    if ts_after is not None and ts_after != ts_before:
        return xml_string  # TrOMR already detected a change

    margin = (staffs[0].average_unit_size if staffs else 10) * 1.5

    # Build cropped MultiStaff objects: keep grid points at x ≥ (dbl_x - margin)
    new_multi = []
    for ms in multi_staffs:
        cropped = []
        for s in ms.staffs:
            new_grid = [p for p in s.grid if p.x >= dbl_x - margin]
            if len(new_grid) < 3:
                continue
            ns = _Staff(new_grid)
            # symbols intentionally empty — TrOMR reads from the image directly
            cropped.append(ns)
        if cropped:
            new_multi.append(_MultiStaff(cropped, []))

    if not new_multi:
        return xml_string

    print(f"[TrOMR-retry] Re-running on post-double region (x≥{dbl_x:.0f})...")
    try:
        retry_result = _parse_staffs(debug, new_multi, preprocessed_image, transformer_config)
        retry_root_obj = _gen_xml(_XmlArgs(), retry_result, "retry")
        retry_root = ET.fromstring(retry_root_obj.to_string())
    except Exception as e:
        print(f"[TrOMR-retry] Failed: {e}")
        return xml_string

    # Majority vote on time sig from first measure of retry output
    v = _C()
    for part in retry_root.findall("part"):
        ms_list = part.findall("measure")
        if ms_list:
            t = ms_list[0].find(".//time")
            if t is not None:
                try:
                    bt = int(t.findtext("beat-type", "4"))
                    if bt != 1:
                        v[(int(t.findtext("beats", "4")), bt)] += 1
                except ValueError:
                    pass

    new_ts = v.most_common(1)[0][0] if v else None
    if new_ts is not None and new_ts != ts_before:
        print(f"[TrOMR-retry] {ts_before[0] if ts_before else '?'}/{ts_before[1] if ts_before else '?'}"
              f" → {new_ts[0]}/{new_ts[1]} at m{first_dbl + 1}")
        _apply_ts_to_segment(root, new_ts, first_dbl + 1, m_end)
        return ET.tostring(root, encoding="unicode", xml_declaration=False)

    print(f"[TrOMR-retry] No change detected")
    return xml_string


def _apply_ts_to_segment(root, ts: tuple, from_m: int, to_m: int) -> None:
    """Set time sig in first measure of [from_m, to_m] per part; remove it from others."""
    beats, beat_type = ts
    for part in root.findall("part"):
        first = True
        for m in part.findall("measure"):
            try:
                mnum = int(m.get("number", ""))
            except (ValueError, TypeError):
                continue
            if not (from_m <= mnum <= to_m):
                continue
            attrs = m.find("attributes")
            if attrs is not None:
                for t in attrs.findall("time"):
                    attrs.remove(t)
            if first:
                if attrs is None:
                    attrs = ET.Element("attributes")
                    m.insert(0, attrs)
                time_el = ET.SubElement(attrs, "time")
                ET.SubElement(time_el, "beats").text = str(beats)
                ET.SubElement(time_el, "beat-type").text = str(beat_type)
                first = False


def _get_ks_contexts(root) -> list:
    """Return [(part_name, chromatic_transpose, fifths), ...] from first-measure key sigs."""
    name_map = {}
    part_list = root.find("part-list")
    if part_list is not None:
        for sp in part_list.findall("score-part"):
            pid = sp.get("id", "")
            name = sp.findtext("part-name") or sp.findtext(".//instrument-name") or ""
            name_map[pid] = name

    contexts = []
    for part in root.findall("part"):
        pid = part.get("id", "")
        fifths = None
        chromatic = 0
        for m in part.findall("measure"):
            for attrs in m.findall("attributes"):
                k = attrs.find("key")
                if k is not None and fifths is None:
                    try:
                        fifths = int(k.findtext("fifths", "0"))
                    except ValueError:
                        fifths = 0
                t = attrs.find("transpose")
                if t is not None:
                    try:
                        chromatic = int(t.findtext("chromatic", "0"))
                    except ValueError:
                        pass
        if fifths is not None:
            contexts.append((name_map.get(pid, ""), chromatic, fifths))
    return contexts


def _apply_ks_contexts(root, contexts: list, from_m: int, to_m: int) -> None:
    """Apply key sigs from contexts to matching parts (matched by name+chromatic_transpose).

    Each matching part gets its key sig element replaced in the first measure of the range
    and removed from subsequent measures in [from_m, to_m].
    When a part has no name (instrument label absent from system start), falls back to
    positional matching: context[i] → part[i].
    """
    lookup = {(name, chrom): fifths for name, chrom, fifths in contexts}
    pos_fifths = [fifths for _, _, fifths in contexts]
    name_map = {}
    part_list = root.find("part-list")
    if part_list is not None:
        for sp in part_list.findall("score-part"):
            pid = sp.get("id", "")
            name = sp.findtext("part-name") or sp.findtext(".//instrument-name") or ""
            name_map[pid] = name

    for part_idx, part in enumerate(root.findall("part")):
        pid = part.get("id", "")
        part_name = name_map.get(pid, "")
        chromatic = 0
        for m in part.findall("measure"):
            for attrs in m.findall("attributes"):
                t = attrs.find("transpose")
                if t is not None:
                    try:
                        chromatic = int(t.findtext("chromatic", "0"))
                    except ValueError:
                        pass
            break

        key = (part_name, chromatic)
        if key in lookup:
            fifths = lookup[key]
        elif not part_name and part_idx < len(pos_fifths):
            fifths = pos_fifths[part_idx]
        else:
            continue

        first = True
        for m in part.findall("measure"):
            try:
                mnum = int(m.get("number", ""))
            except (ValueError, TypeError):
                continue
            if not (from_m <= mnum <= to_m):
                continue
            attrs = m.find("attributes")
            if attrs is not None:
                for k_el in attrs.findall("key"):
                    attrs.remove(k_el)
            if first:
                if attrs is None:
                    attrs = ET.Element("attributes")
                    m.insert(0, attrs)
                key_el = ET.SubElement(attrs, "key")
                ET.SubElement(key_el, "fifths").text = str(fifths)
                first = False


def _fixup_ks_in_file(path: str, contexts: list, from_m: int, to_m: int) -> None:
    """Read a MusicXML file, apply key sig contexts to [from_m, to_m], and rewrite."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            xml_string = f.read()
        root = ET.fromstring(xml_string)
        _apply_ks_contexts(root, contexts, from_m, to_m)
        out = ET.tostring(root, encoding="unicode", xml_declaration=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"[KeySig] fixup {path} m{from_m}-{to_m}: applied {len(contexts)} parts")
    except Exception as e:
        print(f"[KeySig] fixup failed for {path}: {e}")


def _merge_canonical_name(name: str) -> str:
    """Normalize a part name for merge matching.

    - Standalone 'B' → 'Contrabass' (German score shorthand)
    - Expands abbreviations for unknown names ('Ci.Es' → 'Clarinet in Eb')
    - Strips trailing Roman/Arabic numbering ('Violin I' → 'Violin')
    - Fills in default transposition key ('Horn' → 'Horn in F')
    - Returns display form ('Clarinet:Eb' → 'Clarinet in Eb')
    """
    # Standalone single-letter special case: 'B' = Kontrabass in German scores
    if name.strip() in ("B", "b"):
        return "Contrabass"

    work_name = name
    if _instrument_base(name) not in INSTRUMENT_MIDI:
        # Try abbreviation expansion for unknown names (e.g. 'Ci.Es', 'Tam-tam')
        normalized = _normalize_instrument_name(name)
        if _instrument_base(normalized) in INSTRUMENT_MIDI:
            orig_base, orig_key = _parse_instrument_key(name)
            norm_base, norm_key = _parse_instrument_key(normalized)
            # Preserve key if normalization stripped it ('Clarinet in A' → keep 'in A')
            if orig_key and not norm_key:
                work_name = f"{norm_base} in {orig_key}"
            else:
                work_name = normalized

    # Strip trailing Roman numerals or Arabic numbers ("Violin I" → "Violin")
    n = re.sub(r'\s+(?:[IVX]+|\d+)$', '', work_name).strip()
    instr_base_n, key_n = _parse_instrument_key(n)
    canonical_base = _instrument_base(n)
    if canonical_base not in INSTRUMENT_MIDI:
        return name  # unknown instrument, leave as-is

    # Fill in default transposition key when omitted ("Horn" → "Horn in F")
    if key_n is None:
        default_key = DEFAULT_TRANSPOSE_KEY.get(canonical_base)
        if default_key:
            n = f"{instr_base_n}:{default_key}"

    # Always return display form (normalises 'Clarinet:Eb' → 'Clarinet in Eb')
    return _display_name(n)


def merge_pages(page_xmls: List[str], output_path: str):
    """Merge multiple per-page MusicXML files into a single score."""
    from math import gcd

    def _lcm(a, b):
        return a * b // gcd(a, b)

    page_data = []
    for pxml in page_xmls:
        tree = ET.parse(pxml)
        root = tree.getroot()
        parts = root.findall("part")
        info = []
        for part in parts:
            pid = part.get("id")
            sp = root.find(f'.//score-part[@id="{pid}"]')
            name = sp.findtext("part-name", "?") if sp is not None else "?"
            name = _merge_canonical_name(name)
            measures = part.findall("measure")
            info.append({"name": name, "part": part, "score_part": sp,
                         "measures": measures, "n_measures": len(measures)})
        page_data.append(info)

    # Build occurrence-keyed part list per page: (name, occ_idx) → part info
    page_keys = []
    for pinfo in page_data:
        occ_count = {}
        keyed = {}
        for pi in pinfo:
            name = pi["name"]
            idx = occ_count.get(name, 0)
            occ_count[name] = idx + 1
            keyed[(name, idx)] = pi
        page_keys.append(keyed)

    # Collect all unique instruments — use max occurrence count
    all_instruments = {}
    for pinfo in page_data:
        counts = {}
        for pi in pinfo:
            counts[pi["name"]] = counts.get(pi["name"], 0) + 1
        for name, cnt in counts.items():
            all_instruments[name] = max(all_instruments.get(name, 0), cnt)

    # Build sorted master part list
    order_map = {name: i for i, name in enumerate(ORCHESTRAL_ORDER)}
    master_parts = []
    for name, max_occ in sorted(all_instruments.items(),
                                 key=lambda x: order_map.get(_instrument_base(x[0]), 99)):
        for occ in range(max_occ):
            master_parts.append((name, occ))

    print(f"[Merge] {len(master_parts)} parts across {len(page_xmls)} pages")
    _ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
    for name, occ in master_parts:
        suffix = f" {_ROMAN.get(occ+1, occ+1)}" if all_instruments[name] > 1 else ""
        pages_present = [i+1 for i, pk in enumerate(page_keys) if (name, occ) in pk]
        print(f"  {name}{suffix}: present on pages {pages_present}")

    # Compute global target divisions = LCM of all divisions across all pages/parts
    all_divs = set()
    for pinfo in page_data:
        for pi in pinfo:
            for m in pi["measures"]:
                d = m.findtext(".//divisions")
                if d:
                    try:
                        all_divs.add(int(d))
                    except ValueError:
                        pass
    target_divs = 1
    for d in all_divs:
        target_divs = _lcm(target_divs, d)
    target_divs = min(target_divs, 96)
    print(f"[Merge] Normalizing divisions to {target_divs} (from {sorted(all_divs)})")

    # Collect time signatures per page, per measure offset.
    # Rules:
    #   1. Only count votes from measures with an explicit <time> element, OR that contain
    #      actual pitched notes (not just whole rests — a whole rest looks the same in 6/4 and 4/4).
    #   2. Parts that never show an explicit <time> on this page don't vote (they carry no
    #      information about which time sig applies here).
    #   3. For measure offsets with zero qualifying votes, inherit from the most recent
    #      known time sig — either the previous measure on this page, or the last time sig
    #      from the preceding page/system.
    from collections import Counter
    page_measure_ts = []
    for pinfo in page_data:
        n = max(pi["n_measures"] for pi in pinfo) if pinfo else 0
        explicit_votes: dict = {}   # mi → [(beats, beat_type), ...]
        for pi in pinfo:
            cur = None  # None = no explicit <time> seen yet for this part
            for mi, m in enumerate(pi["measures"]):
                t = m.find(".//time")
                if t is not None:
                    try:
                        cur = (int(t.findtext("beats", "4")),
                               int(t.findtext("beat-type", "4")))
                        # Explicit <time> tag → always vote, regardless of note content
                        explicit_votes.setdefault(mi, []).append(cur)
                        continue
                    except ValueError:
                        pass
                # No <time> in this measure
                if cur is None:
                    continue  # Part never had explicit time sig → skip
                # cur was established earlier; only vote if this measure has pitched notes
                # (whole rests cannot distinguish e.g. 6/4 from 4/4)
                has_pitch = any(n.find("pitch") is not None
                                for n in m.findall(".//note"))
                if has_pitch:
                    explicit_votes.setdefault(mi, []).append(cur)

        # Propagate: start from the last time sig of the preceding page/system
        prev_ts = page_measure_ts[-1][-1] if page_measure_ts else (4, 4)
        ts_list = []
        cur_ts = prev_ts
        for mi in range(n):
            if mi in explicit_votes:
                cur_ts = Counter(explicit_votes[mi]).most_common(1)[0][0]
            ts_list.append(cur_ts)
        page_measure_ts.append(ts_list if ts_list else [cur_ts])

    # Build merged XML
    merged_root = ET.Element("score-partwise")
    ET.SubElement(merged_root, "defaults")
    part_list_el = ET.SubElement(merged_root, "part-list")

    for mi, (name, occ) in enumerate(master_parts):
        pid = f"P{mi+1}"
        dn = _display_name(name)
        _ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
        display_name = f"{dn} {_ROMAN.get(occ+1, occ+1)}" if all_instruments[name] > 1 else dn
        sp = ET.SubElement(part_list_el, "score-part", id=pid)
        ET.SubElement(sp, "part-name").text = display_name
        si = ET.SubElement(sp, "score-instrument", id=f"{pid}-I1")
        ET.SubElement(si, "instrument-name").text = dn
        sound, midi_prog = INSTRUMENT_MIDI.get(_instrument_base(name), ("keyboard.piano", 1))
        ET.SubElement(si, "instrument-sound").text = sound
        midi_el = ET.SubElement(sp, "midi-instrument", id=f"{pid}-I1")
        ET.SubElement(midi_el, "midi-channel").text = "1"
        ET.SubElement(midi_el, "midi-program").text = str(midi_prog)
        ET.SubElement(midi_el, "volume").text = "100"
        ET.SubElement(midi_el, "pan").text = "0"

    # Detect pickup (anacrusis) in first page's first measure
    pickup_ticks = None
    beats_0, bt_0 = page_measure_ts[0][0]
    full_m1 = target_divs * beats_0 * 4 // bt_0
    for pi in page_data[0]:
        if not pi["measures"]:
            continue
        m1 = pi["measures"][0]
        m1_divs, pos, max_pos = 1, 0, 0
        for el in m1:
            if el.tag == "attributes":
                d = el.findtext("divisions")
                if d:
                    try: m1_divs = int(d)
                    except ValueError: pass
            elif el.tag == "note" and el.find("chord") is None:
                d = el.findtext("duration")
                if d:
                    pos += round(int(d) * target_divs / m1_divs)
                    max_pos = max(max_pos, pos)
            elif el.tag == "backup":
                d = el.findtext("duration")
                if d: pos -= round(int(d) * target_divs / m1_divs)
            elif el.tag == "forward":
                d = el.findtext("duration")
                if d:
                    pos += round(int(d) * target_divs / m1_divs)
                    max_pos = max(max_pos, pos)
        if 0 < max_pos < full_m1:
            pickup_ticks = max_pos
            print(f"[Merge] Pickup detected: {pickup_ticks}/{full_m1} ticks")
            break

    # Concatenate measures for each master part, normalizing divisions
    for mi, (name, occ) in enumerate(master_parts):
        pid = f"P{mi+1}"
        part_el = ET.SubElement(merged_root, "part", id=pid)
        measure_num = 1

        for page_idx, pk in enumerate(page_keys):
            page_info = page_data[page_idx]
            n_measures = max(pi["n_measures"] for pi in page_info) if page_info else 0
            pg_ts = page_measure_ts[page_idx]  # list of (beats, beat_type) per measure offset

            if (name, occ) in pk:
                pi = pk[(name, occ)]
                cur_divs = 1
                for page_mi, m in enumerate(pi["measures"]):
                    new_m = _deep_copy_element(m)
                    new_m.set("number", str(measure_num))
                    # update divisions tracking
                    d_el = new_m.find(".//divisions")
                    if d_el is not None:
                        try:
                            cur_divs = int(d_el.text)
                        except (ValueError, TypeError):
                            pass
                    # rescale all durations to target_divs
                    if cur_divs != target_divs:
                        scale = target_divs / cur_divs
                        for dur_el in _iter_duration_elements(new_m):
                            try:
                                old = int(dur_el.text)
                                dur_el.text = str(round(old * scale))
                            except (ValueError, TypeError):
                                pass
                    # set divisions to target in attributes; inject time sig when needed
                    cur_ts = pg_ts[page_mi] if page_mi < len(pg_ts) else (4, 4)
                    is_page_boundary = (measure_num == 1 or page_mi == 0)
                    ts_changed = (page_mi > 0 and pg_ts[page_mi] != pg_ts[page_mi - 1])
                    attrs = new_m.find("attributes")
                    if attrs is not None:
                        d_el = attrs.find("divisions")
                        if d_el is not None:
                            d_el.text = str(target_divs)
                        if (is_page_boundary or ts_changed) and attrs.find("time") is None:
                            time_el = ET.SubElement(attrs, "time")
                            ET.SubElement(time_el, "beats").text = str(cur_ts[0])
                            ET.SubElement(time_el, "beat-type").text = str(cur_ts[1])
                    elif is_page_boundary:
                        attrs = ET.Element("attributes")
                        ET.SubElement(attrs, "divisions").text = str(target_divs)
                        time_el = ET.Element("time")
                        ET.SubElement(time_el, "beats").text = str(cur_ts[0])
                        ET.SubElement(time_el, "beat-type").text = str(cur_ts[1])
                        attrs.append(time_el)
                        new_m.insert(0, attrs)
                    elif ts_changed:
                        new_attrs = ET.Element("attributes")
                        time_el = ET.Element("time")
                        ET.SubElement(time_el, "beats").text = str(cur_ts[0])
                        ET.SubElement(time_el, "beat-type").text = str(cur_ts[1])
                        new_attrs.append(time_el)
                        new_m.insert(0, new_attrs)
                    part_el.append(new_m)
                    measure_num += 1
                for filler_mi in range(n_measures - pi["n_measures"]):
                    offset = pi["n_measures"] + filler_mi
                    f_ts = pg_ts[offset] if offset < len(pg_ts) else (4, 4)
                    part_el.append(_make_rest_measure(measure_num, target_divs, *f_ts))
                    measure_num += 1
            else:
                prev_page_last_ts = (page_measure_ts[page_idx - 1][-1]
                                     if page_idx > 0 and page_measure_ts[page_idx - 1]
                                     else (4, 4))
                for mi2 in range(n_measures):
                    cur_ts = pg_ts[mi2] if mi2 < len(pg_ts) else (4, 4)
                    prev_ts = pg_ts[mi2 - 1] if mi2 > 0 else prev_page_last_ts
                    ts_changed = (cur_ts != prev_ts)
                    include_attrs = (measure_num == 1 or mi2 == 0 or ts_changed)
                    dur_ovr = pickup_ticks if (measure_num == 1 and pickup_ticks is not None) else None
                    part_el.append(_make_rest_measure(
                        measure_num, target_divs, *cur_ts,
                        include_attrs=include_attrs,
                        duration_override=dur_ovr))
                    measure_num += 1

        # ensure first measure always has attributes/divisions
        first_m = part_el.find("measure")
        if first_m is not None:
            attrs = first_m.find("attributes")
            if attrs is None:
                attrs = ET.Element("attributes")
                ET.SubElement(attrs, "divisions").text = str(target_divs)
                first_m.insert(0, attrs)
            elif attrs.find("divisions") is None:
                d_el = ET.Element("divisions")
                d_el.text = str(target_divs)
                attrs.insert(0, d_el)
            if attrs.find("transpose") is None:
                _inject_transpose(attrs, name)

    # Post-merge type-duration alignment
    overflow_n = _fix_notated_overflow(merged_root)
    if overflow_n:
        print(f"[Merge] Notated overflow: removed {overflow_n} excess rests")
    td_fixes = _fix_type_duration_alignment(merged_root, label="merge-")
    if td_fixes:
        print(f"[Merge] Type-duration alignment: {len(td_fixes)} fixes")

    # Post-merge per-voice duration fix
    vfix_count = _fix_voice_durations(merged_root)
    if vfix_count:
        print(f"[Merge] Voice duration fixes: {vfix_count}")

    # Post-merge triplet marking
    trip_count = _mark_unmarked_triplets(merged_root)
    if trip_count:
        print(f"[Merge] Triplet marking: {trip_count} notes")

    # Final type-duration alignment (cleans up mismatches created by V-fix)
    overflow_n2 = _fix_notated_overflow(merged_root)
    if overflow_n2:
        print(f"[Merge] Final overflow: removed {overflow_n2} excess rests")
    td_fixes2 = _fix_type_duration_alignment(merged_root, label="final-")
    if td_fixes2:
        print(f"[Merge] Final type-duration alignment: {len(td_fixes2)} fixes")

    xml_string = ET.tostring(merged_root, encoding="unicode", xml_declaration=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_string)
    print(f"[Merge] Written: {output_path}")
    return output_path


def _iter_duration_elements(measure):
    """Yield all <duration> elements inside notes, backups, and forwards."""
    for child in measure:
        if child.tag in ("note", "backup", "forward"):
            d = child.find("duration")
            if d is not None:
                yield d


def _deep_copy_element(elem):
    """Deep copy an XML element."""
    new = ET.Element(elem.tag, elem.attrib)
    new.text = elem.text
    new.tail = elem.tail
    for child in elem:
        new.append(_deep_copy_element(child))
    return new


def _make_rest_measure(number, divs, beats=4, beat_type=4, include_attrs=True,
                       duration_override=None):
    """Create an empty rest measure with correct divisions and time signature."""
    m = ET.Element("measure", number=str(number))
    if include_attrs:
        attrs = ET.SubElement(m, "attributes")
        ET.SubElement(attrs, "divisions").text = str(divs)
        time_el = ET.SubElement(attrs, "time")
        ET.SubElement(time_el, "beats").text = str(beats)
        ET.SubElement(time_el, "beat-type").text = str(beat_type)
    rest = ET.SubElement(m, "note")
    ET.SubElement(rest, "rest")
    if duration_override is not None:
        rest_dur = duration_override
    else:
        rest_dur = divs * beats * 4 // beat_type
    ET.SubElement(rest, "duration").text = str(rest_dur)
    ql = rest_dur / divs
    rest_type = _TD_QL_TO_TYPE.get(ql, "whole")
    ET.SubElement(rest, "type").text = rest_type
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(img_path: str, output_path: str, use_gpu: bool = True, use_vlm: bool = True,
                 tremolo_templates: str = None, part_names_override: List[str] = None,
                 plugin_output: str = None, collect_plugin_data: bool = False,
                 page_index: int = 0, ts_context=None):
    """Full pipeline: image → MusicXML.
    Returns (output_path, part_names, ts_context) where
    ts_context=(last_ts, ended_with_bar, last_ks, ks_changed, pending_ks_fixup) is suitable
    for passing to the next page's run_pipeline call. pending_ks_fixup, if non-None, is
    (file_path, from_m, to_m) — a post-double key-sig range in the current output that will
    be retroactively corrected once the next page's starting key is known."""
    print(f"\n{'='*60}")
    print(f"Processing: {img_path}")
    print(f"{'='*60}")
    t_start = time.time()

    results = run_homr_pipeline(
        img_path, use_gpu=use_gpu, use_vlm=use_vlm,
        tremolo_templates=tremolo_templates,
        part_names_override=part_names_override,
        collect_plugin_data=collect_plugin_data or plugin_output is not None,
        page_index=page_index,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    final_names = None
    plugin_pages = []
    final_xml_string = None
    ctx = ts_context if ts_context else (None, False, None, False, None)
    prev_ts, prev_bar_ended, prev_ks, prev_ks_changed, pending_ks_fixup = ctx

    if len(results) == 1:
        xml_string, part_names, plugin_page = results[0]
        if not part_names and part_names_override is not None:
            part_names = list(part_names_override)
            print(f"[Override] No labels on this page, reusing: {len(part_names)} instruments")
        xml_string = _inject_part_names(xml_string, part_names)
        incoming_ks_changed = prev_ks_changed
        xml_string, prev_ts, prev_bar_ended, prev_ks, prev_ks_changed, ks_fixup_range = \
            _cross_part_post_process(xml_string, prev_ts, prev_bar_ended, prev_ks, prev_ks_changed)
        if incoming_ks_changed and pending_ks_fixup is not None and prev_ks:
            _fixup_ks_in_file(pending_ks_fixup[0], prev_ks, pending_ks_fixup[1], pending_ks_fixup[2])
            pending_ks_fixup = None
        pending_ks_fixup = (output_path,) + ks_fixup_range if ks_fixup_range else pending_ks_fixup
        final_xml_string = xml_string
        if plugin_page is not None:
            plugin_pages.append(plugin_page)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_string)
        final_names = part_names
    else:
        temp_files = []
        system_master_names = None
        for sys_idx, (xml_string, sys_names, plugin_page) in enumerate(results):
            if sys_idx == 0:
                final_names = sys_names
                system_master_names = list(part_names_override or sys_names or [])
            else:
                master_names = list(system_master_names or [])
                if not sys_names and master_names:
                    sys_names = list(master_names)
                    print(f"[Override] System {sys_idx+1}: no labels, reusing: {len(sys_names)} instruments")
            xml_string = _inject_part_names(xml_string, sys_names)
            incoming_ks_changed = prev_ks_changed
            xml_string, prev_ts, prev_bar_ended, prev_ks, prev_ks_changed, ks_fixup_range = \
                _cross_part_post_process(xml_string, prev_ts, prev_bar_ended, prev_ks, prev_ks_changed)
            base, ext = os.path.splitext(output_path)
            sys_path = f"{base}_sys{sys_idx}{ext}"
            # Retroactive fixup for previous system's post-double range.
            if incoming_ks_changed and pending_ks_fixup is not None and prev_ks:
                _fixup_ks_in_file(pending_ks_fixup[0], prev_ks, pending_ks_fixup[1], pending_ks_fixup[2])
                pending_ks_fixup = None
            pending_ks_fixup = (sys_path,) + ks_fixup_range if ks_fixup_range else pending_ks_fixup
            with open(sys_path, "w", encoding="utf-8") as f:
                f.write(xml_string)
            temp_files.append(sys_path)
            if plugin_page is not None:
                if plugin_pages:
                    plugin_pages[0].notes.extend(plugin_page.notes)
                else:
                    plugin_pages.append(plugin_page)
            print(f"[MultiSys] System {sys_idx + 1}: {sys_path}")
        merge_pages(temp_files, output_path)
        final_xml_string = Path(output_path).read_text(encoding="utf-8")

    if plugin_output is not None and final_xml_string is not None:
        plugin_xml_path = write_plugin_output(
            plugin_output,
            musicxml_path=output_path,
            xml_string=final_xml_string,
            plugin_pages=plugin_pages,
        )
        print(f"[Plugin] score.musicxml: {plugin_xml_path}")

    elapsed = time.time() - t_start
    print(f"\n[Done] {output_path} ({elapsed:.1f}s)")
    ts_ctx = (prev_ts, prev_bar_ended, prev_ks, prev_ks_changed, pending_ks_fixup)
    if collect_plugin_data:
        return output_path, final_names, ts_ctx, plugin_pages, final_xml_string
    return output_path, final_names, ts_ctx


def main():
    parser = argparse.ArgumentParser(
        description="OMR Pipeline: HOMR recognition + OCR instrument names → MusicXML"
    )
    parser.add_argument("input", nargs="+",
                        help="Image file(s) (.png/.jpg) or directory of images. "
                             "Multiple files are processed and merged into one score.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .musicxml path (or directory for batch)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU inference")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM instrument OCR (use RapidOCR fallback)")
    parser.add_argument("--check", action="store_true",
                        help="Run quality check after processing (piano roll + issue report)")
    parser.add_argument("--tremolo-templates", default=None,
                        help="Directory with tremolo_tight_*.png templates for tremolo detection")
    parser.add_argument("--plugin-output", default=None,
                        help="Write GrandOMR plugin bundle to this directory")
    args = parser.parse_args()

    use_gpu = not args.no_gpu
    use_vlm = not args.no_vlm
    tremolo_tpl = args.tremolo_templates
    plugin_output = args.plugin_output

    inputs = [Path(p) for p in args.input]

    # Single directory mode
    if len(inputs) == 1 and inputs[0].is_dir():
        out_dir = args.output or str(inputs[0] / "pipeline_output")
        os.makedirs(out_dir, exist_ok=True)
        detected_names = None
        if plugin_output:
            print("[Plugin] --plugin-output is ignored in directory batch mode")
        for img_file in sorted(inputs[0].glob("*.png")):
            out_path = os.path.join(out_dir, img_file.stem + ".musicxml")
            try:
                _, names, _ = run_pipeline(str(img_file), out_path, use_gpu=use_gpu, use_vlm=use_vlm,
                             tremolo_templates=tremolo_tpl, part_names_override=detected_names)
                if detected_names is None and names:
                    detected_names = names
                if args.check:
                    quality_check(out_path)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                import traceback; traceback.print_exc()
        return

    # Single file mode
    if len(inputs) == 1 and inputs[0].is_file():
        out = args.output or str(inputs[0].with_suffix(".musicxml"))
        run_pipeline(str(inputs[0]), out, use_gpu=use_gpu, use_vlm=use_vlm,
                     tremolo_templates=tremolo_tpl, plugin_output=plugin_output)
        if args.check:
            quality_check(out)
        return

    # Multi-file merge mode
    if len(inputs) > 1:
        page_xmls = []
        detected_names = None
        all_plugin_pages = []
        for page_idx, img_path in enumerate(inputs):
            if not img_path.is_file():
                print(f"Error: {img_path} not found")
                sys.exit(1)
            out = str(img_path.with_suffix(".musicxml"))
            if plugin_output:
                _, names, _, plugin_pages, _xml_string = run_pipeline(
                    str(img_path), out, use_gpu=use_gpu, use_vlm=use_vlm,
                    tremolo_templates=tremolo_tpl, part_names_override=detected_names,
                    collect_plugin_data=True, page_index=page_idx,
                )
                all_plugin_pages.extend(plugin_pages)
            else:
                _, names, _ = run_pipeline(str(img_path), out, use_gpu=use_gpu, use_vlm=use_vlm,
                             tremolo_templates=tremolo_tpl, part_names_override=detected_names)
            if detected_names is None and names:
                detected_names = names
            page_xmls.append(out)

        merged_out = args.output
        if not merged_out:
            stem = inputs[0].parent / f"{inputs[0].stem}-{inputs[-1].stem}_merged.musicxml"
            merged_out = str(stem)
        merge_pages(page_xmls, merged_out)
        if plugin_output:
            final_xml = Path(merged_out).read_text(encoding="utf-8")
            write_plugin_output(
                plugin_output,
                musicxml_path=merged_out,
                xml_string=final_xml,
                plugin_pages=all_plugin_pages,
            )
        if args.check:
            quality_check(merged_out)
        return

    print(f"Error: {args.input[0]} not found")
    sys.exit(1)


if __name__ == "__main__":
    main()

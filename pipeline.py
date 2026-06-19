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
import xml.etree.ElementTree as ET
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
            results.append((best_si, best_nh.center[0], best_nh.center[1], tscore))
    return results


def inject_tremolo(result_staffs, matched_tremolo, staffs_sorted):
    """Inject tremolo articulation into the nearest note EncodedSymbol.
    matched_tremolo: list of (staff_index, nh_cx_homr, nh_cy_homr, score).
    Uses notehead position in HOMR space → canvas space → match to EncodedSymbol."""
    from collections import defaultdict
    import math

    by_staff = defaultdict(list)
    for si, nhx, nhy, score in matched_tremolo:
        by_staff[si].append((nhx, nhy, score))

    n_injected = 0
    for si, det_list in by_staff.items():
        if si >= len(result_staffs):
            continue
        staff = staffs_sorted[si]
        symbols = result_staffs[si]

        unit = staff.average_unit_size
        region_x_min = staff.min_x - 2 * unit
        region_x_max = staff.max_x + 2 * unit
        region_w = region_x_max - region_x_min

        canvas_w = 1280.0
        scale = canvas_w / region_w

        for nhx, nhy, score in det_list:
            canvas_x = (nhx - region_x_min) * scale

            best_idx, best_dist = -1, float("inf")
            for idx, sym in enumerate(symbols):
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


# ══════════════════════════════════════════════════════════════════════════════
# Key signature correction via accidental template matching
# ══════════════════════════════════════════════════════════════════════════════

BRAVURA_FONT_PATH = os.path.join(
    os.path.dirname(__file__), "audiveris", "app", "res", "Bravura.otf"
)

ACCIDENTAL_CODEPOINTS = {
    "flat":    0xE260,
    "natural": 0xE261,
    "sharp":   0xE262,
}


def _render_glyph(font_path, codepoint, font_size):
    """Render a single SMuFL glyph, return binary image (white-on-black)."""
    from PIL import Image as PILImage, ImageFont, ImageDraw

    font = ImageFont.truetype(font_path, font_size)
    ch = chr(codepoint)
    dummy = PILImage.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), ch, font=font)
    w = bbox[2] - bbox[0] + 4
    h = bbox[3] - bbox[1] + 4
    if w < 3 or h < 3:
        return None
    img = PILImage.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    draw.text((2 - bbox[0], 2 - bbox[1]), ch, font=font, fill=0)
    arr = np.array(img)
    _, binary = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, cw, ch2 = cv2.boundingRect(coords)
    return binary[y:y+ch2, x:x+cw]


def render_accidental_templates(unit_size):
    """Render flat/natural/sharp templates at multiple font sizes scaled to unit_size.
    Returns dict: {accidental_name: [list of grayscale template images (dark on white)]}."""
    base_font_size = int(unit_size * 4.5)
    templates = {}
    for name, cp in ACCIDENTAL_CODEPOINTS.items():
        tmpls = []
        for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            fs = max(12, int(base_font_size * scale))
            t = _render_glyph(BRAVURA_FONT_PATH, cp, fs)
            if t is not None:
                tmpls.append(255 - t)
        templates[name] = tmpls
    return templates


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


def _match_accidental_templates(gray_crop, templates_list, threshold=0.7):
    """Match a list of template images against a gray crop.
    Returns list of (cx, cy, w, h, score)."""
    all_dets = []
    for tmpl in templates_list:
        th, tw = tmpl.shape[:2]
        if th > gray_crop.shape[0] or tw > gray_crop.shape[1]:
            continue
        result = cv2.matchTemplate(gray_crop, tmpl, cv2.TM_CCOEFF_NORMED)
        locs = np.where(result >= threshold)
        for py, px in zip(*locs):
            score = float(result[py, px])
            cx = px + tw / 2
            cy = py + th / 2
            all_dets.append((cx, cy, tw, th, score))
    return all_dets


def _nms_accidentals(detections, iou_threshold=0.3):
    """NMS for accidental detections. Input: list of (cx, cy, w, h, score)."""
    if not detections:
        return []
    boxes = np.array([[d[0] - d[2]/2, d[1] - d[3]/2,
                       d[0] + d[2]/2, d[1] + d[3]/2] for d in detections])
    scores = np.array([d[4] for d in detections])
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
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return [detections[i] for i in keep]


def detect_accidentals_in_region(gray_crop, accidental_templates, threshold=0.7):
    """Detect flat/natural/sharp counts in a cropped region.
    Uses a stricter threshold for flat/sharp to reduce false positives.
    Returns dict: {'flat': N, 'natural': N, 'sharp': N}."""
    thresholds = {"natural": threshold, "flat": threshold + 0.05, "sharp": threshold + 0.05}
    counts = {}
    for name, tmpls in accidental_templates.items():
        t = thresholds.get(name, threshold)
        dets = _match_accidental_templates(gray_crop, tmpls, t)
        dets = _nms_accidentals(dets)
        counts[name] = len(dets)
    return counts


# ── Accidental CNN classifier ──────────────────────────────────────────────

import torch
import torch.nn as nn

ACCIDENTAL_CLASSES = ["flat", "natural", "sharp"]
ACCIDENTAL_CNN_PATH = os.path.join(os.path.dirname(__file__), "accidental_cnn.pth")
_ACCIDENTAL_CNN_PATCH_H = 32
_ACCIDENTAL_CNN_PATCH_W = 24


class AccidentalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, len(ACCIDENTAL_CLASSES))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _generate_accidental_data(n_per_class=2000, seed=42):
    """Generate synthetic training patches from Bravura font with augmentation."""
    rng = np.random.RandomState(seed)
    H, W = _ACCIDENTAL_CNN_PATCH_H, _ACCIDENTAL_CNN_PATCH_W
    images, labels = [], []

    for cls_idx, (name, cp) in enumerate(ACCIDENTAL_CODEPOINTS.items()):
        for _ in range(n_per_class):
            font_size = rng.randint(28, 51)
            glyph = _render_glyph(BRAVURA_FONT_PATH, cp, font_size)
            if glyph is None:
                continue

            gh, gw = glyph.shape
            scale = rng.uniform(0.7, 1.3)
            new_h = max(5, int(gh * scale))
            new_w = max(3, int(gw * scale))
            glyph = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA)
            gh, gw = glyph.shape

            canvas = np.full((H, W), 255, dtype=np.uint8)
            if gh > H or gw > W:
                glyph = cv2.resize(glyph, (min(gw, W - 2), min(gh, H - 2)))
                gh, gw = glyph.shape

            max_y = max(0, H - gh)
            max_x = max(0, W - gw)
            oy = rng.randint(0, max_y + 1)
            ox = rng.randint(0, max_x + 1)
            canvas[oy:oy+gh, ox:ox+gw] = np.minimum(
                canvas[oy:oy+gh, ox:ox+gw],
                255 - glyph
            )

            line_spacing = rng.randint(5, 11)
            line_start = rng.randint(0, max(1, line_spacing))
            line_thickness = rng.choice([1, 1, 1, 2])
            y = line_start
            while y < H:
                canvas[y:min(H, y+line_thickness), :] = 255
                y += line_spacing

            if rng.rand() < 0.5:
                k = rng.choice([3, 3, 5])
                canvas = cv2.GaussianBlur(canvas, (k, k), rng.uniform(0.3, 1.0))

            noise_sigma = rng.uniform(0, 15)
            if noise_sigma > 1:
                noise = rng.randn(H, W) * noise_sigma
                canvas = np.clip(canvas.astype(float) + noise, 0, 255).astype(np.uint8)

            if rng.rand() < 0.3:
                kern = np.ones((2, 2), np.uint8)
                canvas = cv2.erode(canvas, kern, iterations=1)
            elif rng.rand() < 0.3:
                kern = np.ones((2, 2), np.uint8)
                canvas = cv2.dilate(canvas, kern, iterations=1)

            alpha = rng.uniform(0.7, 1.3)
            beta = rng.uniform(-20, 20)
            canvas = np.clip(canvas.astype(float) * alpha + beta, 0, 255).astype(np.uint8)

            images.append(canvas)
            labels.append(cls_idx)

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int64)
    return images, labels


def _train_accidental_cnn(save_path=None, n_per_class=2000, epochs=30, lr=1e-3):
    """Train the AccidentalCNN on synthetic data and save weights."""
    if save_path is None:
        save_path = ACCIDENTAL_CNN_PATH

    print("[AccidentalCNN] Generating training data...")
    images, labels = _generate_accidental_data(n_per_class=n_per_class)

    perm = np.random.RandomState(0).permutation(len(images))
    images, labels = images[perm], labels[perm]
    split = int(0.8 * len(images))
    train_x, val_x = images[:split], images[split:]
    train_y, val_y = labels[:split], labels[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AccidentalCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    batch_size = 128

    train_x_t = torch.from_numpy(train_x).unsqueeze(1).to(device)
    train_y_t = torch.from_numpy(train_y).to(device)
    val_x_t = torch.from_numpy(val_x).unsqueeze(1).to(device)
    val_y_t = torch.from_numpy(val_y).to(device)

    for epoch in range(epochs):
        model.train()
        perm_idx = torch.randperm(len(train_x_t))
        total_loss = 0
        n_batches = 0
        for i in range(0, len(train_x_t), batch_size):
            idx = perm_idx[i:i+batch_size]
            out = model(train_x_t[idx])
            loss = criterion(out, train_y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(val_x_t)
                val_pred = val_out.argmax(dim=1)
                val_acc = (val_pred == val_y_t).float().mean().item()
            print(f"  Epoch {epoch+1:2d}: loss={total_loss/n_batches:.4f} val_acc={val_acc:.3f}")

    model.eval()
    with torch.no_grad():
        val_out = model(val_x_t)
        val_pred = val_out.argmax(dim=1)
        val_acc = (val_pred == val_y_t).float().mean().item()
        for ci, cn in enumerate(ACCIDENTAL_CLASSES):
            mask = val_y_t == ci
            if mask.sum() > 0:
                acc = (val_pred[mask] == ci).float().mean().item()
                print(f"  {cn}: {acc:.3f} ({mask.sum().item()} samples)")

    torch.save(model.state_dict(), save_path)
    print(f"[AccidentalCNN] Saved to {save_path} (val_acc={val_acc:.3f})")
    return model


_accidental_cnn_cache = None


def _load_accidental_cnn():
    """Load or train the accidental CNN. Caches in memory."""
    global _accidental_cnn_cache
    if _accidental_cnn_cache is not None:
        return _accidental_cnn_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AccidentalCNN().to(device)

    if os.path.exists(ACCIDENTAL_CNN_PATH):
        model.load_state_dict(torch.load(ACCIDENTAL_CNN_PATH, map_location=device))
        model.eval()
    else:
        model = _train_accidental_cnn(ACCIDENTAL_CNN_PATH)
        model = model.to(device)
        model.eval()

    _accidental_cnn_cache = model
    return model


def _classify_accidental_patch(model, gray_patch):
    """Classify a single grayscale patch. Returns (class_name, confidence)."""
    H, W = _ACCIDENTAL_CNN_PATCH_H, _ACCIDENTAL_CNN_PATCH_W
    patch = cv2.resize(gray_patch, (W, H), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        cls_idx = probs.argmax(dim=1).item()
        conf = probs[0, cls_idx].item()
    return ACCIDENTAL_CLASSES[cls_idx], conf


def correct_key_signatures(result_staffs, staffs_sorted, original_image, bar_line_boxes,
                           full_res_image=None):
    """Detect double barlines, classify accidentals with CNN in key change zones,
    and correct keySignature tokens in EncodedSymbol sequences.

    result_staffs: list[list[EncodedSymbol]] from parse_staffs()
    staffs_sorted: list[Staff] (HOMR staff objects)
    original_image: HOMR-resolution image (BGR or grayscale)
    bar_line_boxes: list[RotatedBoundingBox] from HOMR barline detection
    full_res_image: full-resolution page image for CNN classification (optional;
                    if None, falls back to original_image)
    """
    import math
    from collections import Counter

    if len(original_image.shape) == 3:
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = original_image
    img_h, img_w = gray.shape[:2]

    if full_res_image is not None:
        if len(full_res_image.shape) == 3:
            gray_full = cv2.cvtColor(full_res_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_full = full_res_image
        coord_scale = gray_full.shape[1] / img_w
    else:
        gray_full = gray
        coord_scale = 1.0

    full_h, full_w = gray_full.shape[:2]

    if not staffs_sorted:
        return 0

    unit = float(np.median([s.average_unit_size for s in staffs_sorted]))
    acc_templates = render_accidental_templates(unit)

    # ── Consensus double barline: find the most common x across all staves ──
    all_valid_staffs = [s for s in staffs_sorted if s.max_x - s.min_x >= 100]
    all_dbl_xs = []
    for s in all_valid_staffs:
        dxs = detect_double_barlines(s, bar_line_boxes)
        all_dbl_xs.extend(dxs)

    if not all_dbl_xs:
        return 0

    rounded = [round(x / 5) * 5 for x in all_dbl_xs]
    most_common_x = Counter(rounded).most_common(1)[0][0]
    consensus_xs = [x for x in all_dbl_xs if abs(round(x / 5) * 5 - most_common_x) <= 20]
    if not consensus_xs:
        return 0
    consensus_dbl_x = float(np.median(consensus_xs))

    # ── Load CNN classifier ──
    cnn = _load_accidental_cnn()

    n_corrected = 0
    for si, staff in enumerate(staffs_sorted):
        if si >= len(result_staffs):
            break
        if staff.max_x - staff.min_x < 100:
            continue

        u = staff.average_unit_size

        # Check this staff has a double barline near consensus
        staff_dbl_xs = detect_double_barlines(staff, bar_line_boxes)
        dbl_x = None
        for dx in staff_dbl_xs:
            if abs(dx - consensus_dbl_x) <= 3 * u:
                dbl_x = dx
                break
        if dbl_x is None:
            if abs(consensus_dbl_x - staff.max_x) < staff.max_x * 0.5:
                dbl_x = consensus_dbl_x
            else:
                continue

        # ── Crop key change zone at HOMR res for candidate detection ──
        x1 = int(max(0, dbl_x))
        x2 = int(min(img_w, dbl_x + 12 * u))
        y1 = int(max(0, staff.min_y - u))
        y2 = int(min(img_h, staff.max_y + u))
        if x2 <= x1 + 5 or y2 <= y1 + 5:
            continue

        crop = gray[y1:y2, x1:x2]

        # ── Find candidate accidental positions via template matching ──
        all_cand = _match_accidental_templates(crop, acc_templates["natural"], 0.55)
        all_cand = _nms_accidentals(all_cand)

        if len(all_cand) < 2:
            continue

        # ── Crop key change zone at full res for CNN classification ──
        fx1 = int(max(0, x1 * coord_scale))
        fx2 = int(min(full_w, x2 * coord_scale))
        fy1 = int(max(0, y1 * coord_scale))
        fy2 = int(min(full_h, y2 * coord_scale))
        crop_full = gray_full[fy1:fy2, fx1:fx2]

        # ── Classify each candidate with CNN using full-res patches ──
        u_full = u * coord_scale
        patch_h = int(u_full * 2.5)
        patch_w = int(u_full * 1.8)
        counts = {"flat": 0, "natural": 0, "sharp": 0}

        for cx, cy, tw, th, score in all_cand:
            fcx = cx * coord_scale
            fcy = cy * coord_scale
            px1 = int(max(0, fcx - patch_w / 2))
            py1 = int(max(0, fcy - patch_h / 2))
            px2 = int(min(crop_full.shape[1], fcx + patch_w / 2))
            py2 = int(min(crop_full.shape[0], fcy + patch_h / 2))
            if px2 - px1 < 6 or py2 - py1 < 10:
                continue

            patch = crop_full[py1:py2, px1:px2]
            cls_name, conf = _classify_accidental_patch(cnn, patch)
            if conf >= 0.5:
                counts[cls_name] += 1

        n_nat = counts["natural"]
        n_flat = counts["flat"]
        n_sharp = counts["sharp"]

        if n_nat < 2:
            continue

        new_fifths = n_sharp - n_flat

        # ── Find the keySignature token to modify ──
        symbols = result_staffs[si]
        region_xmin = staff.min_x - 2 * u
        region_w = (staff.max_x + 2 * u) - region_xmin
        canvas_dbl_x = (dbl_x - region_xmin) / region_w * 1280

        best_idx = -1
        best_dist = float("inf")
        for idx, sym in enumerate(symbols):
            if not sym.rhythm.startswith("keySignature_"):
                continue
            if sym.coordinates is None or math.isnan(sym.coordinates[0]):
                continue
            if sym.coordinates[0] > canvas_dbl_x:
                dist = sym.coordinates[0] - canvas_dbl_x
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

        if best_idx >= 0 and best_dist < 300:
            old_rhythm = symbols[best_idx].rhythm
            new_rhythm = f"keySignature_{new_fifths}"
            if old_rhythm != new_rhythm:
                print(f"  [KeySig] Staff {si}: {old_rhythm} → {new_rhythm} "
                      f"(detected {n_nat}♮ {n_flat}♭ {n_sharp}♯)")
                symbols[best_idx].rhythm = new_rhythm
                n_corrected += 1

    return n_corrected


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
                      part_names_override: List[str] = None) -> List[Tuple[str, List[str]]]:
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
    from homr.music_xml_generator import generate_xml, XmlGeneratorArguments
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

    print(f"[HOMR] Detected {len(system_groups)} system(s): {sys_sizes} staves each")

    first_sys_count = sys_sizes[0]
    all_same = all(sz == first_sys_count for sz in sys_sizes)
    multi_system_mode = (n_parts > 0 and first_sys_count == n_parts
                         and not all_same)

    if multi_system_mode:
        # ── MULTI-SYSTEM with different staff counts ──
        print(f"[HOMR] Multi-system page: {sys_sizes}")

        transformer_config = Config()
        transformer_config.use_gpu_inference = use_gpu
        xml_args = XmlGeneratorArguments()

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

        full_image_ks = cv2.imread(img_path)

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
                    n_inj = inject_tremolo(sys_result, matched, sys_staves)
                    print(f"[Tremolo] System {sys_idx + 1}: {n_inj} injected")

            n_ks = correct_key_signatures(
                sys_result, sys_staves, predictions.original,
                bar_line_boxes, full_res_image=full_image_ks)
            if n_ks:
                print(f"[KeySig] System {sys_idx + 1}: {n_ks} corrected")

            sys_names = _detect_names_for_system(
                sys_staves, predictions.original, brace_dots, use_vlm=use_vlm)

            xml_root = generate_xml(xml_args, sys_result, title)
            xml_string = xml_root.to_string()

            sys_dynamics = detect_dynamics(sys_staves, img_path, bar_line_boxes, predictions.original.shape)
            if sys_dynamics:
                xml_string = _inject_dynamics(xml_string, sys_dynamics)
                print(f"[Dynamics] System {sys_idx + 1}: {len(sys_dynamics)} marking(s)")

            results.append((xml_string, sys_names))

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
            n_inj = inject_tremolo(result_staffs, matched, staffs_sorted)
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

    xml_args = XmlGeneratorArguments()
    xml_root = generate_xml(xml_args, result_staffs, title)
    xml_string = xml_root.to_string()

    t_dyn = time.time()
    dynamics = detect_dynamics(staffs_sorted, img_path, bar_line_boxes, predictions.original.shape)
    if dynamics:
        xml_string = _inject_dynamics(xml_string, dynamics)
        print(f"[Dynamics] Injected {len(dynamics)} marking(s) ({time.time() - t_dyn:.1f}s)")

    return [(xml_string, part_names)]


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


def _cross_part_post_process(xml_string: str) -> str:
    """
    Post-process MusicXML using cross-part consistency constraints.

    Guiding principle: all parts in an orchestral score share one timeline.
    TrOMR recognizes each staff independently, so we use multi-part consensus
    to correct individual errors.

    Layer 1 — Metadata alignment: time sig, key sig (majority vote per measure)
    Layer 2 — Structural alignment: unify measure count across parts
    Layer 3 — Content repair: fix measures whose duration != time signature
    """
    from collections import Counter

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return xml_string

    parts = root.findall("part")
    if len(parts) < 2:
        return xml_string

    fixes = []

    # ── Layer 0: Infer time signature if missing ──
    has_any_time_sig = any(
        m.find(".//time") is not None
        for part in parts for m in part.findall("measure")
    )
    if not has_any_time_sig:
        # compute m1 quarter-lengths from all parts via position tracking
        m1_qls = []
        for part in parts:
            m1 = part.findall("measure")
            if not m1:
                continue
            m1 = m1[0]
            divs = 1
            d_el = m1.findtext(".//divisions")
            if d_el:
                try:
                    divs = int(d_el)
                except ValueError:
                    pass
            pos = 0
            max_pos = 0
            for child in m1:
                if child.tag == "note":
                    if child.find("chord") is not None:
                        continue
                    dur = child.findtext("duration", "0")
                    try:
                        pos += int(dur)
                    except ValueError:
                        pass
                    max_pos = max(max_pos, pos)
                elif child.tag == "backup":
                    dur = child.findtext("duration", "0")
                    try:
                        pos -= int(dur)
                    except ValueError:
                        pass
                elif child.tag == "forward":
                    dur = child.findtext("duration", "0")
                    try:
                        pos += int(dur)
                    except ValueError:
                        pass
            if max_pos > 0:
                m1_qls.append(max_pos / divs)

        if m1_qls:
            median_ql = sorted(m1_qls)[len(m1_qls) // 2]
            # map to standard time signatures
            ql_to_ts = {2.0: (2, 4), 3.0: (3, 4), 4.0: (4, 4), 6.0: (6, 4)}
            rounded = round(median_ql * 2) / 2  # round to nearest 0.5
            beats, beat_type = ql_to_ts.get(rounded, (round(rounded), 4))
            fixes.append(f"Inferred time sig {beats}/{beat_type} from m1 durations (median={median_ql:.2f})")
            # inject into first measure of every part
            for part in parts:
                m1 = part.findall("measure")
                if not m1:
                    continue
                m1 = m1[0]
                attrs = m1.find("attributes")
                if attrs is None:
                    attrs = ET.SubElement(m1, "attributes")
                    m1.remove(attrs)
                    m1.insert(0, attrs)
                time_el = ET.SubElement(attrs, "time")
                ET.SubElement(time_el, "beats").text = str(beats)
                ET.SubElement(time_el, "beat-type").text = str(beat_type)

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

    if fixes:
        print(f"[PostProcess] {len(fixes)} fixes:")
        for f in fixes:
            print(f"  {f}")
    else:
        print("[PostProcess] No fixes needed")

    return ET.tostring(root, encoding="unicode", xml_declaration=True)


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
    for name, occ in master_parts:
        suffix = f" {occ+1}" if all_instruments[name] > 1 else ""
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
        display_name = f"{dn} {occ+1}" if all_instruments[name] > 1 else dn
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
                 tremolo_templates: str = None, part_names_override: List[str] = None):
    """Full pipeline: image → MusicXML.
    Returns (output_path, part_names) — part_names is the detected/applied instrument list,
    useful for passing as override to subsequent pages."""
    print(f"\n{'='*60}")
    print(f"Processing: {img_path}")
    print(f"{'='*60}")
    t_start = time.time()

    results = run_homr_pipeline(
        img_path, use_gpu=use_gpu, use_vlm=use_vlm,
        tremolo_templates=tremolo_templates,
        part_names_override=part_names_override,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    final_names = None

    if len(results) == 1:
        xml_string, part_names = results[0]
        if not part_names and part_names_override is not None:
            part_names = list(part_names_override)
            print(f"[Override] No labels on this page, reusing: {len(part_names)} instruments")
        elif part_names_override is not None and len(part_names_override) > len(part_names):
            part_names = _match_override_to_detected(part_names_override, part_names, xml_string)
        xml_string = _inject_part_names(xml_string, part_names)
        xml_string = _cross_part_post_process(xml_string)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_string)
        final_names = part_names
    else:
        import tempfile
        temp_files = []
        system_master_names = None
        for sys_idx, (xml_string, sys_names) in enumerate(results):
            if not sys_names and part_names_override is not None:
                sys_names = list(part_names_override)
                print(f"[Override] System {sys_idx+1}: no labels, reusing: {len(sys_names)} instruments")
            elif part_names_override is not None and len(part_names_override) > len(sys_names):
                sys_names = _match_override_to_detected(part_names_override, sys_names, xml_string)
            if sys_idx == 0:
                final_names = sys_names
                system_master_names = list(part_names_override or sys_names or [])
            else:
                master_names = list(system_master_names or [])
                if not sys_names and master_names:
                    sys_names = list(master_names)
                    print(f"[Override] System {sys_idx+1}: no labels, reusing: {len(sys_names)} instruments")
                elif master_names and len(master_names) > len(sys_names):
                    sys_names = _match_override_to_detected(master_names, sys_names, xml_string)
            xml_string = _inject_part_names(xml_string, sys_names)
            xml_string = _cross_part_post_process(xml_string)
            base, ext = os.path.splitext(output_path)
            sys_path = f"{base}_sys{sys_idx}{ext}"
            with open(sys_path, "w", encoding="utf-8") as f:
                f.write(xml_string)
            temp_files.append(sys_path)
            print(f"[MultiSys] System {sys_idx + 1}: {sys_path}")
        merge_pages(temp_files, output_path)

    elapsed = time.time() - t_start
    print(f"\n[Done] {output_path} ({elapsed:.1f}s)")
    return output_path, final_names


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
    args = parser.parse_args()

    use_gpu = not args.no_gpu
    use_vlm = not args.no_vlm
    tremolo_tpl = args.tremolo_templates

    inputs = [Path(p) for p in args.input]

    # Single directory mode
    if len(inputs) == 1 and inputs[0].is_dir():
        out_dir = args.output or str(inputs[0] / "pipeline_output")
        os.makedirs(out_dir, exist_ok=True)
        detected_names = None
        for img_file in sorted(inputs[0].glob("*.png")):
            out_path = os.path.join(out_dir, img_file.stem + ".musicxml")
            try:
                _, names = run_pipeline(str(img_file), out_path, use_gpu=use_gpu, use_vlm=use_vlm,
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
                     tremolo_templates=tremolo_tpl)
        if args.check:
            quality_check(out)
        return

    # Multi-file merge mode
    if len(inputs) > 1:
        page_xmls = []
        detected_names = None
        for img_path in inputs:
            if not img_path.is_file():
                print(f"Error: {img_path} not found")
                sys.exit(1)
            out = str(img_path.with_suffix(".musicxml"))
            _, names = run_pipeline(str(img_path), out, use_gpu=use_gpu, use_vlm=use_vlm,
                         tremolo_templates=tremolo_tpl, part_names_override=detected_names)
            if detected_names is None and names:
                detected_names = names
            page_xmls.append(out)

        merged_out = args.output
        if not merged_out:
            stem = inputs[0].parent / f"{inputs[0].stem}-{inputs[-1].stem}_merged.musicxml"
            merged_out = str(stem)
        merge_pages(page_xmls, merged_out)
        if args.check:
            quality_check(merged_out)
        return

    print(f"Error: {args.input[0]} not found")
    sys.exit(1)


if __name__ == "__main__":
    main()

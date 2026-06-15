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
    # Clarinet
    "cl": "Clarinet", "cl.": "Clarinet", "clar": "Clarinet", "clarinetto": "Clarinet",
    "clarinette": "Clarinet", "clarinetten": "Clarinet", "klar": "Clarinet",
    # Bass Clarinet
    "baßclarinette": "Bass Clarinet", "bassclarinette": "Bass Clarinet",
    "baßklarinette": "Bass Clarinet", "bassklarinette": "Bass Clarinet",
    "bcl": "Bass Clarinet", "b.cl": "Bass Clarinet",
    "babclarinette": "Bass Clarinet", "babklarinette": "Bass Clarinet",
    # Bassoon
    "fg": "Bassoon", "fg.": "Bassoon", "fag": "Bassoon", "fagotto": "Bassoon",
    "fagotte": "Bassoon", "fagott": "Bassoon",
    # Contrabassoon
    "contrafagott": "Contrabassoon", "contrafag": "Contrabassoon", "cfg": "Contrabassoon",
    "kontrafagott": "Contrabassoon", "c.-fag": "Contrabassoon", "c.fag": "Contrabassoon",
    "c.-fag.": "Contrabassoon", "c.fag.": "Contrabassoon",
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
    "Violin":       ("strings.violin",        41),
    "Viola":        ("strings.viola",          42),
    "Cello":        ("strings.cello",          43),
    "Contrabass":   ("strings.contrabass",     44),
    "Flute":        ("wind.flutes.flute",      74),
    "Piccolo":      ("wind.flutes.flute.piccolo", 73),
    "Oboe":         ("wind.reed.oboe",         69),
    "English Horn": ("wind.reed.english-horn", 70),
    "Clarinet":     ("wind.reed.clarinet",     72),
    "Bass Clarinet": ("wind.reed.clarinet.bass", 72),
    "Bassoon":      ("wind.reed.bassoon",      71),
    "Contrabassoon": ("wind.reed.contrabassoon", 71),
    "Horn":         ("brass.french-horn",      61),
    "Trumpet":      ("brass.trumpet",          57),
    "Trombone":     ("brass.trombone",          58),
    "Tuba":         ("brass.tuba",             59),
    "Bass Tuba":    ("brass.tuba",             59),
    "Timpani":      ("percussion.timpani",     48),
    "Bass Drum":    ("drum.bass-drum",         117),
    "Harp":         ("pluck.harp",             47),
    "Piano":        ("keyboard.piano",          1),
    "Celesta":      ("keyboard.celesta",        9),
}


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

_VLM_PROMPT = """This is a page from an orchestral music score.
On the left margin there are instrument names or abbreviations.
Read each instrument name from top to bottom and output its standard English name.
Use these standard names: Flute, Piccolo, Oboe, English Horn, Clarinet, Bass Clarinet, Bassoon, Contrabassoon, Horn, Trumpet, Trombone, Tuba, Bass Tuba, Timpani, Bass Drum, Harp, Celesta, Piano, Violin, Viola, Cello, Contrabass.
Important rules:
- If a bracket groups two staves under one label (e.g. "Pos." with "1/2" and "3"), output the SAME name for EACH staff in that bracket.
- If one label covers multiple numbered staves (e.g. "Hr.F" with "1/3" and "2/4"), output the same name for each.
- German abbreviations: Fl.=Flute, Ob.=Oboe, Kl./Cl.=Clarinet, Fg./Fag.=Bassoon, C-Fag.=Contrabassoon, Hr./Hrn.=Horn, Trp.=Trumpet, Pos.=Trombone, Pk.=Timpani, Gr.Tr.=Bass Drum, Hrf./Hfe.=Harp, Cel.=Celesta, Vl.=Violin, Va./Br.=Viola, Vc.=Cello, B./Kb.=Contrabass.
There are exactly {n} staves. Output exactly {n} lines, one standard name per staff line, from top to bottom. No numbering, no extra text."""


def _vlm_read_instrument_names(image_pil, n_staves: int) -> list:
    """Use VLM API (Qwen3-VL-235B) to read instrument names from a score page."""
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

    client = openai.OpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/v1/")
    prompt = _VLM_PROMPT.format(n=n_staves)

    response = client.chat.completions.create(
        model="Qwen3-VL-235B-A22B-Instruct",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        max_tokens=500,
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()
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


def ocr_instrument_names_from_staves(homr_staffs, image, brace_dots=None, use_vlm=True) -> List[str]:
    """
    Identify instrument names for each staff.
    Primary: VLM reads the full page image and outputs instrument names.
    Fallback: RapidOCR on left margin crops with bracket grouping.
    """
    if not homr_staffs:
        return []

    sorted_staffs = sorted(homr_staffs, key=lambda s: s.min_y)
    avg_unit = float(np.median([s.average_unit_size for s in sorted_staffs]))

    # Detect system boundaries
    gaps = []
    for i in range(1, len(sorted_staffs)):
        gaps.append(sorted_staffs[i].min_y - sorted_staffs[i-1].max_y)
    if gaps:
        median_gap = float(np.median(gaps))
        system_break_threshold = max(median_gap * 2.0, avg_unit * 6)
    else:
        system_break_threshold = float('inf')
    systems = [[sorted_staffs[0]]]
    for i in range(1, len(sorted_staffs)):
        if gaps[i-1] > system_break_threshold:
            systems.append([])
        systems[-1].append(sorted_staffs[i])

    n_parts = len(systems[0])
    first_system = systems[0]
    staff_left = min(s.min_x for s in first_system)

    # Bracket grouping
    if brace_dots:
        bracket_groups = _group_staves_by_brackets(first_system, brace_dots)
    else:
        bracket_groups = [[i] for i in range(n_parts)]

    print(f"[OCR] {n_parts} staves, {len(bracket_groups)} groups, staff_left={int(staff_left)}")

    # ── Try VLM first ──
    if use_vlm:
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
            vlm_lines = _vlm_read_instrument_names(pil_img, n_staves=n_parts)
            print(f"[VLM] {len(vlm_lines)} names: {vlm_lines}")

            if len(vlm_lines) == n_parts:
                # Validate: names should be from our known set
                known_instruments = set(INSTRUMENT_MIDI.keys())
                valid = sum(1 for n in vlm_lines if n in known_instruments)
                if valid >= n_parts * 0.5:
                    print(f"[VLM] Direct match: {valid}/{n_parts} known instruments")
                    return vlm_lines

            if vlm_lines:
                labels = _match_vlm_names_to_staves(vlm_lines, bracket_groups, n_parts)
                if labels:
                    for i, name in enumerate(labels):
                        print(f"  Staff {i}: {name}")
                    return labels
            print("[VLM] Could not match, falling back to RapidOCR")
        except Exception as e:
            print(f"[VLM] Failed: {e}, falling back to RapidOCR")

    # ── Fallback: RapidOCR ──
    return _rapidocr_instrument_names(first_system, image, bracket_groups, n_parts, avg_unit, staff_left)


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
# HOMR pipeline (all GPU)
# ══════════════════════════════════════════════════════════════════════════════

def run_homr_pipeline(img_path: str, use_gpu: bool = True, use_vlm: bool = True) -> Tuple[str, List[str]]:
    """
    Run HOMR's complete pipeline with automatic system grouping override
    for orchestral scores. Returns (MusicXML string, list of part names).
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
    part_names = ocr_instrument_names_from_staves(staffs, predictions.original, brace_dots, use_vlm=use_vlm)

    # ── Determine parts-per-system and group staves ──
    n_parts = len(part_names)
    staffs_sorted = sorted(staffs, key=lambda s: s.min_y)

    if n_parts > 0 and len(staffs_sorted) >= n_parts:
        n_systems = len(staffs_sorted) // n_parts
        remainder = len(staffs_sorted) % n_parts
        multi_staffs = []
        for sys_idx in range(n_systems):
            start = sys_idx * n_parts
            system_staves = [staffs_sorted[start + p] for p in range(n_parts)]
            multi_staffs.append(MultiStaff(system_staves, []))
        if remainder > 0:
            print(f"[HOMR] Warning: {remainder} extra staves ignored (not a full system)")
        print(f"[HOMR] Grouped: {n_parts} parts × {n_systems} systems")
    else:
        brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
        brace_dot = create_rotated_bounding_boxes(
            brace_dot_img, skip_merging=True, max_size=(100, -1),
        )
        multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)
        print(f"[HOMR] Auto-grouped: {[len(ms.staffs) for ms in multi_staffs]}")
        n_parts = len(multi_staffs[0].staffs) if multi_staffs else 0

    # ── Per-staff recognition (TrOMR) ──
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
    print(f"[HOMR] TrOMR done: {n_result} parts, {n_symbols} symbols ({t4-t3:.1f}s)")

    xml_args = XmlGeneratorArguments()
    xml_root = generate_xml(xml_args, result_staffs, title)
    xml_string = xml_root.to_string()

    return xml_string, part_names


# ══════════════════════════════════════════════════════════════════════════════
# Post-process MusicXML
# ══════════════════════════════════════════════════════════════════════════════

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
    for sp, name in zip(score_parts, part_names):
        pn = sp.find("part-name")
        if pn is not None:
            pn.text = name
        else:
            ET.SubElement(sp, "part-name").text = name

        sound, midi_prog = INSTRUMENT_MIDI.get(name, ("keyboard.piano", 1))

        si = sp.find("score-instrument")
        if si is not None:
            iname = si.find("instrument-name")
            if iname is not None:
                iname.text = name
            isound = si.find("instrument-sound")
            if isound is not None:
                isound.text = sound

        mi = sp.find("midi-instrument")
        if mi is not None:
            mp = mi.find("midi-program")
            if mp is not None:
                mp.text = str(midi_prog)

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

        # Key signature consensus
        key_sigs = []
        for m in measures:
            k = m.find(".//key")
            if k is not None:
                key_sigs.append(k.findtext("fifths", "0"))
        if key_sigs:
            maj_fifths, cnt = Counter(key_sigs).most_common(1)[0]
            if cnt < len(key_sigs):
                fixes.append(f"m{mn}: key sig → fifths={maj_fifths}")
            for m in measures:
                k = m.find(".//key")
                if k is not None:
                    f_el = k.find("fifths")
                    if f_el is not None:
                        f_el.text = maj_fifths

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

    def _get_expected_quarter_length(measure):
        """Get expected quarter-note length from time signature in context."""
        t = measure.find(".//time")
        if t is not None:
            try:
                beats = int(t.findtext("beats", "4"))
                bt = int(t.findtext("beat-type", "4"))
                return beats * 4.0 / bt
            except ValueError:
                pass
        return 4.0

    def _get_divisions(measure):
        d = measure.findtext(".//divisions")
        return int(d) if d else 1

    for pi, part in enumerate(parts):
        pid = part.get("id", f"P{pi+1}")
        current_expected = 4.0
        current_divs = 1

        for measure in part.findall("measure"):
            mn = measure.get("number", "?")

            t = measure.find(".//time")
            if t is not None:
                current_expected = _get_expected_quarter_length(measure)
            d = measure.find(".//divisions")
            if d is not None:
                try:
                    current_divs = int(d.text)
                except (ValueError, TypeError):
                    pass

            notes = measure.findall("note")
            if not notes:
                continue

            # compute actual measure duration via position tracking
            pos = 0
            max_pos = 0
            for child in measure:
                if child.tag == "note":
                    if child.find("chord") is not None:
                        continue
                    dur_el = child.find("duration")
                    if dur_el is not None:
                        try:
                            pos += int(dur_el.text)
                        except (ValueError, TypeError):
                            pass
                    max_pos = max(max_pos, pos)
                elif child.tag == "backup":
                    dur_el = child.find("duration")
                    if dur_el is not None:
                        try:
                            pos -= int(dur_el.text)
                        except (ValueError, TypeError):
                            pass
                elif child.tag == "forward":
                    dur_el = child.find("duration")
                    if dur_el is not None:
                        try:
                            pos += int(dur_el.text)
                        except (ValueError, TypeError):
                            pass

            actual_ql = max_pos / current_divs if current_divs else max_pos
            expected_ql = current_expected

            if abs(actual_ql - expected_ql) < 0.01 or max_pos == 0:
                continue

            ratio = expected_ql / actual_ql
            if 0.3 < ratio < 3.0 and ratio != 1.0:
                # scale all duration elements: notes, backups, forwards
                dur_elements = []
                for child in measure:
                    if child.tag == "note":
                        dur_el = child.find("duration")
                        if dur_el is not None:
                            dur_elements.append(dur_el)
                    elif child.tag in ("backup", "forward"):
                        dur_el = child.find("duration")
                        if dur_el is not None:
                            dur_elements.append(dur_el)
                for dur_el in dur_elements:
                    try:
                        old = int(dur_el.text)
                        dur_el.text = str(max(1, round(old * ratio)))
                    except (ValueError, TypeError):
                        pass
                fixes.append(f"{pid} m{mn}: scaled durations ×{ratio:.2f} ({actual_ql}→{expected_ql})")

    if fixes:
        print(f"[PostProcess] {len(fixes)} fixes:")
        for f in fixes:
            print(f"  {f}")
    else:
        print("[PostProcess] No fixes needed")

    return ET.tostring(root, encoding="unicode", xml_declaration=True)


# ══════════════════════════════════════════════════════════════════════════════
# Quality check
# ══════════════════════════════════════════════════════════════════════════════

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
                                 key=lambda x: order_map.get(x[0], 99)):
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

    # Collect time signatures per page (majority vote across all parts)
    page_time_sigs = []
    for pinfo in page_data:
        from collections import Counter
        ts_votes = []
        for pi in pinfo:
            for m in pi["measures"]:
                t = m.find(".//time")
                if t is not None:
                    try:
                        b = int(t.findtext("beats", "4"))
                        bt = int(t.findtext("beat-type", "4"))
                        ts_votes.append((b, bt))
                    except ValueError:
                        pass
        if ts_votes:
            page_time_sigs.append(Counter(ts_votes).most_common(1)[0][0])
        else:
            page_time_sigs.append((4, 4))

    # Build merged XML
    merged_root = ET.Element("score-partwise")
    ET.SubElement(merged_root, "defaults")
    part_list_el = ET.SubElement(merged_root, "part-list")

    for mi, (name, occ) in enumerate(master_parts):
        pid = f"P{mi+1}"
        display_name = f"{name} {occ+1}" if all_instruments[name] > 1 else name
        sp = ET.SubElement(part_list_el, "score-part", id=pid)
        ET.SubElement(sp, "part-name").text = display_name
        si = ET.SubElement(sp, "score-instrument", id=f"{pid}-I1")
        ET.SubElement(si, "instrument-name").text = name
        sound, midi_prog = INSTRUMENT_MIDI.get(name, ("keyboard.piano", 1))
        ET.SubElement(si, "instrument-sound").text = sound
        midi_el = ET.SubElement(sp, "midi-instrument", id=f"{pid}-I1")
        ET.SubElement(midi_el, "midi-channel").text = "1"
        ET.SubElement(midi_el, "midi-program").text = str(midi_prog)
        ET.SubElement(midi_el, "volume").text = "100"
        ET.SubElement(midi_el, "pan").text = "0"

    # Concatenate measures for each master part, normalizing divisions
    for mi, (name, occ) in enumerate(master_parts):
        pid = f"P{mi+1}"
        part_el = ET.SubElement(merged_root, "part", id=pid)
        measure_num = 1

        for page_idx, pk in enumerate(page_keys):
            page_info = page_data[page_idx]
            n_measures = max(pi["n_measures"] for pi in page_info) if page_info else 0
            beats, beat_type = page_time_sigs[page_idx]

            if (name, occ) in pk:
                pi = pk[(name, occ)]
                cur_divs = 1
                for m in pi["measures"]:
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
                    # set divisions to target in attributes
                    attrs = new_m.find("attributes")
                    if attrs is not None:
                        d_el = attrs.find("divisions")
                        if d_el is not None:
                            d_el.text = str(target_divs)
                    elif measure_num == 1 or (page_idx > 0 and m is pi["measures"][0]):
                        attrs = ET.Element("attributes")
                        ET.SubElement(attrs, "divisions").text = str(target_divs)
                        new_m.insert(0, attrs)
                    part_el.append(new_m)
                    measure_num += 1
                for _ in range(n_measures - pi["n_measures"]):
                    part_el.append(_make_rest_measure(measure_num, target_divs, beats, beat_type))
                    measure_num += 1
            else:
                for mi2 in range(n_measures):
                    part_el.append(_make_rest_measure(
                        measure_num, target_divs, beats, beat_type,
                        include_attrs=(measure_num == 1 or mi2 == 0)))
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

    xml_string = ET.tostring(merged_root, encoding="unicode", xml_declaration=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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


def _make_rest_measure(number, divs, beats=4, beat_type=4, include_attrs=True):
    """Create an empty rest measure with correct divisions."""
    m = ET.Element("measure", number=str(number))
    if include_attrs:
        attrs = ET.SubElement(m, "attributes")
        ET.SubElement(attrs, "divisions").text = str(divs)
    rest = ET.SubElement(m, "note")
    ET.SubElement(rest, "rest")
    dur = ET.SubElement(rest, "duration")
    dur.text = str(divs * beats * 4 // beat_type)
    ET.SubElement(rest, "type").text = "whole"
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(img_path: str, output_path: str, use_gpu: bool = True, use_vlm: bool = True) -> str:
    """Full pipeline: image → MusicXML."""
    print(f"\n{'='*60}")
    print(f"Processing: {img_path}")
    print(f"{'='*60}")
    t_start = time.time()

    xml_string, part_names = run_homr_pipeline(img_path, use_gpu=use_gpu, use_vlm=use_vlm)
    xml_string = _inject_part_names(xml_string, part_names)
    xml_string = _cross_part_post_process(xml_string)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_string)

    elapsed = time.time() - t_start
    print(f"\n[Done] {output_path} ({elapsed:.1f}s)")
    return output_path


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
    args = parser.parse_args()

    use_gpu = not args.no_gpu
    use_vlm = not args.no_vlm

    inputs = [Path(p) for p in args.input]

    # Single directory mode
    if len(inputs) == 1 and inputs[0].is_dir():
        out_dir = args.output or str(inputs[0] / "pipeline_output")
        os.makedirs(out_dir, exist_ok=True)
        for img_file in sorted(inputs[0].glob("*.png")):
            out_path = os.path.join(out_dir, img_file.stem + ".musicxml")
            try:
                run_pipeline(str(img_file), out_path, use_gpu=use_gpu, use_vlm=use_vlm)
                if args.check:
                    quality_check(out_path)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                import traceback; traceback.print_exc()
        return

    # Single file mode
    if len(inputs) == 1 and inputs[0].is_file():
        out = args.output or str(inputs[0].with_suffix(".musicxml"))
        run_pipeline(str(inputs[0]), out, use_gpu=use_gpu, use_vlm=use_vlm)
        if args.check:
            quality_check(out)
        return

    # Multi-file merge mode
    if len(inputs) > 1:
        page_xmls = []
        for img_path in inputs:
            if not img_path.is_file():
                print(f"Error: {img_path} not found")
                sys.exit(1)
            out = str(img_path.with_suffix(".musicxml"))
            run_pipeline(str(img_path), out, use_gpu=use_gpu, use_vlm=use_vlm)
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

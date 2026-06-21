#!/usr/bin/env python3
"""End-to-end orchestral score OMR: PDF → per-page images → recognition → merged MusicXML.

Usage:
    python run_score.py score.pdf 3 8                    # pages 3-8, output to outputs/score_3_8.musicxml
    python run_score.py score.pdf 5 5                    # single page 5
    python run_score.py score.pdf 1 10 -o result.musicxml  # custom output path
    python run_score.py score.pdf 1 10 --dpi 400         # higher resolution
"""
import argparse
import os
import sys
import time
from pathlib import Path

from pdf2image import convert_from_path
from pipeline import run_pipeline, merge_pages, write_plugin_output


def main():
    parser = argparse.ArgumentParser(description="PDF orchestral score → MusicXML")
    parser.add_argument("pdf", help="Path to the score PDF")
    parser.add_argument("start", type=int, help="First page number (1-based, inclusive)")
    parser.add_argument("end", type=int, help="Last page number (1-based, inclusive)")
    parser.add_argument("-o", "--output", default=None, help="Output .musicxml path")
    parser.add_argument("--dpi", type=int, default=300, help="PDF rendering DPI (default: 300)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU inference")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM instrument detection")
    parser.add_argument("--plugin-output", default=None,
                        help="Write GrandOMR plugin bundle to this directory")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"Error: {pdf_path} not found")
        sys.exit(1)
    if args.start < 1 or args.end < args.start:
        print(f"Error: invalid page range {args.start}-{args.end}")
        sys.exit(1)

    stem = pdf_path.stem
    out_path = args.output or f"outputs/{stem}_{args.start}_{args.end}.musicxml"
    work_dir = os.path.join("outputs", stem)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    use_gpu = not args.no_gpu
    use_vlm = not args.no_vlm
    t_total = time.time()

    # 1. PDF → PNG
    print(f"[PDF] Rendering pages {args.start}-{args.end} at {args.dpi} DPI ...")
    images = convert_from_path(
        str(pdf_path), dpi=args.dpi,
        first_page=args.start, last_page=args.end,
    )
    png_paths = []
    for i, img in enumerate(images):
        page_num = args.start + i
        png_path = os.path.join(work_dir, f"page-{page_num:03d}.png")
        img.save(png_path, "PNG")
        png_paths.append(png_path)
        print(f"  {png_path} ({img.size[0]}×{img.size[1]})")

    # 2. Per-page recognition with cross-page name and time-sig propagation
    page_xmls = []
    plugin_pages = []
    detected_names = None
    ts_context = None
    for page_idx, png_path in enumerate(png_paths):
        xml_path = png_path.replace(".png", ".musicxml")
        if args.plugin_output:
            _, names, ts_context, page_plugin_pages, _xml_string = run_pipeline(
                png_path, xml_path,
                use_gpu=use_gpu, use_vlm=use_vlm,
                part_names_override=detected_names,
                collect_plugin_data=True,
                page_index=page_idx,
                ts_context=ts_context,
            )
            plugin_pages.extend(page_plugin_pages)
        else:
            _, names, ts_context = run_pipeline(
                png_path, xml_path,
                use_gpu=use_gpu, use_vlm=use_vlm,
                part_names_override=detected_names,
                ts_context=ts_context,
            )
        if detected_names is None and names:
            detected_names = names
        page_xmls.append(xml_path)

    # 3. Merge
    if len(page_xmls) > 1:
        merge_pages(page_xmls, out_path)
    else:
        import shutil
        shutil.copy2(page_xmls[0], out_path)

    if args.plugin_output:
        final_xml = Path(out_path).read_text(encoding="utf-8")
        write_plugin_output(
            args.plugin_output,
            musicxml_path=out_path,
            xml_string=final_xml,
            plugin_pages=plugin_pages,
        )

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Done: {out_path}  ({elapsed:.1f}s total, {len(page_xmls)} pages)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

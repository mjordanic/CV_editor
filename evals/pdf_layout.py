"""Deterministic layout checks of a rendered PDF, and page images for the evaluator's visual review.

`layout_checks(pdf)` reads every character's font, size and position (pdfplumber) and reports:

    pages, page_size_pt        page count and size of the first page
    fonts                      distinct font families (subset prefixes such as ABCDEF+ removed)
    body_font_size, min_font_size, small_text_chars   most common size, smallest size, chars under 7.5 pt
    margins_pt                 smallest left / right / top / bottom distance from text to the page edge
    overflow_chars             characters that reach past the page edge (cut off when printed)
    tight_margin               some text within 18 pt (a quarter inch) of an edge
    last_page_fill_pct         how far down the last page the text reaches
    last_page_nearly_empty     more than one page and the last one is under 20% filled
    missing_glyphs             '(cid:..)' or U+FFFD in the text: glyphs the font could not map
    issues                     the above, as short sentences

`render_pages(pdf, out_dir, prefix, max_pages)` writes PNG images of the first pages (pypdfium2 via
pdfplumber), which evaluate.py gives the evaluator model for the visual rubric.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

SMALL_FONT_PT = 7.5
TIGHT_MARGIN_PT = 18.0
NEARLY_EMPTY_PCT = 20.0
RENDER_DPI = 110  # A4 at 110 dpi is about 910 x 1290 px: enough to see spacing, about 1.1-1.5k image tokens


def _family(fontname: str | None) -> str:
    name = (fontname or "unknown").split("+", 1)[-1]
    return re.sub(r"[-,](Bold|Italic|Oblique|BoldItalic|BoldOblique|Regular|Roman|Medium|Light)$", "", name,
                  flags=re.IGNORECASE) or name


def layout_checks(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {"exists": False}
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = list(pdf.pages)
            if not pages:
                return {"exists": True, "pages": 0, "issues": ["the PDF has no pages"]}
            fonts, sizes = Counter(), Counter()
            margins = {"left": None, "right": None, "top": None, "bottom": None}
            overflow = small = 0
            text_parts, last_bottom, last_height = [], 0.0, pages[-1].height
            for i, page in enumerate(pages):
                w, h = page.width, page.height
                text_parts.append(page.extract_text() or "")
                for c in page.chars:
                    if not (c.get("text") or "").strip():
                        continue
                    fonts[_family(c.get("fontname"))] += 1
                    size = round(float(c.get("size") or 0) * 2) / 2
                    sizes[size] += 1
                    if size and size < SMALL_FONT_PT:
                        small += 1
                    x0, x1, top, bottom = c["x0"], c["x1"], c["top"], c["bottom"]
                    if x0 < -0.5 or x1 > w + 0.5 or top < -0.5 or bottom > h + 0.5:
                        overflow += 1
                    for key, value in (("left", x0), ("right", w - x1), ("top", top), ("bottom", h - bottom)):
                        margins[key] = value if margins[key] is None else min(margins[key], value)
                    if i == len(pages) - 1:
                        last_bottom = max(last_bottom, bottom)
    except Exception as e:  # noqa: BLE001 - an unreadable PDF is reported, not raised
        return {"exists": True, "error": f"{type(e).__name__}: {e}"}

    text = "\n".join(text_parts)
    missing = text.count("(cid:") + text.count("�")
    margins = {k: (round(v, 1) if v is not None else None) for k, v in margins.items()}
    fill = round(100 * last_bottom / last_height, 1) if last_height else None
    result = {
        "exists": True,
        "pages": len(pages),
        "page_size_pt": [round(pages[0].width, 1), round(pages[0].height, 1)],
        "fonts": [f for f, _ in fonts.most_common()],
        "body_font_size": sizes.most_common(1)[0][0] if sizes else None,
        "min_font_size": min(sizes) if sizes else None,
        "small_text_chars": small,
        "margins_pt": margins,
        "overflow_chars": overflow,
        "tight_margin": any(v is not None and v < TIGHT_MARGIN_PT for v in margins.values()),
        "last_page_fill_pct": fill,
        "last_page_nearly_empty": len(pages) > 1 and fill is not None and fill < NEARLY_EMPTY_PCT,
        "missing_glyphs": missing,
    }
    issues = []
    if overflow:
        issues.append(f"{overflow} characters reach past the page edge (cut off)")
    if result["tight_margin"]:
        issues.append(f"text within {TIGHT_MARGIN_PT:g} pt of a page edge (margins {margins})")
    if small:
        issues.append(f"{small} characters smaller than {SMALL_FONT_PT:g} pt")
    if len(result["fonts"]) > 3:
        issues.append(f"{len(result['fonts'])} font families: {', '.join(result['fonts'][:6])}")
    if result["last_page_nearly_empty"]:
        issues.append(f"the last page is only {fill:g}% filled")
    if missing:
        issues.append(f"{missing} glyphs could not be mapped to text")
    if not fonts:
        issues.append("no extractable text (image-only or empty PDF)")
    result["issues"] = issues
    return result


def render_pages(path: str | Path, out_dir: str | Path, prefix: str, max_pages: int, dpi: int = RENDER_DPI) -> list[Path]:
    """PNG images of the first max_pages pages: out_dir/<prefix>_page<N>.png. Empty if it cannot render."""
    path, out_dir = Path(path), Path(out_dir)
    if not path.is_file() or max_pages <= 0:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages], start=1):
                target = out_dir / f"{prefix}_page{i}.png"
                page.to_image(resolution=dpi).save(str(target), format="PNG")
                images.append(target)
    except Exception:  # noqa: BLE001 - no images: the evaluator is told the PDF could not be rendered
        return images
    return images

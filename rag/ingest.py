"""
Ingests the regulatory PDF corpus (rag/corpus/india_regulatory/) into a small local
vector store. Chunks per-page, with a regex pass that tries to recover a section/item
number for a more useful citation label (e.g. "Factories Act 1948, Section 36") --
falls back to a page-number citation if no clear numbered-section pattern is found.

These are real, sourced documents (the team found and verified them):
  - factories_act_1948.pdf       -- The Factories Act, 1948 (Act No. 63 of 1948), India
  - oisd_gdn_207_contractor_safety.pdf -- OISD-GDN-207 "Contractor Safety" (references
    OISD-STD-105 Work Permit System directly)
  - dgms_tech_circular_05_2020.pdf -- DGMS (Tech.) Circular No. 05 of 2020, "Safe Conduct
    of Drilling and Production operations in Oil & Gas Mines"

No fabricated regulation numbers -- every citation the orchestrator agent produces
traces back to one of these three real source files.

Run directly: `python -m rag.ingest`
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "rag" / "corpus" / "india_regulatory"
INDEX_PATH = REPO_ROOT / "rag" / "corpus" / "index.npz"
CHUNKS_PATH = REPO_ROOT / "rag" / "corpus" / "chunks.json"

DOC_LABELS = {
    "factories_act_1948.pdf": "The Factories Act, 1948 (Act No. 63 of 1948)",
    "oisd_gdn_207_contractor_safety.pdf": "OISD-GDN-207, Contractor Safety",
    "dgms_tech_circular_05_2020.pdf": "DGMS (Tech.) Circular No. 05 of 2020",
}

SECTION_PATTERNS = [
    re.compile(r"^\s*(\d{1,3}[A-Z]?)\.\s+([A-Z][a-zA-Z ,.'\-]{3,60})\.?\s*[—\-.]"),  # Factories Act: "36. Precautions..."
    re.compile(r"^\s*(\d{1,3})\.\s+[A-Z]"),  # DGMS numbered items: "16. A system for..."
]


@dataclass
class Chunk:
    text: str
    source: str
    citation: str
    page: int


def _find_citation(doc_label: str, page_text: str, page_num: int) -> str:
    for pattern in SECTION_PATTERNS:
        m = pattern.match(page_text.strip())
        if m:
            num = m.group(1)
            return f"{doc_label}, Section/Item {num} (p.{page_num})"
    return f"{doc_label} (p.{page_num})"


def _detect_embedded_doc_switch(text: str) -> str | None:
    """The oisd_gdn_207 PDF was extracted from a larger tender bundle and, from p.~55
    onward, actually contains a DIFFERENT document (an Engineers India Limited internal
    spec) appended after OISD-GDN-207 ends -- verified by reading the page headers
    directly. Mislabeling those pages as OISD-GDN-207 would be a fabricated citation,
    so detect the header and relabel."""
    if "STANDARD SPECIFICATION" in text.upper() and "6-82-0001" in text:
        return "EIL Standard Specification 6-82-0001 Rev.2, Health Safety & Environmental Management at Construction Sites"
    return None


ITEM_SPLIT_PATTERN = re.compile(r"\n(?=\d{1,3}\.\s+[A-Z])")


def _load_txt_chunks(txt_path: Path, doc_label: str) -> list[Chunk]:
    """DGMS circular source PDF is a scanned image with no text layer (verified: pypdf
    extract_text() returns empty for all 6 pages) -- the team transcribed it directly
    while reviewing the document, saved here as .txt, chunked by numbered item."""
    text = txt_path.read_text()
    items = ITEM_SPLIT_PATTERN.split(text)
    chunks = []
    for item in items:
        item = item.strip()
        if len(item) < 80:
            continue
        citation = _find_citation(doc_label, item, 1)
        chunks.append(Chunk(text=item, source=txt_path.name, citation=citation, page=1))
    return chunks


def load_chunks() -> list[Chunk]:
    chunks = []
    for pdf_path in sorted(CORPUS_DIR.glob("*.pdf")):
        default_label = DOC_LABELS.get(pdf_path.name, pdf_path.stem)
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if len(text) < 80:  # skip near-empty pages (covers, member lists, etc.)
                continue
            doc_label = _detect_embedded_doc_switch(text) or default_label
            citation = _find_citation(doc_label, text, i + 1)
            chunks.append(Chunk(text=text, source=pdf_path.name, citation=citation, page=i + 1))

    for txt_path in sorted(CORPUS_DIR.glob("*.txt")):
        doc_label = DOC_LABELS.get(txt_path.stem + ".pdf", txt_path.stem)
        chunks.extend(_load_txt_chunks(txt_path, doc_label))

    return chunks


def build_index():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} page-level chunks from {CORPUS_DIR}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([c.text for c in chunks], show_progress_bar=True, normalize_embeddings=True)

    np.savez(INDEX_PATH, embeddings=embeddings)
    CHUNKS_PATH.write_text(json.dumps([asdict(c) for c in chunks], indent=2))
    print(f"Wrote index to {INDEX_PATH} and chunk metadata to {CHUNKS_PATH}")


if __name__ == "__main__":
    build_index()

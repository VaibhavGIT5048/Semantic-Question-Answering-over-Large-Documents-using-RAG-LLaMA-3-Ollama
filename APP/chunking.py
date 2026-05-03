"""
╔══════════════════════════════════════════════════════════════════════╗
║  PHASE 2 — TEXT CHUNKING                                            ║
║  Goal: Split pages into retrieval-sized chunks with overlap.        ║
║  Input:  List[Document] from Phase 1 (one per page)                 ║
║  Output: List[Document] (many chunks, each with full metadata)      ║
╚══════════════════════════════════════════════════════════════════════╝

WHY CHUNKING EXISTS:
  The LLM has a context window limit (~4k–128k tokens depending on model).
  You cannot paste an entire 80-page PDF into the prompt. You must break
  the document into smaller pieces, retrieve only the RELEVANT pieces,
  and feed those to the LLM.

  But the size and quality of chunks directly determines retrieval quality.
  Too small → not enough context to answer. Too large → dilutes similarity.

  The goal: each chunk should be a SELF-CONTAINED unit of meaning.
"""

import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# THE CHUNKING STRATEGY — EXPLAINED
# ─────────────────────────────────────────────────────────────────────

"""
STRATEGY: RecursiveCharacterTextSplitter

HOW IT WORKS — Step by step:
  The splitter is given a PRIORITY LIST of separators:
      ["\n\n", "\n", ". ", " ", ""]

  It tries each separator in order:
    1. Try to split on "\n\n" (paragraph breaks)
       → If the resulting pieces fit inside chunk_size → done ✅
    2. If pieces are still too large, split on "\n" (line breaks)
    3. Then ". " (sentence boundaries)
    4. Then " " (word boundaries)
    5. Last resort: "" (raw characters — very rare)

WHY THIS IS BETTER THAN FIXED-SIZE SPLITTING:
  ┌──────────────────────────────────────────────────────┐
  │  Fixed:       "The tourism sector grew by 14% in 20" │
  │               "24, driven by..."                     │
  │               (cuts mid-word → breaks embeddings)    │
  │                                                      │
  │  Recursive:   "The tourism sector grew by 14% in     │
  │                2024, driven by..."                   │
  │               (respects paragraph → clean chunk)     │
  └──────────────────────────────────────────────────────┘

CHUNK SIZE CHOICE — chunk_size=1000 chars (~750 tokens):
  • Too small (<300 chars): chunk lacks context. Retrieval finds it but
    the LLM can't answer from it alone.
  • Too large (>3000 chars): similarity search gets noisy — a long chunk
    will match almost any query because it covers too many topics.
  • 800–1200 chars is the sweet spot for document Q&A.

OVERLAP CHOICE — chunk_overlap=150 chars (~15%):
  WHY OVERLAP AT ALL?
  Imagine a key sentence sits at the boundary of two chunks:
    Chunk 4 ends: "...The main recommendation is to invest in"
    Chunk 5 starts: "digital infrastructure across emerging markets."

  Without overlap, neither chunk 4 nor chunk 5 contains the full
  thought. With overlap=150, the last 150 chars of chunk 4 appear
  again at the start of chunk 5 → full sentence is preserved in chunk 5.

METADATA INHERITANCE:
  Every chunk inherits the metadata from its parent page Document.
  That means every chunk knows: source filename, page number, total pages.
  We ADD: chunk_id (unique), chunk_size (for quality gate).
"""

# ─────────────────────────────────────────────────────────────────────
# CORE FUNCTION
# ─────────────────────────────────────────────────────────────────────

def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Split a list of page-level Documents into smaller, overlapping chunks.

    Parameters:
        documents:     Output from Phase 1 (one Document per page)
        chunk_size:    Max characters per chunk (default 1000 ≈ 750 tokens)
        chunk_overlap: Characters repeated between adjacent chunks (default 150)

    Returns:
        List[Document] — each chunk has:
            page_content : the text slice
            metadata     : inherited (source, page) + added (chunk_id, chunk_size)
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order (most→least semantic)
        length_function=len,                        # measure in characters, not tokens
        is_separator_regex=False,                   # treat separators as plain strings
    )

    # split_documents preserves + inherits metadata from parent Documents
    chunks = splitter.split_documents(documents)

    # ── Enrich metadata on every chunk ───────────────────────────────
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]   = i                        # unique position in corpus
        chunk.metadata["chunk_size"] = len(chunk.page_content)  # actual char count
        # chunk.metadata["page"] already inherited from Phase 1 ✅
        # chunk.metadata["source"] already inherited from Phase 1 ✅

    # ── Summary ──────────────────────────────────────────────────────
    sizes = [len(c.page_content) for c in chunks]
    print(f"✅ [Phase 2] Chunking Complete")
    print(f"   Total chunks : {len(chunks)}")
    print(f"   Chunk size   : {chunk_size} chars (overlap: {chunk_overlap})")
    print(f"   Avg size     : {sum(sizes) // len(sizes)} chars")
    print(f"   Min size     : {min(sizes)} chars")
    print(f"   Max size     : {max(sizes)} chars")

    return chunks


# ─────────────────────────────────────────────────────────────────────
# HELPER: Visualize overlap between two adjacent chunks
# ─────────────────────────────────────────────────────────────────────

def show_overlap(chunks: list[Document], chunk_index: int = 0):
    """
    Print two adjacent chunks and highlight the overlapping text.
    Use this to VERIFY the overlap is working correctly.
    """
    if chunk_index + 1 >= len(chunks):
        print("Not enough chunks to show overlap.")
        return

    a = chunks[chunk_index].page_content
    b = chunks[chunk_index + 1].page_content

    # Find the overlapping suffix of chunk A in the prefix of chunk B
    overlap_text = ""
    for length in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:length]):
            overlap_text = b[:length]
            break

    print(f"\n{'═'*60}")
    print(f"  OVERLAP CHECK: chunk {chunk_index} → chunk {chunk_index + 1}")
    print(f"{'═'*60}")
    print(f"\n[Chunk {chunk_index}] (last 200 chars):")
    print(f"  ...{a[-200:]}")
    print(f"\n[Chunk {chunk_index + 1}] (first 200 chars):")
    print(f"  {b[:200]}...")
    if overlap_text:
        print(f"\n✅ Overlapping text ({len(overlap_text)} chars):")
        print(f"  '{overlap_text[:150]}'")
    else:
        print(f"\n⚠️  No overlap detected — check chunk_overlap setting")
    print(f"\n{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────
# HELPER: Preview a few chunks for quality inspection
# ─────────────────────────────────────────────────────────────────────

def preview_chunks(chunks: list[Document], n: int = 3):
    print(f"\n{'═'*60}")
    print(f"  CHUNK PREVIEW (first {n})")
    print(f"{'═'*60}")
    for chunk in chunks[:n]:
        m = chunk.metadata
        print(f"\n── chunk_id={m['chunk_id']} | page={m['page']} | size={m['chunk_size']} chars ──")
        print(chunk.page_content[:400])
        if m['chunk_size'] > 400:
            print(f"  ... [{m['chunk_size'] - 400} more chars]")
    print(f"\n{'═'*60}\n")


def save_chunks_jsonl(chunks: list[Document], output_path: str) -> None:
    """Save chunks as JSON Lines with content and metadata."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────
# RUN AS SCRIPT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from APP.pdf_loading import load_pdf

    def choose_pdf() -> Path:
        pdf_dir = Path("data")
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError("No PDFs found in the data folder.")

        print("Available PDFs:")
        for idx, pdf in enumerate(pdfs, start=1):
            print(f"  {idx}. {pdf.name}")

        selection = input("Select a PDF number (or press Enter to cancel): ").strip()
        if not selection:
            raise SystemExit("Cancelled.")

        try:
            return pdfs[int(selection) - 1]
        except (ValueError, IndexError):
            raise SystemExit("Invalid selection.")

    parser = argparse.ArgumentParser(description="Load a PDF, chunk it, and preview results.")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file")
    parser.add_argument("--preview", type=int, default=3, help="Number of chunks to preview")
    parser.add_argument("--overlap-check", action="store_true", help="Show overlap for chunk 0->1")
    parser.add_argument("--output", type=str, default="chunks/chunks.jsonl", help="Path to save chunks as JSONL")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path) if args.pdf_path else choose_pdf()

    documents = load_pdf(str(pdf_path))
    chunks = chunk_documents(documents)

    preview_chunks(chunks, n=args.preview)
    if args.overlap_check:
        show_overlap(chunks, chunk_index=0)

    save_chunks_jsonl(chunks, args.output)
    print(f"Saved chunks to {args.output}")

    # WHAT TO CHECK:
    # 1. Does each chunk read like a complete thought?
    #    If it ends mid-sentence → try smaller chunk_size or add ". " separator
    # 2. Is overlap visible between adjacent chunks?
    #    Run show_overlap() and confirm the shared text is meaningful
    # 3. Are chunk sizes reasonable? (300–1200 chars = healthy range)
    #    If many chunks are tiny (<200 chars) → your PDF has a lot of
    #    short isolated lines (headers, captions) — consider merging small chunks
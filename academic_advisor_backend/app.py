# app.py — single entry point: ingest PDFs → extract majors → build vectors → run bagrut + CLI
import os
import json
from typing import Optional
from dotenv import load_dotenv

from ingestion.ingest import PDFIngestor
from ingestion.majors_extractor import build_majors_profiles
from embeddings.majors_embeddings import build_majors_embeddings
from ingestion.bagrut_reader import BagrutReader
from ingestion.pdf_processor import json_payload_len
from query.query import HybridRAG
from query.bagrut_features import load_bagrut, bagrut_signals


# ---------- Paths (match your repo tree) ----------
MAJORS_DATA_DIR = "majors_data"   # majors.pdf + extracted_majors.json + majors_embeddings.json live here
STATE_DIR       = "state"         # bagrut_example.pdf + extracted_bagrut.json + ingested_files.json + person state

EXTRACTED_JSON        = os.path.join(MAJORS_DATA_DIR, "extracted_majors.json")
EMBEDDINGS_JSON       = os.path.join(MAJORS_DATA_DIR, "majors_embeddings.json")
BAGRUT_JSON           = os.path.join(STATE_DIR, "extracted_bagrut.json")
INGESTED_FILES_JSON   = os.path.join(STATE_DIR, "ingested_files.json")

MAJORS_PDF = os.path.join(MAJORS_DATA_DIR, "majors.pdf")
BAGRUT_PDF = os.path.join(STATE_DIR, "bagrut_example.pdf")

# ---------- Vector DB / Models ----------
PERSIST_DIR  = os.getenv("PERSIST_DIR", "chroma_db")
COLLECTION   = os.getenv("COLLECTION_NAME", "pdf_collection")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def _newer(src: str, dst: str) -> bool:
    """True if dst is missing or older than src (i.e., src is newer → should rebuild)."""
    if not os.path.exists(dst):
        return True
    try:
        return os.path.getmtime(dst) < os.path.getmtime(src)
    except OSError:
        return True


def main():
    load_dotenv()

    # Make Bagrut path discoverable by helpers that rely on env (if any)
    os.environ["BAGRUT_JSON"] = BAGRUT_JSON

    # ---------- 1) Ingest PDFs (RAG explanations context) ----------
    scan_dirs = [d for d in (MAJORS_DATA_DIR, STATE_DIR) if os.path.isdir(d)]
    print(f"[index] Scanning folders: {', '.join(scan_dirs) if scan_dirs else '(none)'}")

    total_pdfs = total_pages = total_chunks = 0
    new_chunks = []

    for scan_dir in scan_dirs:
        try:
            ingestor = PDFIngestor(pdf_dir=scan_dir, meta_file=INGESTED_FILES_JSON, verbose=False)
            docs, stats = ingestor.ingest_new(return_stats=True)
            new_chunks.extend(docs)
            total_pdfs  += stats.get("pdfs", 0)
            total_pages += stats.get("pages", 0)
            total_chunks+= stats.get("chunks", 0)
        except TypeError:
            # Back-compat if your installed ingest doesn't support return_stats yet
            ingestor = PDFIngestor(pdf_dir=scan_dir, meta_file=INGESTED_FILES_JSON)
            docs = ingestor.ingest_new()
            new_chunks.extend(docs)

    if total_pdfs == 0 and len(new_chunks) == 0:
        print("[ingest] No new PDFs found in any folder.")
    else:
        if total_pdfs == 0 and total_pages == 0 and total_chunks == 0:
            print(f"[ingest] Added {len(new_chunks)} chunks from new PDFs.")
        else:
            print(f"[ingest] New PDFs: {total_pdfs} | Pages: {total_pages} | Chunks: {total_chunks}")

    # ---------- 2) Bagrut extraction ----------
    if os.path.exists(BAGRUT_PDF):
        already = os.path.exists(BAGRUT_JSON)
        count = json_payload_len(BAGRUT_JSON, prefer="items") if already else 0
        needs_refresh = _newer(BAGRUT_PDF, BAGRUT_JSON)
        if already and count > 0 and not needs_refresh:
            print(f"[bagrut] Using existing {BAGRUT_JSON} ({count} subjects).")
        else:
            try:
                print(f"[bagrut] Extracting grades from {BAGRUT_PDF} ...")
                reader = BagrutReader(strict_star_only=True, use_llm_fallback=True, debug=True)
                items, summary = reader.run(BAGRUT_PDF, out_json=BAGRUT_JSON)
                print(f"[bagrut] Saved {len(items)} subjects -> {BAGRUT_JSON}")
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"[bagrut:error] {e}")
    else:
        print(f"[bagrut] No {os.path.basename(BAGRUT_PDF)} found under {STATE_DIR}")

    # ---------- 3) Compute Bagrut signals (once) & init RAG ----------
    try:
        try:
            _bjson = load_bagrut(BAGRUT_JSON)  # try path-first
        except TypeError:
            _bjson = load_bagrut()            # fallback to env-based
        _b_sig = bagrut_signals(_bjson) or {}
    except Exception:
        _b_sig = {}

    rag = HybridRAG(
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION,
        embedding_model=EMBED_MODEL,
        llm_model=GEMINI_MODEL,
        k=int(os.getenv("RAG_K", "3")),
        bagrut_sig=_b_sig,  # pass down to generator/recommender
    )

    if new_chunks:
        try:
            rag.upsert_chunks(new_chunks)
            print("[app] Upsert done.")
        except Exception as e:
            print(f"[warn] Upsert failed: {e}")

    # ---------- 4) Extract majors JSON ----------
    if os.path.exists(MAJORS_PDF):
        if os.path.exists(EXTRACTED_JSON):
            wrote = json_payload_len(EXTRACTED_JSON, prefer="majors")
            print(f"[app] Using existing {EXTRACTED_JSON} (majors: {wrote}).")
            if wrote == 0:
                print("[warn] Existing extracted file has 0 majors. Delete it to trigger a fresh extraction.")
        else:
            print(f"[app] Extracting majors from: {MAJORS_PDF} -> {EXTRACTED_JSON}")
            try:
                build_majors_profiles(MAJORS_PDF, out_json=EXTRACTED_JSON)
            except TypeError:
                build_majors_profiles(MAJORS_PDF, EXTRACTED_JSON)
            wrote = json_payload_len(EXTRACTED_JSON, prefer="majors")
            print(f"[majors_extractor] Wrote {wrote} majors -> {EXTRACTED_JSON}")
            if wrote == 0:
                print("[warn] Extractor returned 0 majors. Ensure GOOGLE_API_KEY is set and PDF text is readable.")
    else:
        print(f"[hint] No majors PDF found at {MAJORS_PDF}")

    # ---------- 5) Build embeddings JSON ----------
    if os.path.exists(EMBEDDINGS_JSON):
        emb_n = json_payload_len(EMBEDDINGS_JSON, prefer="majors")
        print(f"[app] Using existing {EMBEDDINGS_JSON} (majors: {emb_n}).")
    elif os.path.exists(EXTRACTED_JSON):
        print(f"[app] Building majors embeddings from: {EXTRACTED_JSON}")
        try:
            build_majors_embeddings(EXTRACTED_JSON, EMBEDDINGS_JSON)
        except TypeError:
            build_majors_embeddings(EXTRACTED_JSON)
        emb_n = json_payload_len(EMBEDDINGS_JSON, prefer="majors")
        print(f"[embed] Wrote {emb_n} majors -> {EMBEDDINGS_JSON}")
    else:
        print("[warn] Neither extracted_majors.json nor a majors PDF is available; recommendations will be empty.")

    # ---------- 6) Load majors ----------
    loaded = rag.load_majors_rubric_vectors(EMBEDDINGS_JSON) if os.path.exists(EMBEDDINGS_JSON) else 0
    print(f"[app] Loaded {loaded} majors into memory.")

    # ---------- 7) CLI ----------
    print("\nReady! Ask about your PDFs or anything else (type 'exit' to quit).")
    print("Tip: type 'interview' to start the AI-driven advisor interview.")
    while True:
        try:
            q = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if q.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if q.lower() in {"interview"}:
            try:
                rag.run_interview()
            except Exception as e:
                print(f"[error] Interview failed: {e}")
            continue
        if not q:
            continue
        try:
            answer = rag.ask(q)
        except Exception as e:
            answer = f"(Error while answering: {e})"
        print("\n--- Answer ---")
        print(answer)


if __name__ == "__main__":
    main()

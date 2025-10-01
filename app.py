# app.py — single entry point: ingest PDFs → extract majors → build vectors → run bagrut + CLI
import json, re, os
from typing import Optional
from dotenv import load_dotenv
from ingestion.ingest import PDFIngestor
from ingestion.majors_extractor import build_majors_profiles
from ingestion.majors_embeddings import build_majors_embeddings
from ingestion.bagrut_reader import BagrutReader
from ingestion.pdf_processor import json_payload_len 
from query.query import HybridRAG

DATA_DIR = "data"
EXTRACTED_JSON = "extracted_majors.json"
EMBEDDINGS_JSON = "majors_embeddings.json"
BAGRUT_JSON = "extracted_bagrut.json"
INGESTED_FILES_JASON = "ingested_files.json"  
MAJORS_PDF = "majors.pdf"
BAGRUT_PDF = "bagrut_example.pdf"
PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_db")
COLLECTION = os.getenv("COLLECTION_NAME", "pdf_collection")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
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

    # ---------- 1) Ingest PDFs (RAG) ----------
    ingestor = PDFIngestor(pdf_dir=DATA_DIR, meta_file=INGESTED_FILES_JASON)
    try:
        new_chunks = ingestor.ingest_new()  # discover and split all new PDFs together
    except Exception as e:
        print(f"[ingest:error] {e}")
        new_chunks = []

    # ---------- 2) Bagrut extraction (run after ingest; no pre-ingest) ----------
    bagrut_pdf = os.path.join(DATA_DIR, BAGRUT_PDF)
    if os.path.exists(bagrut_pdf):
        already = os.path.exists(BAGRUT_JSON)
        count = json_payload_len(BAGRUT_JSON, prefer="items") if already else 0
        needs_refresh = _newer(bagrut_pdf, BAGRUT_JSON)  # True if PDF newer or JSON missing
        if already and count > 0 and not needs_refresh:
            print(f"[bagrut] Using existing {BAGRUT_JSON} ({count} subjects).")
        else:
            try:
                print(f"[bagrut] Extracting grades from {bagrut_pdf} ...")
                reader = BagrutReader(strict_star_only=True, use_llm_fallback=True, debug=True)
                items, summary = reader.run(bagrut_pdf, out_json=BAGRUT_JSON)
                print(f"[bagrut] Saved {len(items)} subjects -> {BAGRUT_JSON}")
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"[bagrut:error] {e}")
    else:
        print(f"[bagrut] No {BAGRUT_PDF} found under {DATA_DIR}")

    # ---------- 3) RAG init + upsert ----------
    rag = HybridRAG(
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION,
        embedding_model=EMBED_MODEL,
        llm_model=GEMINI_MODEL,
        k=int(os.getenv("RAG_K", "3")),
    )
    if new_chunks:
        try:
            rag.upsert_chunks(new_chunks)
            print("[app] Upsert done.")
        except Exception as e:
            print(f"[warn] Upsert failed: {e}")

    # ---------- 4) Extract majors JSON ----------
    majors_pdf = os.path.exists(MAJORS_PDF)
    if majors_pdf:
        if os.path.exists(EXTRACTED_JSON):
            wrote = json_payload_len(EXTRACTED_JSON, prefer="majors")
            print(f"[app] Using existing {EXTRACTED_JSON} (majors: {wrote}).")
            if wrote == 0:
                print("[warn] Existing extracted file has 0 majors. Delete it to trigger a fresh extraction.")
        else:
            print(f"[app] Extracting majors from: {majors_pdf} -> {EXTRACTED_JSON}")
            try:
                build_majors_profiles(majors_pdf, out_json=EXTRACTED_JSON)
            except TypeError:
                # backward-compat with (pdf, out_json) vs (pdf, out_path) signatures
                build_majors_profiles(majors_pdf, EXTRACTED_JSON)
            wrote = json_payload_len(EXTRACTED_JSON, prefer="majors")
            print(f"[majors_extractor] Wrote {wrote} majors -> {EXTRACTED_JSON}")
            if wrote == 0:
                print("[warn] Extractor returned 0 majors. Ensure GOOGLE_API_KEY is set and PDF text is readable.")
    else:
        print("[hint] No majors PDF found. Put one at data/majors.pdf or set MAJORS_PDF in .env")

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

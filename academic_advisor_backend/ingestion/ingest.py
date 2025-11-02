# ingestion/ingest.py
import os
import json
from typing import List, Tuple, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFIngestor:
    """
    Handles ingestion of PDF files into the system:
      - Detects new PDF files in a directory.
      - Loads their content (page by page).
      - Splits text into manageable chunks for embedding/vector storage.
      - Keeps track of which files have already been ingested (via a metadata file).
    """

    def __init__(self, pdf_dir: str = "state", meta_file: str = "ingested_files.json", verbose: bool = True):
        """
        Initialize the PDF ingestor.

        Args:
            pdf_dir (str): Directory where PDFs are stored.
            meta_file (str): JSON file that records which PDFs have already been ingested.
            verbose (bool): Whether to print progress logs.
        """
        self.pdf_dir = pdf_dir
        self.meta_file = meta_file
        self.verbose = verbose

        if os.path.exists(self.meta_file):
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self.ingested_files = set(json.load(f))
            except Exception:
                self.ingested_files = set()
        else:
            self.ingested_files = set()

        # Recursive splitter ensures natural breakpoints (paragraphs, sentences).
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,   # max characters per chunk
            chunk_overlap=200  # overlap ensures context continuity
        )

    def _log(self, msg: str, *, v: Optional[bool] = None) -> None:
        if (self.verbose if v is None else v):
            print(msg)

    def _save_ingested(self) -> None:
        """Save the list of ingested files into the metadata JSON file."""
        try:
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump(sorted(list(self.ingested_files)), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"[ingest] Failed to save metadata: {e}")

    def _load_new_pdfs(self) -> Tuple[List[Any], List[str]]:
        """
        Load new PDFs that haven’t been ingested yet.

        Returns:
            Tuple[List, List[str]]:
                - A list of LangChain Documents (text + metadata).
                - A list of filenames of the new PDFs found.
        """
        docs, new_files = [], []
        if not os.path.isdir(self.pdf_dir):
            self._log(f"[ingest] PDF directory '{self.pdf_dir}' not found.")
            return docs, new_files

        for fname in os.listdir(self.pdf_dir):
            if fname.lower().endswith(".pdf") and fname not in self.ingested_files:
                path = os.path.join(self.pdf_dir, fname)
                try:
                    loader = PyPDFLoader(path)
                    file_docs = loader.load()  # list of per-page docs

                    for d in file_docs:
                        d.metadata = d.metadata or {}
                        d.metadata["source"] = fname

                    docs.extend(file_docs)
                    new_files.append(fname)
                except Exception as e:
                    self._log(f"[ingest] Failed to load {fname}: {e}")
        return docs, new_files

    def ingest_new(self, *, verbose: Optional[bool] = None, return_stats: bool = False):
        """
        Ingest new PDFs into the system.

        Steps:
            1. Look for PDFs in the directory that haven't been ingested yet.
            2. Load their pages as documents.
            3. Split documents into smaller chunks (for embeddings).
            4. Update metadata file to mark these PDFs as processed.

        Args:
            verbose: override instance-level verbosity for this call.
            return_stats: if True, returns (chunks, stats_dict).

        Returns:
            List[Document] or (List[Document], Dict[str,int])
        """
        v = self.verbose if verbose is None else verbose
        if v:
            print("[ingest] Checking for new PDFs...")

        docs, new_files = self._load_new_pdfs()

        if not docs:
            if v:
                print("[ingest] No new PDFs to ingest.")
            if return_stats:
                return [], {"pdfs": 0, "pages": 0, "chunks": 0}
            return []

        total_pages = len(docs)
        if v:
            print(f"[ingest] Found {len(new_files)} new PDFs, total pages: {total_pages}")

        chunks = self.splitter.split_documents(docs)
        if v:
            print(f"[ingest] Split into {len(chunks)} chunks.")

        # Update ingested metadata
        self.ingested_files.update(new_files)
        self._save_ingested()

        if return_stats:
            return chunks, {"pdfs": len(new_files), "pages": total_pages, "chunks": len(chunks)}
        return chunks

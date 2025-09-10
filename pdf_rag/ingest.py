# pdf_rag/ingest.py
import os
import json
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFIngestor:
    def __init__(self, pdf_dir: str = "data", meta_file: str = "ingested_files.json"):
        self.pdf_dir = pdf_dir
        self.meta_file = meta_file
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.ingested_files = set(json.load(f))
        else:
            self.ingested_files = set()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def _save_ingested(self):
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(sorted(list(self.ingested_files)), f, ensure_ascii=False, indent=2)

    def _load_new_pdfs(self) -> Tuple[List, List[str]]:
        docs, new_files = [], []
        if not os.path.isdir(self.pdf_dir):
            print(f"[ingest] PDF directory '{self.pdf_dir}' not found.")
            return docs, new_files
        for fname in os.listdir(self.pdf_dir):
            if fname.lower().endswith(".pdf") and fname not in self.ingested_files:
                path = os.path.join(self.pdf_dir, fname)
                try:
                    loader = PyPDFLoader(path)
                    file_docs = loader.load()
                    for d in file_docs:
                        d.metadata = d.metadata or {}
                        d.metadata["source"] = fname
                    docs.extend(file_docs)
                    new_files.append(fname)
                except Exception as e:
                    print(f"[ingest] Failed to load {fname}: {e}")
        return docs, new_files

    def ingest_new(self):
        print("[ingest] Checking for new PDFs...")
        docs, new_files = self._load_new_pdfs()
        if not docs:
            print("[ingest] No new PDFs to ingest.")
            return []
        print(f"[ingest] Found {len(new_files)} new PDFs, total pages: {len(docs)}")
        chunks = self.splitter.split_documents(docs)
        print(f"[ingest] Split into {len(chunks)} chunks.")
        self.ingested_files.update(new_files)
        self._save_ingested()
        return chunks

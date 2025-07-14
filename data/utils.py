import os
from typing import List
import uuid
from langchain_core.documents import Document
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_fixed_sized_chunks(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_splits = text_splitter.split_documents(docs)

    for i, split in enumerate(all_splits):
        split.metadata["chunk_id"] = f"{str(uuid.uuid4())}_{i}"
        
    return all_splits


def load_pdfs_from_directory(pdf_directory: str) -> List[Document]:
    """Load all PDF files from a directory."""
    pdf_path = Path(pdf_directory)
    all_docs = []

    # Get all PDF files from the directory
    pdf_files = list(pdf_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: {pdf_directory}")

    print(f"Found {len(pdf_files)} PDF files to process...")

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        # Add source metadata to each document
        for doc in docs:
            doc.metadata["source_file"] = pdf_file.name
            doc.metadata["source_path"] = str(pdf_file)

        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} pages from {len(pdf_files)} PDF files")
    return all_docs


def load_specific_pdfs(pdf_paths: List[str]) -> List[Document]:
    """Load specific PDF files from a list of paths."""
    all_docs = []

    print(f"Loading {len(pdf_paths)} PDF files...")

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found: {pdf_path}")
            continue

        print(f"Loading: {os.path.basename(pdf_path)}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Add source metadata to each document
        for doc in docs:
            doc.metadata["source_file"] = os.path.basename(pdf_path)
            doc.metadata["source_path"] = pdf_path

        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} pages from PDF files")
    return all_docs

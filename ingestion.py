import os
from pathlib import Path
from pypdf import PdfReader

import chromadb

from config import CHROMA_PATH

PDF_DOCUMENTS_PATH = Path("pdf_documents")

os.environ["ANONYMIZED_TELEMETRY"] = "false"


DOCUMENTS_PATH = Path("documents")


def chunk_text(
    text,
    chunk_size=500,
    overlap=50
):

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []

    current_chunk = []
    current_size = 0

    for paragraph in paragraphs:

        paragraph_words = paragraph.split()
        paragraph_size = len(
            paragraph_words
        )

        # If the current chunk can accept
        # the paragraph, add it.
        if (
            current_size + paragraph_size
            <= chunk_size
        ):

            current_chunk.append(
                paragraph
            )

            current_size += (
                paragraph_size
            )

        else:

            # Save current chunk
            if current_chunk:

                chunks.append(
                    "\n\n".join(
                        current_chunk
                    )
                )

            # Start a new chunk
            current_chunk = [
                paragraph
            ]

            current_size = paragraph_size

    # Save final chunk
    if current_chunk:

        chunks.append(
            "\n\n".join(
                current_chunk
            )
        )

    return chunks


def create_chroma_collection(collection_name="company_policies"):

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=collection_name)

    ingest_documents(collection)
    ingest_pdf_documents(collection)

    return client, collection


def print_results(results):
    for document, metadata in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "N/A")

        print(f"\nSource: {source}")
        print(f"Page: {page}")
        print(document)


def ingest_documents(
    collection
):

    for file_path in DOCUMENTS_PATH.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        chunks = chunk_text(
            text,
            chunk_size=100,
            overlap=20
        )

        for index, chunk in enumerate(chunks):

            document_id = (
                f"{file_path.stem}_chunk_{index}"
            )

            existing = collection.get(
                ids=[document_id]
            )

            if existing["ids"]:
                continue

            base_metadata =     get_document_metadata(
                file_path
            )

            collection.add(
                documents=[chunk],
                ids=[document_id],
                metadatas=[
                    {
                        "source": file_path.name,
                        "chunk_index": chunk_index,
                        "file_type": "txt",
                        **base_metadata
                    }
                ]
            )

def ingest_pdf_documents(collection):

    for file_path in PDF_DOCUMENTS_PATH.glob("*.pdf"):

        try:
            reader = PdfReader(str(file_path))
        except Exception as e:
            print(f"  [warn] Skipping unreadable PDF {file_path.name}: {e}")
            continue

        for page_number, page in enumerate(reader.pages, start=1):

            try:
                text = page.extract_text()
            except Exception as e:
                print(f"  [warn] Failed to extract text from {file_path.name} p.{page_number}: {e}")
                continue

            if not text:
                continue

            chunks = chunk_text(text, chunk_size=100, overlap=20)

            for chunk_index, chunk in enumerate(chunks):

                document_id = (
                    f"{file_path.stem}"
                    f"_page_{page_number}"
                    f"_chunk_{chunk_index}"
                )

                existing = collection.get(ids=[document_id])

                if existing["ids"]:
                    continue

                base_metadata = get_document_metadata(
                    file_path
                )        
                collection.add(
                    documents=[chunk],
                    ids=[document_id],
                    metadatas=[
                        {
                            "source": file_path.name,
                            "page": page_number,
                            "chunk_index": chunk_index,
                            "file_type": "pdf",
                            **base_metadata
                        }
                    ]
                )

def get_document_metadata(file_path):

    filename = file_path.name.lower()

    if "leave" in filename:

        return {
            "department": "HR",
            "policy_type": "leave"
        }

    if "remote" in filename:

        return {
            "department": "IT",
            "policy_type": "remote_work"
        }

    return {
        "department": "unknown",
        "policy_type": "unknown"
    }
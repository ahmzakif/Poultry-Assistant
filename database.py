from __future__ import annotations

from pathlib import Path
from typing import Final, List

from tqdm.auto import tqdm
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings

from model import get_embeddings

_INDEX_DIR: Final[Path] = Path("data/faiss_index")


def _load_pdfs(folder: Path) -> List[str]:
    loader = DirectoryLoader(
        str(folder), glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True
    )
    return loader.load()


def create_vector_store(
    folder_path: str | Path,
    *,
    chunk_size: int = 10_000,
    chunk_overlap: int = 2_000,
    index_dir: str | Path = _INDEX_DIR,
    embeddings: Embeddings | None = None,
) -> FAISS:
    """Bangun index FAISS dari kumpulan PDF."""
    folder_path, index_dir = Path(folder_path), Path(index_dir)
    embeddings = embeddings or get_embeddings()

    docs = _load_pdfs(folder_path)
    print(f"Loaded {len(docs)} PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )

    chunks = []
    for doc in tqdm(docs, desc="Splitting"):
        chunks.extend(splitter.split_documents([doc]))

    print(f"Storing {len(chunks):,} chunks in FAISS…")
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vs.save_local(str(index_dir))
    return vs


def load_vector_store(
    index_dir: str | Path = _INDEX_DIR, embeddings: Embeddings | None = None
) -> FAISS:
    """Muat FAISS vector-store yang sudah tersimpan."""
    embeddings = embeddings or get_embeddings()
    return FAISS.load_local(
        str(index_dir), embeddings=embeddings, allow_dangerous_deserialization=True
    )

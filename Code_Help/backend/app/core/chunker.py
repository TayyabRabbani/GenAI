# app/core/chunker.py
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

def split_code(code: str, chunk_size: int = 300) -> List[str]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    return splitter.split_text(code)

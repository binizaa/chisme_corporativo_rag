# data_loader.py

import os
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import Colors, DOCUMENT_URL

def load_documents_with_fallback(path_or_url: str) -> list[Document]:
    """
    Carga documentos desde un archivo .txt local o desde una URL.
    Si falla, crea un documento de respaldo.
    """
    documents = []
    print(f"{Colors.HEADER}--- 1. Carga de Documentos ---{Colors.RESET}")

    try:
        # Lógica de carga... (código idéntico al que ya tenías en rag_core.py)
        if path_or_url.endswith(".txt") and os.path.exists(path_or_url):
            loader = TextLoader(path_or_url, encoding="utf-8")
            documents = loader.load()
            print(f"{Colors.OKGREEN}   -> TXT cargado exitosamente. Total: {len(documents)}.{Colors.RESET}")
        else:
            loader = WebBaseLoader(path_or_url)
            documents = loader.load()
            print(f"{Colors.OKGREEN}   -> URL cargada exitosamente. Total: {len(documents)}.{Colors.RESET}")

        if not documents:
            raise Exception("El recurso cargado no tiene contenido útil.")

    except Exception as e:
        print(f"{Colors.WARNING}   -> ADVERTENCIA al cargar el recurso: {e}. Usando documento de respaldo.{Colors.RESET}")
        documents = [
            Document(
                page_content="""
                Documento de respaldo: información básica de Roborregos.
                """,
                metadata={"source": "documento-ejemplo-local"}
            )
        ]
        print(f"{Colors.OKGREEN}   -> Documento de respaldo creado. Total: 1.{Colors.RESET}")

    return documents

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Divide los documentos en fragmentos (chunks) para el Vector Store.
    """
    print(f"\n{Colors.HEADER}--- 2. División en Chunks ---{Colors.RESET}")
    if not documents:
        print(f"{Colors.FAIL}{Colors.BOLD}   -> ¡ADVERTENCIA! No hay documentos para procesar.{Colors.RESET}")
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Documentos divididos. {Colors.BOLD}Total de chunks: {len(chunks)}.{Colors.RESET}")
    return chunks
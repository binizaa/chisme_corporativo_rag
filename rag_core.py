# rag_core.py

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from utils import Colors, OLLAMA_MODEL, DOCUMENT_URL
import os

def load_documents_with_fallback(path_or_url: str) -> list[Document]:
    """
    Carga documentos desde un archivo .txt local o desde una URL.
    Si falla, crea un documento de respaldo.
    """
    documents = []
    print(f"{Colors.HEADER}--- 1. Carga de Documentos ---{Colors.RESET}")

    try:
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
                Documento de respaldo: información básica de Roborregos y del modelo llama3.
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

def setup_rag_chain(chunks: list[Document]):
    """
    Inicializa el Vector Store y configura la cadena RAG completa.
    """
    print(f"\n{Colors.HEADER}--- 3. Inicialización de FAISS y Embeddings ---{Colors.RESET}")

    if not chunks:
        print(f"{Colors.FAIL}\n🚨 ERROR FATAL: No se puede crear el Vector Store porque no hay CHUNKS (piezas de texto).{Colors.RESET}")
        raise ValueError("La lista de chunks está vacía. Terminando el programa.")

    # 4. Inicialización de Embeddings y Vector Store
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"{Colors.OKGREEN}   -> Vector Store FAISS creado exitosamente con el modelo {OLLAMA_MODEL}.{Colors.RESET}")

    # 5. Configuración de la Cadena RAG
    llm = Ollama(model=OLLAMA_MODEL)

    prompt = ChatPromptTemplate.from_template("""
        Responde a la pregunta basándote únicamente en el contexto proporcionado.
        Si la respuesta no se encuentra en el contexto, di amablemente que no tienes la información.
        
        Contexto: {context}
        
        Pregunta: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain
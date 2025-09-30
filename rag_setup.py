# rag_setup.py

import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from config import Colors, OLLAMA_MODEL, CHROMA_DB_DIR

def setup_rag_chain(chunks: list[Document]):
    """
    Inicializa el Vector Store con ChromaDB (persistente) y configura la cadena RAG.
    """
    print(f"\n{Colors.HEADER}--- 3. Inicialización de ChromaDB y Embeddings ---{Colors.RESET}")
    
    if not chunks and not os.path.exists(CHROMA_DB_DIR):
        print(f"{Colors.FAIL}\n🚨 ERROR FATAL: No hay CHUNKS y el Vector Store no existe en disco. Terminando el programa.{Colors.RESET}")
        raise ValueError("La lista de chunks está vacía y no hay una DB persistente para cargar.")

    # 4. Inicialización de Embeddings 
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    vector_store = None

    if os.path.exists(CHROMA_DB_DIR):
        # Cargar la DB existente
        print(f"   -> {Colors.OKBLUE}Cargando Vector Store ChromaDB existente desde: {CHROMA_DB_DIR}{Colors.RESET}")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR, 
            embedding_function=embeddings
        )
        print(f"{Colors.OKGREEN}   -> Vector Store ChromaDB cargado exitosamente.{Colors.RESET}")
    else:
        # Crear la DB por primera vez
        print(f"   -> {Colors.WARNING}Directorio '{CHROMA_DB_DIR}' no encontrado. Creando Vector Store desde los chunks...{Colors.RESET}")
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=CHROMA_DB_DIR
        )
        vector_store.persist() 
        print(f"{Colors.OKGREEN}   -> Vector Store ChromaDB creado y persistido exitosamente con el modelo {OLLAMA_MODEL}.{Colors.RESET}")

    # 5. Configuración de la Cadena RAG
    llm = OllamaLLM(model=OLLAMA_MODEL) 

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
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# NUEVAS IMPORTACIONES para Hybrid Retrieval
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from config import Colors, OLLAMA_MODEL, CHROMA_DB_DIR

def setup_rag_chain(chunks: list[Document]):
    """
    Inicializa el Vector Store con ChromaDB (persistente) y configura la cadena RAG.
    Implementa Recuperación Híbrida (Hybrid Retrieval) combinando Embeddings y BM25.
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
        
        if not chunks:
             print(f"   -> {Colors.WARNING}⚠️ Aviso: La lista de chunks está vacía. Solo se usará el recuperador de Embeddings.{Colors.RESET}")

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


    # 5. Configuración de la Cadena RAG y Hybrid Retrieval
    llm = OllamaLLM(model=OLLAMA_MODEL) 

    # --- Configuración del Hybrid Retrieval ---
    if chunks:
        print(f"\n{Colors.HEADER}--- 5. Configuración del Hybrid Retrieval ---{Colors.RESET}")
        # A. Recuperador basado en Embeddings (Vector Store)
        # Se busca k=5 documentos semánticamente similares
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5}) 
        print(f"   -> {Colors.OKBLUE}Recuperador de Embeddings (ChromaDB) listo.{Colors.RESET}")

        # B. Recuperador basado en BM25 (Keywords)
        # Se busca k=5 documentos con coincidencia de palabras clave
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 5 
        print(f"   -> {Colors.OKBLUE}Recuperador BM25 (Keywords) listo.{Colors.RESET}")

        # C. Combinar ambos recuperadores con EnsembleRetriever
        # Se da más peso a BM25 (0.6) para mejorar coincidencias exactas (nombres, fechas)
        retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.4, 0.6] # 40% Embeddings, 60% BM25
        )
        print(f"   -> {Colors.OKGREEN}Recuperador Híbrido (Ensemble) configurado. Pesos: Embeddings={retriever.weights[0]}, BM25={retriever.weights[1]}.{Colors.RESET}")
    else:
        # Usar solo el Vector Store Retriever si no hay chunks disponibles para BM25
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        print(f"\n{Colors.HEADER}--- 5. Configuración de la Cadena RAG (Solo Embeddings) ---{Colors.RESET}")
        print(f"   -> {Colors.WARNING}Usando solo el recuperador de Embeddings. BM25 requiere los chunks para inicializarse.{Colors.RESET}")
    # ----------------------------------------

    prompt = ChatPromptTemplate.from_template("""
    Responde **solo con la información exacta** encontrada en el contexto proporcionado.
    Si la respuesta no se encuentra en el contexto, escribe únicamente: "No tengo información sobre esto."

    Contexto: {context}

    Pregunta: {input}

    Respuesta:
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

def restructureQuestion(question: str) -> str:
    """
    Función para reestructurar preguntas complejas en preguntas simples para un sistema RAG.
    """
    llm = OllamaLLM(model=OLLAMA_MODEL)

    prompt = ChatPromptTemplate.from_template("""
        Eres un asistente especializado en procesamiento de lenguaje natural.
        Tu tarea es tomar preguntas complejas o redundantes de un usuario
        y reescribirlas como consultas simples, concisas y directas,
        manteniendo toda la información relevante. Solo envia la consulta reescrita.

        Ejemplo:
        Pregunta: ¿Quién fue el que descubrió primero la penicilina y en qué lugar ocurrió?
        Consulta reescrita: Nombre del descubridor y lugar del descubrimiento de la penicilina.

        Pregunta: {question}
        Consulta reescrita:
    """)

    prompt_text = prompt.format(question=question)

    response = llm.invoke(prompt_text)
    print(f"   -> {Colors.OKBLUE}Consulta reescrita por LLM: {response.strip()}{Colors.RESET}")

    return response.strip()
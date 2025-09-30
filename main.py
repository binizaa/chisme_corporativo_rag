# main.py

import sys
# ⬇️ Importaciones desde los nuevos módulos
from config import Colors, OLLAMA_MODEL 
from data_loader import load_documents_with_fallback, split_documents 
from rag_setup import setup_rag_chain

def main():
    # ⬇️ CAMBIO AQUÍ: Usamos la ruta local en lugar de la URL
    DOCUMENT_PATH = "data/chisme_corporativo.txt" 
    
    # La función load_documents_with_fallback está diseñada para manejar esto.
    documents = load_documents_with_fallback(DOCUMENT_PATH) 
    chunks = split_documents(documents)
    # ... (el resto del código permanece igual)
    # ...
    try:
        retrieval_chain = setup_rag_chain(chunks)
    except ValueError as e:
        print(f"{Colors.FAIL}\nTerminando la ejecución debido a: {e}{Colors.RESET}")
        sys.exit(1)

    # 3. Interacción con el usuario
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{'='*50}{Colors.RESET}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}       ¡El sistema RAG está listo! 🤖      {Colors.RESET}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{'='*50}{Colors.RESET}")
    print(f"Pregunta sobre el 'chisme corporativo'. Escribe 'salir' para terminar.")

    while True:
        user_input = input(f"\n{Colors.WARNING}Tu pregunta: {Colors.RESET}{Colors.BOLD}")
        print(Colors.RESET, end="") 
        
        if user_input.lower() in ["salir", "exit"]:
            print(f"{Colors.OKBLUE}Saliendo del sistema RAG. ¡Adiós! 👋{Colors.RESET}")
            break

        try:
            # Invoca la cadena RAG
            response = retrieval_chain.invoke({"input": user_input})
            
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}▶️ Respuesta del RAG ({OLLAMA_MODEL}):{Colors.RESET}")
            print(response["answer"])
            print(f"{Colors.OKGREEN}{'-'*50}{Colors.RESET}")
            
        except Exception as e:
            print(f"\n{Colors.FAIL}🚫 ERROR durante la invocación del RAG: {e}{Colors.RESET}")
            print(f"{Colors.FAIL}Asegúrate de que el modelo Ollama ({OLLAMA_MODEL}) esté corriendo en otra terminal.{Colors.RESET}")
            print(f"{Colors.FAIL}{'-'*50}{Colors.RESET}")

if __name__ == "__main__":
    main()
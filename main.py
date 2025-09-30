# main.py

import sys
from config import Colors, OLLAMA_MODEL 
from data_loader import load_documents_with_fallback, split_documents 
from rag_setup import setup_rag_chain, restructureQuestion

def main():
    DOCUMENT_PATH = "data/chisme_corporativo.txt" 
    
    documents = load_documents_with_fallback(DOCUMENT_PATH) 
    chunks = split_documents(documents)

    try:
        retrieval_chain = setup_rag_chain(chunks)
    except ValueError as e:
        print(f"{Colors.FAIL}\nTerminando la ejecución debido a: {e}{Colors.RESET}")
        sys.exit(1)

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
            user_input = restructureQuestion(user_input)
            response = retrieval_chain.invoke({"input": user_input})
            
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}▶️ Respuesta del RAG ({OLLAMA_MODEL}):{Colors.RESET}")
            print(response["answer"])
            print(f"{Colors.OKGREEN}{'-'*50}{Colors.RESET}")
            
        except Exception as e:
            print(f"\n{Colors.FAIL}🚫 ERROR durante la invocación del RAG: {e}{Colors.RESET}")
            print(f"{Colors.FAIL}{'-'*50}{Colors.RESET}")

if __name__ == "__main__":
    main()
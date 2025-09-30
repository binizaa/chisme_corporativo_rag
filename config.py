# config.py

# --- Constantes del Sistema ---
OLLAMA_MODEL = "llama3" # Modelo a usar en Ollama
DOCUMENT_URL = "https://roborregos.org/equipo" # URL de los documentos
CHROMA_DB_DIR = "./chroma_db" # Directorio para persistir ChromaDB

# --- Clase para colores de la consola ---
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
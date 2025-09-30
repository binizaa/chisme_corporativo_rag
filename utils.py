# utils.py

# --- Definición de Colores ANSI ---
class Colors:
    """Clase simple para códigos de color ANSI."""
    RESET = '\033[0m'
    HEADER = '\033[95m'  # Morado
    OKBLUE = '\033[94m'  # Azul
    OKGREEN = '\033[92m' # Verde
    WARNING = '\033[93m' # Amarillo
    FAIL = '\033[91m'    # Rojo
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --- Variables de Configuración ---
OLLAMA_MODEL = "llama3" 

# URL para cargar documentos
DOCUMENT_URL = "data/chisme_corporativo.txt" 
# DOCUMENT_URL = "https://www.tecmty.mx/noticias/nacional/cienciaytecnologia/roborregos-ganan-mundial-de-robotica-en-holanda"


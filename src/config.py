"""
Configuración centralizada para el proyecto SIMIEC.
Todas las constantes, keywords, stopwords y parámetros por defecto.
"""
import os

# Directorios
RESULTS_DIR = 'results'
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
DATA_DIR = 'data'

# Archivos por defecto
DEFAULT_INPUT_FILE = os.path.join(DATA_DIR, 'result.json')
DEFAULT_GEOREFERENCED_FILE = os.path.join(RESULTS_DIR, 'datos_georeferenciados.csv')
DEFAULT_RELATIONS_FILE = os.path.join(RESULTS_DIR, 'relaciones_lugares.csv')
DEFAULT_CANDIDATES_FILE = os.path.join(RESULTS_DIR, 'candidatos_lugares.csv')
DEFAULT_RESCUED_FILE = os.path.join(RESULTS_DIR, 'datos_rescatados.csv')
DEFAULT_BLOCK_RELATIONS_FILE = os.path.join(RESULTS_DIR, 'relaciones_bloque_municipio.csv')

# Keywords para análisis de sentimiento
KEYWORDS_INICIO = ['se fue', 'quito', 'apagon', 'ñooo', 'pinga', 'coño', 'otra vez']
KEYWORDS_FIN = ['llego', 'vino', 'pusieron', 'gracias', 'al fin', 'llegó']

# Keywords para filtrado de mensajes relevantes
KEYWORD_LUZ = ['luz', 'corriente', 'apagon', 'apagón', 'fui', 'vino', 'quito', 'puso']

# Stopwords básicas (para análisis de sentimiento)
STOPWORDS_BASICAS = set([
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'se', 'por', 'con', 
    'no', 'si', 'del'
])

# Stopwords extensas (para análisis de palabras más frecuentes)
STOPWORDS_EXTENSAS = set([
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'se', 'por', 'con', 'no', 'si', 'del',
    'las', 'un', 'una', 'es', 'al', 'le', 'da', 'su', 'sus', 'lo', 'le', 'les', 'me', 'te',
    'nos', 'os', 'mi', 'tu', 'su', 'nuestro', 'vuestro', 'este', 'ese', 'aquel', 'este',
    'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas', 'aquel', 'aquella', 'aquellos',
    'aquellas', 'ser', 'estar', 'haber', 'tener', 'hacer', 'poder', 'decir', 'ir', 'ver',
    'dar', 'saber', 'querer', 'llegar', 'pasar', 'deber', 'poner', 'parecer', 'quedar',
    'hablar', 'llevar', 'dejar', 'seguir', 'encontrar', 'llamar', 'venir', 'pensar',
    'salir', 'volver', 'tomar', 'conocer', 'vivir', 'sentir', 'tratar', 'mirar', 'contar',
    'empezar', 'esperar', 'buscar', 'existir', 'entrar', 'trabajar', 'escribir', 'perder',
    'producir', 'ocurrir', 'entender', 'pedir', 'recibir', 'recordar', 'terminar', 'permitir',
    'aparecer', 'conseguir', 'comenzar', 'servir', 'sacar', 'necesitar', 'mantener', 'resultar',
    'leer', 'caer', 'cambiar', 'presentar', 'crear', 'abrir', 'considerar', 'oír', 'acabar',
    'convertir', 'ganar', 'formar', 'traer', 'partir', 'morir', 'aceptar', 'realizar',
    'suponer', 'comprender', 'lograr', 'explicar', 'preguntar', 'tocar', 'reconocer', 'estudiar',
    'alcanzar', 'nacer', 'dirigir', 'correr', 'utilizar', 'pagar', 'ayudar', 'gustar', 'jugar',
    'escuchar', 'cumplir', 'ofrecer', 'descubrir', 'levantar', 'intentar', 'usar', 'decidir',
    'repetir', 'dormir', 'cerrar', 'quedar', 'limpiar', 'empezar', 'cocinar', 'comprar',
    'vender', 'regresar', 'volver', 'salir', 'entrar', 'subir', 'bajar', 'caminar', 'correr',
    'saltar', 'nadar', 'volar', 'conducir', 'manejar', 'parar', 'continuar', 'seguir',
    'empezar', 'terminar', 'acabar', 'comenzar', 'iniciar', 'finalizar', 'concluir',
    'bloque', 'bloques', 'corriente', 'luz', 'apagon', 'apagón', 'fui', 'vino', 'quito', 'puso'
])

# Stopwords para contexto geográfico
STOPWORDS_CONTEXTO = {
    'el', 'la', 'los', 'las', 'un', 'una', 'mi', 'tu', 'su',
    'casa', 'momento', 'seguida', 'breve', 'noche', 'dia', 'tarde',
    'mañana', 'oscuro', 'oscuridad', 'general', 'fase', 'horario',
    'fuego', 'candela', 'total', 'parte', 'zona', 'lugar', 'calle', 'reparto'
}

# Parámetros por defecto para análisis
DEFAULT_WEIGHT_THRESHOLD = 2  # Umbral mínimo de peso para relaciones en grafos
DEFAULT_TOP_LOCATIONS = 15    # Número de lugares top a mostrar
DEFAULT_TOP_LOCATIONS_MATRIX = 20  # Número de lugares para matriz de transición
DEFAULT_TOP_LOCATIONS_MDS = 30     # Número de lugares para MDS
DEFAULT_TOP_WORDS = 50        # Número de palabras top a mostrar
DEFAULT_TIME_BLOCK_MINUTES = 10   # Tamaño de bloque temporal en minutos

# Definición de períodos del día
NIGHT_START_HOUR = 19  # Hora de inicio de "noche" (19:00)
NIGHT_END_HOUR = 6     # Hora de fin de "noche" (06:00)

# Rango válido de bloques eléctricos
MIN_BLOCK_NUMBER = 1
MAX_BLOCK_NUMBER = 6


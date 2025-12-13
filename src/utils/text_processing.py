"""
Funciones de procesamiento de texto compartidas.
"""
import re
import unicodedata


def extract_block_from_text(text):
    """
    Extrae el número de bloque mencionado en un texto.
    
    Busca patrones como "bloque 1", "blq 2", "b 3 bloque", etc.
    Solo acepta números de bloque entre 1 y 6.
    
    Args:
        text: Texto a analizar (str o None)
        
    Returns:
        Número de bloque (int) si se encuentra, None en caso contrario
    """
    if not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    patterns = [
        r'bloque\s*(\d+)',
        r'blq\s*(\d+)',
        r'b\s*(\d+)\s*bloque',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            block_num = int(match.group(1))
            if 1 <= block_num <= 6:
                return block_num
    
    return None


def normalize_text(texto):
    """
    Normaliza texto eliminando acentos y convirtiendo a minúsculas.
    
    Args:
        texto: Texto a normalizar (str o None)
        
    Returns:
        Texto normalizado (str), cadena vacía si el input no es válido
    """
    if not isinstance(texto, str):
        return ""
    
    texto = texto.lower()
    return ''.join(c for c in unicodedata.normalize('NFD', texto) 
                   if unicodedata.category(c) != 'Mn')


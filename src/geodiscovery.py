import pandas as pd
import re
from collections import Counter
from itertools import combinations

INPUT_FILE = 'datos_rescatados.csv'

STOPWORDS_CONTEXTO = {
    'el', 'la', 'los', 'las', 'un', 'una', 'mi', 'tu', 'su',
    'casa', 'momento', 'seguida', 'breve', 'noche', 'dia', 'tarde',
    'mañana', 'oscuro', 'oscuridad', 'general', 'fase', 'horario',
    'fuego', 'candela', 'total', 'parte', 'zona', 'lugar', 'calle', 'reparto'
}

def limpiar_token(token):
    return token.strip(' .,;!?()').lower()

def extraer_candidatos(texto):
    """
    Usa expresiones regulares para capturar palabras que aparecen 
    en contextos geográficos.
    Patrón: (preposición) + (Palabra Capitalizada o frase corta)
    """
    if not isinstance(texto, str):
        return []
    
    candidates = []
    
    matches = re.findall(r'\b(?:en|de|desde|hacia|zona|reparto|municipio)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)', texto)
    
    for match in matches:
        token = limpiar_token(match)
        if token not in STOPWORDS_CONTEXTO and len(token) > 2:
            candidates.append(token)
            
    return candidates

def descubrir_lugares_y_relaciones():
    print(f"Iniciando descubrimiento de lugares en: {INPUT_FILE}")
    
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Archivo no encontrado.")
        return

    all_candidates = []
    co_occurrences = []

    print("Analizando texto...")
    
    for text in df['text']:
        lugares_en_mensaje = extraer_candidatos(text)
        
        all_candidates.extend(lugares_en_mensaje)
        
        if len(set(lugares_en_mensaje)) > 1:
            unique_places = sorted(list(set(lugares_en_mensaje)))
            pairs = list(combinations(unique_places, 2))
            co_occurrences.extend(pairs)

    conteo_lugares = Counter(all_candidates)
    top_50 = conteo_lugares.most_common(50)
    
    print("\nTOP 20 LUGARES DESCUBIERTOS (Candidatos):")
    print("-" * 40)
    for lugar, freq in top_50[:20]:
        print(f"{lugar.title()}: {freq} menciones")
        
    conteo_relaciones = Counter(co_occurrences)
    top_relaciones = conteo_relaciones.most_common(15)
    
    print("\nTOP 10 RELACIONES (Barrio -> Municipio):")
    print("-" * 40)
    for (lugarA, lugarB), freq in top_relaciones[:10]:
        print(f"{lugarA.title()} <--> {lugarB.title()} ({freq} veces juntos)")

    df_candidatos = pd.DataFrame(top_50, columns=['Lugar_Sugerido', 'Frecuencia'])
    df_candidatos.to_csv('candidatos_lugares.csv', index=False)
    print("\nLista de candidatos guardada en 'candidatos_lugares.csv'.")
    print("Acción recomendada: Abre ese CSV, borra la basura, y usa esa lista para el paso final.")

if __name__ == "__main__":
    descubrir_lugares_y_relaciones()
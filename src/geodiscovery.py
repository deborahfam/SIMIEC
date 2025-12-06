import pandas as pd
import re
from collections import Counter
from itertools import combinations
import os

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

STOPWORDS_CONTEXTO = {
    'el', 'la', 'los', 'las', 'un', 'una', 'mi', 'tu', 'su',
    'casa', 'momento', 'seguida', 'breve', 'noche', 'dia', 'tarde',
    'mañana', 'oscuro', 'oscuridad', 'general', 'fase', 'horario',
    'fuego', 'candela', 'total', 'parte', 'zona', 'lugar', 'calle', 'reparto'
}

def clean_token(token):
    return token.strip(' .,;!?()').lower()

def extract_candidates(texto):
    if not isinstance(texto, str):
        return []
    
    candidates = []
    matches = re.findall(r'\b(?:en|de|desde|hacia|zona|reparto|municipio)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)', texto)
    
    for match in matches:
        token = clean_token(match)
        if token not in STOPWORDS_CONTEXTO and len(token) > 2 and 'bloque' not in token:
            candidates.append(token)
            
    return candidates

def extract_block_from_text(text):
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

def discover_locations_and_relations(input_file=None, output_file=None, output_block_relations_file=None):
    if input_file is None:
        input_file = os.path.join(RESULTS_DIR, 'datos_rescatados.csv')
    if output_file is None:
        output_file = os.path.join(RESULTS_DIR, 'candidatos_lugares.csv')
    if output_block_relations_file is None:
        output_block_relations_file = os.path.join(RESULTS_DIR, 'relaciones_bloque_municipio.csv')
    
    print(f"Starting location discovery in: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("File not found.")
        return None

    all_candidates = []
    co_occurrences = []
    block_location_relations = []

    print("Analyzing text...")
    
    for idx, row in df.iterrows():
        text = row['text']
        lugares_en_mensaje = extract_candidates(text)
        block_num = extract_block_from_text(text)
        
        all_candidates.extend(lugares_en_mensaje)
        
        if len(set(lugares_en_mensaje)) > 1:
            unique_places = sorted(list(set(lugares_en_mensaje)))
            pairs = list(combinations(unique_places, 2))
            co_occurrences.extend(pairs)
        
        if block_num is not None and len(lugares_en_mensaje) > 0:
            for lugar in lugares_en_mensaje:
                block_location_relations.append((f'Bloque {block_num}', lugar))

    location_count = Counter(all_candidates)
    top_50 = location_count.most_common(50)
    
    print("\nTOP 20 DISCOVERED LOCATIONS (Candidates):")
    print("-" * 40)
    for lugar, freq in top_50[:20]:
        print(f"{lugar.title()}: {freq} mentions")
        
    relation_count = Counter(co_occurrences)
    top_relaciones = relation_count.most_common(15)
    
    print("\nTOP 10 RELATIONS (Barrio -> Municipio):")
    print("-" * 40)
    for (lugarA, lugarB), freq in top_relaciones[:10]:
        print(f"{lugarA.title()} <--> {lugarB.title()} ({freq} times together)")

    df_candidatos = pd.DataFrame(top_50, columns=['Lugar_Sugerido', 'Frecuencia'])
    df_candidatos = df_candidatos[~df_candidatos['Lugar_Sugerido'].str.lower().str.contains('bloque', na=False)]
    df_candidatos.to_csv(output_file, index=False)
    print(f"\nCandidate list saved in '{output_file}' (blocks excluded).")
    
    if block_location_relations:
        block_rel_count = Counter(block_location_relations)
        df_block_rel = pd.DataFrame(block_rel_count.most_common(), columns=['Relation', 'Weight'])
        df_block_rel[['Bloque', 'Municipio']] = pd.DataFrame(df_block_rel['Relation'].tolist(), index=df_block_rel.index)
        df_block_rel = df_block_rel[['Bloque', 'Municipio', 'Weight']].sort_values('Weight', ascending=False)
        df_block_rel.to_csv(output_block_relations_file, index=False)
        print(f"Block-Location relations saved in '{output_block_relations_file}'.")
        print(f"   -> {len(df_block_rel)} block-location relationships found.")
    
    return df_candidatos

if __name__ == "__main__":
    discover_locations_and_relations()
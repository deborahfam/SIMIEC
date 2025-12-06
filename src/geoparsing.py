import pandas as pd
import unicodedata
import re
from itertools import combinations
from collections import Counter
import os

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def normalize_text(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

def load_knowledge_base(input_file=None):
    if input_file is None:
        input_file = os.path.join(RESULTS_DIR, 'candidatos_lugares.csv')
    
    print(f"Loading knowledge base from: {input_file}")
    try:
        df_places = pd.read_csv(input_file)
        lugares = df_places['Lugar_Sugerido'].dropna().astype(str).tolist()
        lugares_norm = [(l, normalize_text(l)) for l in lugares]
        lugares_norm.sort(key=lambda x: len(x[1]), reverse=True)
        print(f"   -> {len(lugares_norm)} locations loaded for search.")
        return lugares_norm
    except FileNotFoundError:
        print("❌ Error: 'candidatos_lugares.csv' does not exist. Run discovery step first.")
        return []

def detect_all_locations(texto_norm, location_list):
    encontrados = []
    for nombre_real, nombre_norm in location_list:
        if nombre_norm in texto_norm and 'bloque' not in nombre_norm:
            encontrados.append(nombre_real)
    return list(set(encontrados))

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

def process_geoparsing(input_msg_file=None, input_places_file=None, 
                      output_data_file=None, output_relations_file=None,
                      output_block_relations_file=None):
    if input_msg_file is None:
        input_msg_file = os.path.join(RESULTS_DIR, 'datos_rescatados.csv')
    if input_places_file is None:
        input_places_file = os.path.join(RESULTS_DIR, 'candidatos_lugares.csv')
    if output_data_file is None:
        output_data_file = os.path.join(RESULTS_DIR, 'datos_georeferenciados.csv')
    if output_relations_file is None:
        output_relations_file = os.path.join(RESULTS_DIR, 'relaciones_lugares.csv')
    if output_block_relations_file is None:
        output_block_relations_file = os.path.join(RESULTS_DIR, 'relaciones_bloque_municipio.csv')
    
    lugares_referencia = load_knowledge_base(input_places_file)
    if not lugares_referencia:
        return None, None, None

    try:
        df = pd.read_csv(input_msg_file)
    except FileNotFoundError:
        print("❌ Messages not found.")
        return None, None, None

    print("Starting Dynamic Geoparsing...")
    
    df['text_norm'] = df['text'].apply(normalize_text)
    df['lugares_detectados'] = df['text_norm'].apply(
        lambda x: detect_all_locations(x, lugares_referencia)
    )
    df['bloque'] = df['text'].apply(extract_block_from_text)

    df_geo = df[df['lugares_detectados'].map(len) > 0].copy()
    
    print("🕸️ Generating location relationship network (Co-occurrence)...")
    relaciones = []
    for lugares in df_geo['lugares_detectados']:
        if len(lugares) > 1:
            pairs = list(combinations(sorted(lugares), 2))
            relaciones.extend(pairs)
    
    df_rel = None
    if relaciones:
        df_rel = pd.DataFrame(relaciones, columns=['Source', 'Target'])
        df_rel = df_rel.groupby(['Source', 'Target']).size().reset_index(name='Weight')
        df_rel.sort_values(by='Weight', ascending=False, inplace=True)
        df_rel.to_csv(output_relations_file, index=False)
        print(f"   -> Location relationships saved in '{output_relations_file}'.")

    print("📦 Generating block-location relationships...")
    block_location_relations = []
    for idx, row in df_geo.iterrows():
        block_num = row['bloque']
        lugares = row['lugares_detectados']
        
        if block_num is not None and len(lugares) > 0:
            for lugar in lugares:
                block_location_relations.append((f'Bloque {block_num}', lugar))
    
    df_block_rel = None
    if block_location_relations:
        block_rel_count = Counter(block_location_relations)
        df_block_rel = pd.DataFrame([
            {'Bloque': rel[0], 'Municipio': rel[1], 'Weight': count}
            for rel, count in block_rel_count.items()
        ])
        df_block_rel = df_block_rel.sort_values('Weight', ascending=False)
        df_block_rel.to_csv(output_block_relations_file, index=False)
        print(f"   -> Block-location relationships saved in '{output_block_relations_file}'.")
        print(f"   -> {len(df_block_rel)} block-location relationships found.")

    df_geo['lugar_principal'] = df_geo['lugares_detectados'].apply(lambda x: x[0])
    
    cols = ['date', 'text', 'lugar_principal', 'lugares_detectados']
    df_geo[cols].to_csv(output_data_file, index=False)
    
    print(f"\n✅ PROCESS COMPLETED.")
    print(f"   Processed messages: {len(df)}")
    print(f"   Georeferenced messages: {len(df_geo)}")
    print(f"   File saved: {output_data_file}")
    
    return df_geo, df_rel, df_block_rel

if __name__ == "__main__":
    process_geoparsing()
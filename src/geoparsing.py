import pandas as pd
import unicodedata
from itertools import combinations

INPUT_MSG_FILE = 'datos_rescatados.csv'
INPUT_PLACES_FILE = 'candidatos_lugares.csv'
OUTPUT_DATA_FILE = 'datos_georeferenciados.csv'
OUTPUT_RELATIONS_FILE = 'relaciones_lugares.csv'

def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

def cargar_base_conocimiento():
    print(f"Cargando base de conocimiento desde: {INPUT_PLACES_FILE}")
    try:
        df_places = pd.read_csv(INPUT_PLACES_FILE)
        lugares = df_places['Lugar_Sugerido'].dropna().astype(str).tolist()
        
        lugares_norm = [(l, normalizar_texto(l)) for l in lugares]
        
        lugares_norm.sort(key=lambda x: len(x[1]), reverse=True)
        
        print(f"   -> {len(lugares_norm)} lugares cargados para búsqueda.")
        return lugares_norm
    except FileNotFoundError:
        print("❌ Error: No existe 'candidatos_lugares.csv'. Ejecuta el paso de descubrimiento primero.")
        return []

def detectar_todos_los_lugares(texto_norm, lista_lugares):
    encontrados = []
    for nombre_real, nombre_norm in lista_lugares:
        if nombre_norm in texto_norm:
            encontrados.append(nombre_real)
            
    return list(set(encontrados))

def procesar():
    lugares_referencia = cargar_base_conocimiento()
    if not lugares_referencia: return

    try:
        df = pd.read_csv(INPUT_MSG_FILE)
    except FileNotFoundError:
        print("❌ No se encuentran los mensajes.")
        return

    print("Iniciando Geoparsing Dinámico...")
    
    df['text_norm'] = df['text'].apply(normalizar_texto)

    df['lugares_detectados'] = df['text_norm'].apply(
        lambda x: detectar_todos_los_lugares(x, lugares_referencia)
    )

    df_geo = df[df['lugares_detectados'].map(len) > 0].copy()
    
    print("🕸️  Generando red de relaciones (Co-ocurrencia)...")
    relaciones = []
    for lugares in df_geo['lugares_detectados']:
        if len(lugares) > 1:
            pairs = list(combinations(sorted(lugares), 2))
            relaciones.extend(pairs)
    
    if relaciones:
        df_rel = pd.DataFrame(relaciones, columns=['Source', 'Target'])
        df_rel = df_rel.groupby(['Source', 'Target']).size().reset_index(name='Weight')
        df_rel.sort_values(by='Weight', ascending=False, inplace=True)
        df_rel.to_csv(OUTPUT_RELATIONS_FILE, index=False)
        print(f"   -> Relaciones guardadas en '{OUTPUT_RELATIONS_FILE}' (Útil para Gephi o NetworkX).")

    df_geo['lugar_principal'] = df_geo['lugares_detectados'].apply(lambda x: x[0])
    
    cols = ['date', 'text', 'lugar_principal', 'lugares_detectados']
    df_geo[cols].to_csv(OUTPUT_DATA_FILE, index=False)
    
    print(f"\n✅ PROCESO TERMINADO.")
    print(f"   Mensajes procesados: {len(df)}")
    print(f"   Mensajes geolocalizados: {len(df_geo)}")
    print(f"   Archivo guardado: {OUTPUT_DATA_FILE}")

if __name__ == "__main__":
    procesar()
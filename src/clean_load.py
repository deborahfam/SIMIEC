import json
import pandas as pd
import os

FILE_PATH = 'data/result.json' 
KEYWORD_LUZ = ['luz', 'corriente', 'apagon', 'apagón', 'fui', 'vino', 'quito', 'puso']

def normalizar_texto(texto_raw):
    if isinstance(texto_raw, str):
        return texto_raw
    elif isinstance(texto_raw, list):
        texto_limpio = ""
        for item in texto_raw:
            if isinstance(item, str):
                texto_limpio += item
            elif isinstance(item, dict) and 'text' in item:
                texto_limpio += item['text']
        return texto_limpio
    return ""

def cargar_y_limpiar_datos(filepath):
    print(f"1. Buscando archivo en: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ ERROR: No se encuentra el archivo '{filepath}'.")
        print("   Verifica que el nombre sea correcto o usa la ruta absoluta.")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ ERROR al leer el JSON: {e}")
        return None

    messages = []
    print("2. Procesando estructura del JSON...")
    
    if 'messages' in data:
        raw_msgs = data['messages']
    else:
        raw_msgs = data

    contador_procesados = 0
    for msg in raw_msgs:
        if msg.get('type') == 'message':
            texto_raw = msg.get('text', '')
            texto_final = normalizar_texto(texto_raw)
            
            if len(texto_final) > 1:
                messages.append({
                    'date': msg['date'],
                    'from': msg.get('from', 'Anonimo'),
                    'from_id': msg.get('from_id', 'unknown'),
                    'text': texto_final
                })
                contador_procesados += 1

    print(f"   -> Mensajes extraídos: {contador_procesados}")
    
    if not messages:
        print("⚠️ ALERTA: No se encontraron mensajes de texto válidos.")
        return None

    df = pd.DataFrame(messages)
    df['date'] = pd.to_datetime(df['date'])
    return df

def filtro_heuristico(df):
    print("3. Aplicando filtro de relevancia (palabras clave)...")
    
    if df is None or df.empty:
        print("❌ Error: El DataFrame está vacío, no se puede filtrar.")
        return None

    df['es_reporte'] = df['text'].astype(str).str.lower().apply(
        lambda x: any(k in x for k in KEYWORD_LUZ)
    )
    
    df_relevantes = df[df['es_reporte'] == True].copy()
    
    print(f"   -> Total mensajes: {len(df)}")
    print(f"   -> Reportes potenciales (con palabras clave): {len(df_relevantes)}")
    return df_relevantes
    
if __name__ == "__main__":
    df_raw = cargar_y_limpiar_datos(FILE_PATH)
    
    if df_raw is not None:
        df_limpio = filtro_heuristico(df_raw)
        
        if df_limpio is not None and not df_limpio.empty:
            print("\n--- ✅ VISTA PREVIA (Últimos 5 mensajes) ---")
            print(df_limpio[['date', 'text']].tail(5))
            
            output_csv = 'datos_rescatados.csv'
            df_limpio.to_csv(output_csv, index=False)
            print(f"\nArchivo guardado: {output_csv}")
        else:
            print("\n⚠️ No se encontraron mensajes relevantes con las palabras clave.")
    else:
        print("\n❌ El proceso se detuvo por errores en la carga.")
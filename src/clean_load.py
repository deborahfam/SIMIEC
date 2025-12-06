import json
import pandas as pd
import os

FILE_PATH = 'data/result.json'
RESULTS_DIR = 'results'
KEYWORD_LUZ = ['luz', 'corriente', 'apagon', 'apagón', 'fui', 'vino', 'quito', 'puso']

os.makedirs(RESULTS_DIR, exist_ok=True)

def normalize_text(texto_raw):
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

def load_and_clean_data(filepath):
    print(f"1. Searching file at: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ ERROR: File '{filepath}' not found.")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ ERROR reading JSON: {e}")
        return None

    messages = []
    print("2. Processing JSON structure...")
    
    if 'messages' in data:
        raw_msgs = data['messages']
    else:
        raw_msgs = data

    processed_count = 0
    skipped_assistant = 0
    for msg in raw_msgs:
        if msg.get('type') == 'message':
            from_name = msg.get('from', '') or ''
            if from_name and ('Asistente en Línea' in from_name or 'Asistente en Línea 2' in from_name):
                skipped_assistant += 1
                continue
            
            texto_raw = msg.get('text', '')
            texto_final = normalize_text(texto_raw)
            
            if len(texto_final) > 1:
                messages.append({
                    'date': msg['date'],
                    'from': from_name,
                    'from_id': msg.get('from_id', 'unknown'),
                    'text': texto_final
                })
                processed_count += 1
    
    if skipped_assistant > 0:
        print(f"   -> Skipped {skipped_assistant} messages from 'Asistente en Línea'")

    print(f"   -> Extracted messages: {processed_count}")
    
    if not messages:
        print("⚠️ WARNING: No valid text messages found.")
        return None

    df = pd.DataFrame(messages)
    df['date'] = pd.to_datetime(df['date'])
    return df

def heuristic_filter(df):
    print("3. Applying relevance filter (keywords)...")
    
    if df is None or df.empty:
        print("❌ Error: DataFrame is empty, cannot filter.")
        return None

    df['es_reporte'] = df['text'].astype(str).str.lower().apply(
        lambda x: any(k in x for k in KEYWORD_LUZ)
    )
    
    df_relevantes = df[df['es_reporte'] == True].copy()
    
    print(f"   -> Total messages: {len(df)}")
    print(f"   -> Potential reports (with keywords): {len(df_relevantes)}")
    return df_relevantes

def process_data(input_file=None, output_file=None):
    if input_file is None:
        input_file = FILE_PATH
    if output_file is None:
        output_file = os.path.join(RESULTS_DIR, 'datos_rescatados.csv')
    
    df_raw = load_and_clean_data(input_file)
    
    if df_raw is not None:
        df_limpio = heuristic_filter(df_raw)
        
        if df_limpio is not None and not df_limpio.empty:
            print("\n--- ✅ PREVIEW (Last 5 messages) ---")
            print(df_limpio[['date', 'text']].tail(5))
            
            df_limpio.to_csv(output_file, index=False)
            print(f"\nFile saved: {output_file}")
            return df_limpio
        else:
            print("\n⚠️ No relevant messages found with keywords.")
            return None
    else:
        print("\n❌ Process stopped due to loading errors.")
        return None
    
if __name__ == "__main__":
    process_data()
import pandas as pd
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from datetime import datetime, timedelta, timezone

API_ID = 'TU_API_ID_AQUI'
API_HASH = 'TU_API_HASH_AQUI'
GROUP_USERNAME = 'unionelectricacuba' 
DAYS_BACK = 30

OUTPUT_FILE = 'dataset_electrico_raw.csv'

async def main():
    print("--- INICIANDO SCRAPER INTELIGENTE ---")
    
    async with TelegramClient('mi_sesion', API_ID, API_HASH) as client:
        print(f"Conectando con Telegram y buscando el grupo: {GROUP_USERNAME}...")
        try:
            entity = await client.get_entity(GROUP_USERNAME)
        except ValueError:
            print("Error: No se encontró el grupo. Verifica el nombre de usuario.")
            return

        print(f"Grupo encontrado: {entity.title}")
        
        limit_date = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
        print(f"Descargando mensajes posteriores a: {limit_date.strftime('%Y-%m-%d %H:%M:%S')}")

        messages_data = []
        total_count = 0

        async for message in client.iter_messages(entity):
            
            # Si el mensaje es más viejo que la fecha límite, PARAR.
            if message.date < limit_date:
                print("\nSe alcanzó la fecha límite. Deteniendo descarga.")
                break
            
            if message.text:
                total_count += 1
                
                messages_data.append({
                    'date': message.date,         # Fecha exacta (UTC)
                    'sender_id': message.sender_id, # ID del usuario (para contar reportes por persona)
                    'text': message.text,         # El contenido
                    'views': message.views        # Vistas (opcional, sirve para medir impacto)
                })
                
                if total_count % 100 == 0:
                    print(f"Descargados: {total_count}...", end='\r')

        print(f"\nProcesamiento finalizado. Total recolectado: {len(messages_data)}")
        
        if messages_data:
            df = pd.DataFrame(messages_data)
            df = df.sort_values(by='date')
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
            print(f"Archivo guardado exitosamente: {OUTPUT_FILE}")
        else:
            print("No se encontraron mensajes en el rango de fechas.")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
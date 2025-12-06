import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.dates as mdates
from datetime import datetime

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300      # Alta resolución para impresión  
plt.rcParams['savefig.bbox'] = 'tight' # Cortar bordes blancos sobrantes
plt.rcParams['font.family'] = 'sans-serif' # Fuente limpia

# --- 1. CARGA DE DATOS ---
print("📥 Cargando datos...")
try:
    df = pd.read_csv('results/datos_georeferenciados.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Cargar relaciones para el grafo
    df_rel = pd.read_csv('results/relaciones_lugares.csv')
except FileNotFoundError:
    print("❌ Error: Faltan los archivos CSV (datos_georeferenciados.csv o relaciones_lugares.csv)")
    exit()

print(f"   -> {len(df)} reportes cargados (total histórico).")

# --- FILTRAR POR ÚLTIMO MES (5 de noviembre a 5 de diciembre) ---
fecha_inicio = pd.Timestamp('2024-11-05')
fecha_fin = pd.Timestamp('2024-12-05')
df = df[(df['date'] >= fecha_inicio) & (df['date'] <= fecha_fin)].copy()

print(f"   -> {len(df)} reportes en el período seleccionado (5 nov - 5 dic 2024).")

# Filtrar relaciones para solo incluir lugares que aparecen en el período filtrado
lugares_periodo = set(df['lugar_principal'].unique())
df_rel = df_rel[
    (df_rel['Source'].isin(lugares_periodo)) & 
    (df_rel['Target'].isin(lugares_periodo))
].copy()

print(f"   -> {len(df_rel)} relaciones en el período seleccionado.")

# --- 2. GRÁFICA TEMPORAL (TIMELINE) ---
print("📈 Generando Fig 1: Línea de Tiempo...")
plt.figure(figsize=(12, 6))

# Agrupar por hora
timeline = df.set_index('date').resample('h')['text'].count()

# Plot
ax = timeline.plot(kind='line', color='#d62728', linewidth=1.5)
plt.title('Frecuencia de Reportes de Incidencias Eléctricas (Por Hora)\nPeríodo: 5 Nov - 5 Dic 2024', fontsize=14, fontweight='bold')
plt.ylabel('Cantidad de Reportes')
plt.xlabel('Fecha y Hora')

# Formato de fecha en eje X
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:00'))
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('fig1_evolucion_temporal.png')
plt.show()

# --- 3. GRÁFICA DE CALOR (HEATMAP SEMANAL) ---
print("🔥 Generando Fig 2: Mapa de Calor Semanal...")
plt.figure(figsize=(10, 6))

# Preparar datos
df['hour'] = df['date'].dt.hour
df['day_name'] = df['date'].dt.day_name()
# Traducir días para el paper en español
dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
df['day_name'] = pd.Categorical(df['day_name'], categories=dias_orden, ordered=True)

# Crear matriz pivote
heatmap_data = df.groupby(['day_name', 'hour']).size().unstack(fill_value=0)
heatmap_data.index = dias_es # Renombrar índice a español

# Plot
sns.heatmap(heatmap_data, cmap='YlOrRd', linewidths=.5, cbar_kws={'label': 'Nº Reportes'})
plt.title('Concentración de Reportes: Día de la Semana vs Hora\nPeríodo: 5 Nov - 5 Dic 2024', fontsize=14, fontweight='bold')
plt.xlabel('Hora del Día')
plt.ylabel('Día de la Semana')

plt.savefig('fig2_heatmap_semanal.png')
plt.show()

# --- 4. GRÁFICA ESPACIAL (BAR CHART) ---
print("📊 Generando Fig 3: Top Lugares Afectados...")
plt.figure(figsize=(10, 8))

# Top 15 lugares
top_places = df['lugar_principal'].value_counts().head(15)

# Plot
sns.barplot(x=top_places.values, y=top_places.index, palette='viridis', hue=top_places.index, legend=False)
plt.title('Top 15 Zonas con Mayor Frecuencia de Reportes\nPeríodo: 5 Nov - 5 Dic 2024', fontsize=14, fontweight='bold')
plt.xlabel('Cantidad de Menciones')
plt.ylabel('Zona / Municipio Identificado')

for i, v in enumerate(top_places.values):
    plt.text(v + 0.5, i, str(v), color='black', va='center')

plt.savefig('fig3_top_lugares.png')
plt.show()

# --- 5. TOPOLOGÍA DE RED (GRAFO) ---
print("🕸️ Generando Fig 4: Grafo de Conexiones...")
plt.figure(figsize=(12, 12))

# Crear grafo desde DataFrame
G = nx.from_pandas_edgelist(df_rel, 'Source', 'Target', edge_attr='Weight')

# Filtrar: Eliminar nodos/conexiones muy débiles para limpiar la imagen
# (Solo mostramos conexiones que aparecen al menos X veces)
umbral_peso = 2 
edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['Weight'] >= umbral_peso]
G_filtered = G.edge_subgraph(edges_to_keep)

# Algoritmo de distribución (Layout)
pos = nx.spring_layout(G_filtered, k=0.5, iterations=50, seed=42)

# Tamaños basados en grado (importancia)
node_sizes = [v * 100 for v in dict(G_filtered.degree()).values()]
# Grosores de línea basados en peso
edge_widths = [d['Weight'] * 0.5 for u, v, d in G_filtered.edges(data=True)]

# Dibujar
nx.draw_networkx_nodes(G_filtered, pos, node_size=node_sizes, node_color='#3498db', alpha=0.8)
nx.draw_networkx_edges(G_filtered, pos, width=edge_widths, alpha=0.4, edge_color='gray')
nx.draw_networkx_labels(G_filtered, pos, font_size=8, font_family='sans-serif', font_weight='bold')

plt.title(f'Grafo de Co-ocurrencia de Cortes (Topología Inferida)\nPeríodo: 5 Nov - 5 Dic 2024 | Filtro: Conexiones con >= {umbral_peso} reportes conjuntos', fontsize=14)
plt.axis('off') # Ocultar ejes

plt.savefig('fig4_topologia_red.png')
plt.show()

print("\n✅ ¡Listo! Se han generado 4 imágenes PNG en tu carpeta.")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.dates as mdates
from datetime import datetime

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'sans-serif'

try:
    df = pd.read_csv('results/datos_georeferenciados.csv')
    df['date'] = pd.to_datetime(df['date'])
    df_rel = pd.read_csv('results/relaciones_lugares.csv')
except FileNotFoundError:
    print("❌ Error: Faltan los archivos CSV (datos_georeferenciados.csv o relaciones_lugares.csv)")
    exit()

fecha_inicio = pd.Timestamp('2024-11-05')
fecha_fin = pd.Timestamp('2024-12-05')
df = df[(df['date'] >= fecha_inicio) & (df['date'] <= fecha_fin)].copy()

lugares_periodo = set(df['lugar_principal'].unique())
df_rel = df_rel[
    (df_rel['Source'].isin(lugares_periodo)) & 
    (df_rel['Target'].isin(lugares_periodo))
].copy()

plt.figure(figsize=(12, 6))
timeline = df.set_index('date').resample('h')['text'].count()
ax = timeline.plot(kind='line', color='#d62728', linewidth=1.5)
plt.title('Frecuencia de Reportes de Incidencias Eléctricas (Por Hora)\nPeríodo: 5 Nov - 5 Dic 2024', fontsize=14, fontweight='bold')
plt.ylabel('Cantidad de Reportes')
plt.xlabel('Fecha y Hora')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:00'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('fig1_evolucion_temporal.png')
plt.show()

plt.figure(figsize=(10, 6))
df['hour'] = df['date'].dt.hour
df['day_name'] = df['date'].dt.day_name()
dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
df['day_name'] = pd.Categorical(df['day_name'], categories=dias_orden, ordered=True)
heatmap_data = df.groupby(['day_name', 'hour']).size().unstack(fill_value=0)
heatmap_data.index = dias_es
sns.heatmap(heatmap_data, cmap='YlOrRd', linewidths=.5, cbar_kws={'label': 'Nº Reportes'})
plt.title('Concentración de Reportes: Día de la Semana vs Hora\nPeríodo: 5 Nov - 5 Dic 2024', fontsize=14, fontweight='bold')
plt.xlabel('Hora del Día')
plt.ylabel('Día de la Semana')
plt.savefig('fig2_heatmap_semanal.png')
plt.show()

plt.figure(figsize=(10, 8))
top_places = df['lugar_principal'].value_counts().head(15)
sns.barplot(x=top_places.values, y=top_places.index, palette='viridis', hue=top_places.index, legend=False)
plt.title('Top 15 Zonas con Mayor Frecuencia de Reportes\nPeríodo: 5 Nov - 5 Dic 2024', fontsize=14, fontweight='bold')
plt.xlabel('Cantidad de Menciones')
plt.ylabel('Zona / Municipio Identificado')
for i, v in enumerate(top_places.values):
    plt.text(v + 0.5, i, str(v), color='black', va='center')
plt.savefig('fig3_top_lugares.png')
plt.show()

plt.figure(figsize=(12, 12))
G = nx.from_pandas_edgelist(df_rel, 'Source', 'Target', edge_attr='Weight')
umbral_peso = 2
edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['Weight'] >= umbral_peso]
G_filtered = G.edge_subgraph(edges_to_keep)
pos = nx.spring_layout(G_filtered, k=0.5, iterations=50, seed=42)
node_sizes = [v * 100 for v in dict(G_filtered.degree()).values()]
edge_widths = [d['Weight'] * 0.5 for u, v, d in G_filtered.edges(data=True)]
nx.draw_networkx_nodes(G_filtered, pos, node_size=node_sizes, node_color='#3498db', alpha=0.8)
nx.draw_networkx_edges(G_filtered, pos, width=edge_widths, alpha=0.4, edge_color='gray')
nx.draw_networkx_labels(G_filtered, pos, font_size=8, font_family='sans-serif', font_weight='bold')
plt.title(f'Grafo de Co-ocurrencia de Cortes (Topología Inferida)\nPeríodo: 5 Nov - 5 Dic 2024 | Filtro: Conexiones con >= {umbral_peso} reportes conjuntos', fontsize=14)
plt.axis('off')
plt.savefig('fig4_topologia_red.png')
plt.show()
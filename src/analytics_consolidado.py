import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.dates as mdates
from sklearn.manifold import MDS
from collections import Counter
import re
import os

RESULTS_DIR = 'results'
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'sans-serif'

KEYWORDS_INICIO = ['se fue', 'quito', 'apagon', 'ñooo', 'pinga', 'coño', 'otra vez']
KEYWORDS_FIN = ['llego', 'vino', 'pusieron', 'gracias', 'al fin', 'llegó']

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

def load_and_filter_data(start_date=None, end_date=None):
    print("📥 Loading data...")
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, 'datos_georeferenciados.csv'))
        df['date'] = pd.to_datetime(df['date'])
        df_rel = pd.read_csv(os.path.join(RESULTS_DIR, 'relaciones_lugares.csv'))
    except FileNotFoundError:
        print("❌ Error: Missing CSV files in results/")
        return None, None
    
    print(f"   -> {len(df)} reports loaded (total historical).")
    date_min = df['date'].min()
    date_max = df['date'].max()
    print(f"   -> Date range in data: {date_min} to {date_max}")
    
    if start_date is not None or end_date is not None:
        original_count = len(df)
        if start_date is not None:
            start_date_ts = pd.Timestamp(start_date)
            df = df[df['date'] >= start_date_ts].copy()
            print(f"   -> After filtering from {start_date_ts.date()}: {len(df)} reports")
        if end_date is not None:
            end_date_ts = pd.Timestamp(end_date)
            df = df[df['date'] <= end_date_ts].copy()
            print(f"   -> After filtering until {end_date_ts.date()}: {len(df)} reports")
        
        if start_date is not None and end_date is not None:
            print(f"   -> Final: {len(df)} reports in period {pd.Timestamp(start_date).date()} - {pd.Timestamp(end_date).date()}.")
        elif start_date is not None:
            print(f"   -> Final: {len(df)} reports since {pd.Timestamp(start_date).date()}.")
        elif end_date is not None:
            print(f"   -> Final: {len(df)} reports until {pd.Timestamp(end_date).date()}.")
        
        if len(df) == 0:
            print(f"   ⚠️ WARNING: No data found in specified period!")
            print(f"   -> Available date range: {date_min} to {date_max}")
        
        lugares_periodo = set(df['lugar_principal'].unique())
        df_rel = df_rel[
            (df_rel['Source'].isin(lugares_periodo)) & 
            (df_rel['Target'].isin(lugares_periodo))
        ].copy()
        print(f"   -> {len(df_rel)} relationships in selected period.")
    
    return df, df_rel

def generate_temporal_evolution(df, start_date=None, end_date=None):
    print("📈 Generating Fig 1: Timeline...")
    plt.figure(figsize=(12, 6))
    
    timeline = df.set_index('date').resample('h')['text'].count()
    
    ax = timeline.plot(kind='line', color='#d62728', linewidth=1.5)
    title = 'Frecuencia de Reportes de Incidencias Eléctricas (Por Hora)'
    if start_date and end_date:
        title += f'\nPeríodo: {pd.Timestamp(start_date).strftime("%d %b")} - {pd.Timestamp(end_date).strftime("%d %b %Y")}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Cantidad de Reportes')
    plt.xlabel('Fecha y Hora')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:00'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_evolucion_temporal.png'))
    plt.close()

def generate_weekly_heatmap(df, start_date=None, end_date=None):
    print("🔥 Generating Fig 2: Weekly Heatmap...")
    plt.figure(figsize=(10, 6))
    
    df['hour'] = df['date'].dt.hour
    df['day_name'] = df['date'].dt.day_name()
    
    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    df['day_name'] = pd.Categorical(df['day_name'], categories=dias_orden, ordered=True)
    
    heatmap_data = df.groupby(['day_name', 'hour']).size().unstack(fill_value=0)
    heatmap_data.index = dias_es
    
    sns.heatmap(heatmap_data, cmap='YlOrRd', linewidths=.5, cbar_kws={'label': 'Nº Reportes'})
    title = 'Concentración de Reportes: Día de la Semana vs Hora'
    if start_date and end_date:
        title += f'\nPeríodo: {pd.Timestamp(start_date).strftime("%d %b")} - {pd.Timestamp(end_date).strftime("%d %b %Y")}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Hora del Día')
    plt.ylabel('Día de la Semana')
    
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_heatmap_semanal.png'))
    plt.close()

def generate_top_locations(df, start_date=None, end_date=None):
    print("📊 Generating Fig 3: Top Affected Locations...")
    
    df_filtered = df[~df['lugar_principal'].str.lower().str.contains('bloque', na=False)].copy()
    
    if df_filtered.empty:
        print("   ⚠️ No locations found after filtering.")
        return
    
    top_places = df_filtered['lugar_principal'].value_counts().head(15)
    
    plt.figure(figsize=(10, 8))
    
    sns.barplot(x=top_places.values, y=top_places.index, palette='viridis', hue=top_places.index, legend=False)
    title = 'Top 15 Zonas con Mayor Frecuencia de Reportes'
    if start_date and end_date:
        title += f'\nPeríodo: {pd.Timestamp(start_date).strftime("%d %b")} - {pd.Timestamp(end_date).strftime("%d %b %Y")}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Cantidad de Menciones')
    plt.ylabel('Zona / Municipio Identificado')
    
    for i, v in enumerate(top_places.values):
        plt.text(v + 0.5, i, str(v), color='black', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_top_lugares.png'))
    plt.close()

def generate_top_blocks(df, start_date=None, end_date=None):
    print("📦 Generating Fig 3b: Top Blocks Frequency...")
    
    df['block'] = df['text'].apply(extract_block_from_text)
    df_with_blocks = df[df['block'].notna()].copy()
    
    if df_with_blocks.empty:
        print("   ⚠️ No blocks found in messages.")
        return
    
    block_counts = df_with_blocks['block'].value_counts().sort_index()
    
    all_blocks = list(range(1, 7))
    block_counts_complete = pd.Series({b: block_counts.get(b, 0) for b in all_blocks})
    
    plt.figure(figsize=(10, 6))
    
    colors = sns.color_palette('Set2', n_colors=6)
    bars = plt.bar(all_blocks, block_counts_complete.values, color=colors)
    
    for block_num, count in block_counts_complete.items():
        if count > 0:
            plt.text(block_num, count + max(block_counts_complete.values) * 0.01, str(int(count)), 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.xlabel('Bloque', fontsize=12, fontweight='bold')
    plt.ylabel('Cantidad de Reportes', fontsize=12)
    plt.xticks(all_blocks, [f'Bloque {b}' for b in all_blocks], fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(bottom=0, top=max(block_counts_complete.values) * 1.15 if max(block_counts_complete.values) > 0 else 10)
    
    title = 'Frecuencia de Reportes por Bloque'
    if start_date and end_date:
        title += f'\nPeríodo: {pd.Timestamp(start_date).strftime("%d %b")} - {pd.Timestamp(end_date).strftime("%d %b %Y")}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3b_top_bloques.png'))
    plt.close()
    
    print(f"   -> Block distribution: {dict(block_counts_complete)}")
    print(f"   -> Total reports with block info: {len(df_with_blocks)}/{len(df)}")

def generate_top_words(df, start_date=None, end_date=None):
    print("📝 Generating Fig 3c: Top 50 Most Used Words...")
    
    stopwords_es = set([
        'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'se', 'por', 'con', 'no', 'si', 'del',
        'las', 'un', 'una', 'es', 'al', 'le', 'da', 'su', 'sus', 'lo', 'le', 'les', 'me', 'te',
        'nos', 'os', 'mi', 'tu', 'su', 'nuestro', 'vuestro', 'este', 'ese', 'aquel', 'este',
        'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas', 'aquel', 'aquella', 'aquellos',
        'aquellas', 'ser', 'estar', 'haber', 'tener', 'hacer', 'poder', 'decir', 'ir', 'ver',
        'dar', 'saber', 'querer', 'llegar', 'pasar', 'deber', 'poner', 'parecer', 'quedar',
        'hablar', 'llevar', 'dejar', 'seguir', 'encontrar', 'llamar', 'venir', 'pensar',
        'salir', 'volver', 'tomar', 'conocer', 'vivir', 'sentir', 'tratar', 'mirar', 'contar',
        'empezar', 'esperar', 'buscar', 'existir', 'entrar', 'trabajar', 'escribir', 'perder',
        'producir', 'ocurrir', 'entender', 'pedir', 'recibir', 'recordar', 'terminar', 'permitir',
        'aparecer', 'conseguir', 'comenzar', 'servir', 'sacar', 'necesitar', 'mantener', 'resultar',
        'leer', 'caer', 'cambiar', 'presentar', 'crear', 'abrir', 'considerar', 'oír', 'acabar',
        'convertir', 'ganar', 'formar', 'traer', 'partir', 'morir', 'aceptar', 'realizar',
        'suponer', 'comprender', 'lograr', 'explicar', 'preguntar', 'tocar', 'reconocer', 'estudiar',
        'alcanzar', 'nacer', 'dirigir', 'correr', 'utilizar', 'pagar', 'ayudar', 'gustar', 'jugar',
        'escuchar', 'cumplir', 'ofrecer', 'descubrir', 'levantar', 'intentar', 'usar', 'decidir',
        'repetir', 'dormir', 'cerrar', 'quedar', 'limpiar', 'empezar', 'cocinar', 'comprar',
        'vender', 'regresar', 'volver', 'salir', 'entrar', 'subir', 'bajar', 'caminar', 'correr',
        'saltar', 'nadar', 'volar', 'conducir', 'manejar', 'parar', 'continuar', 'seguir',
        'empezar', 'terminar', 'acabar', 'comenzar', 'iniciar', 'finalizar', 'concluir',
        'bloque', 'bloques', 'corriente', 'luz', 'apagon', 'apagón', 'fui', 'vino', 'quito', 'puso'
    ])
    
    all_text = ' '.join(df['text'].astype(str).str.lower())
    words = re.findall(r'\b[a-záéíóúñü]+\b', all_text)
    words_filtered = [w for w in words if w not in stopwords_es and len(w) > 2]
    
    word_counts = Counter(words_filtered)
    top_50_words = word_counts.most_common(50)
    
    if not top_50_words:
        print("   ⚠️ No words found after filtering.")
        return
    
    words_list, counts_list = zip(*top_50_words)
    
    plt.figure(figsize=(12, 14))
    
    y_pos = range(len(words_list))
    bars = plt.barh(y_pos, counts_list, color=sns.color_palette('viridis', len(words_list)))
    
    plt.yticks(y_pos, words_list, fontsize=9)
    plt.xlabel('Frecuencia', fontsize=12, fontweight='bold')
    plt.ylabel('Palabra', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    
    for i, (word, count) in enumerate(top_50_words):
        plt.text(count + max(counts_list) * 0.01, i, str(count), 
                va='center', fontsize=8, fontweight='bold')
    
    title = 'Top 50 Palabras Más Usadas en los Mensajes'
    if start_date and end_date:
        title += f'\nPeríodo: {pd.Timestamp(start_date).strftime("%d %b")} - {pd.Timestamp(end_date).strftime("%d %b %Y")}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3c_top_palabras.png'))
    plt.close()
    
    print(f"   -> Top 10 words: {dict(top_50_words[:10])}")

def generate_network_topology(df_rel, start_date=None, end_date=None, weight_threshold=2):
    print("🕸️ Generating Fig 4: Network Graph...")
    plt.figure(figsize=(12, 12))
    
    if df_rel.empty:
        print("   ⚠️ No relationships to plot.")
        plt.close()
        return
    
    G = nx.from_pandas_edgelist(df_rel, 'Source', 'Target', edge_attr='Weight')
    
    edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['Weight'] >= weight_threshold]
    G_filtered = G.edge_subgraph(edges_to_keep)
    
    if len(G_filtered.nodes()) == 0:
        print("   ⚠️ No nodes after filtering.")
        plt.close()
        return
    
    pos = nx.spring_layout(G_filtered, k=0.5, iterations=50, seed=42)
    
    node_sizes = [v * 100 for v in dict(G_filtered.degree()).values()]
    edge_widths = [d['Weight'] * 0.5 for u, v, d in G_filtered.edges(data=True)]
    
    nx.draw_networkx_nodes(G_filtered, pos, node_size=node_sizes, node_color='#3498db', alpha=0.8)
    nx.draw_networkx_edges(G_filtered, pos, width=edge_widths, alpha=0.4, edge_color='gray')
    nx.draw_networkx_labels(G_filtered, pos, font_size=8, font_family='sans-serif', font_weight='bold')
    
    title = f'Grafo de Co-ocurrencia de Cortes (Topología Inferida)\nFiltro: Conexiones con >= {weight_threshold} reportes conjuntos'
    if start_date and end_date:
        title = f'Grafo de Co-ocurrencia de Cortes (Topología Inferida)\nPeríodo: {pd.Timestamp(start_date).strftime("%d %b")} - {pd.Timestamp(end_date).strftime("%d %b %Y")} | Filtro: Conexiones con >= {weight_threshold} reportes conjuntos'
    plt.title(title, fontsize=14)
    plt.axis('off')
    
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_topologia_red.png'))
    plt.close()

def generate_transition_matrix(df, start_date=None, end_date=None):
    print("🎲 Generating Fig 5: Conditional Probability Matrix (Domino Effect)...")
    
    df['time_block'] = df['date'].dt.floor('10T')
    
    top_lugares = df['lugar_principal'].value_counts().head(20).index
    df_top = df[df['lugar_principal'].isin(top_lugares)]
    
    pivot = pd.crosstab(df_top['time_block'], df_top['lugar_principal'])
    pivot = (pivot > 0).astype(int)
    
    cooccurrence = pivot.T.dot(pivot)
    diagonal = np.diag(cooccurrence)
    prob_matrix = cooccurrence / diagonal[:, None]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(prob_matrix, annot=False, cmap="rocket_r", linewidths=.5)
    title = "Matriz de Probabilidad Condicional: Si cae X, ¿cae Y?"
    if start_date and end_date:
        title += f"\nPeríodo: {pd.Timestamp(start_date).strftime('%d %b')} - {pd.Timestamp(end_date).strftime('%d %b %Y')}"
    plt.title(title, fontsize=16)
    plt.xlabel("Lugar Afectado (Consecuencia)")
    plt.ylabel("Lugar Reportado Inicialmente (Causa Potencial)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_matriz_domino.png'))
    plt.close()

def generate_semantic_map_mds(df, start_date=None, end_date=None):
    print("🗺️ Generating Fig 6: Semantic Map (Topological Reconstruction)...")
    
    df['time_block'] = df['date'].dt.floor('10T')
    
    top_lugares = df['lugar_principal'].value_counts().head(30).index
    df_top = df[df['lugar_principal'].isin(top_lugares)]
    
    pivot = pd.crosstab(df_top['time_block'], df_top['lugar_principal'])
    corr = pivot.corr().fillna(0)
    distancia = 1 - corr
    
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distancia)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(coords[:, 0], coords[:, 1], color='red', s=100)
    
    for i, txt in enumerate(distancia.index):
        plt.annotate(txt, (coords[i, 0]+0.02, coords[i, 1]+0.02), fontsize=9, alpha=0.8)
    
    title = "Mapa Semántico de la Red Eléctrica (Distancia basada en Fallos Simultáneos)"
    if start_date and end_date:
        title += f"\nPeríodo: {pd.Timestamp(start_date).strftime('%d %b')} - {pd.Timestamp(end_date).strftime('%d %b %Y')}"
    plt.title(title, fontsize=14)
    plt.xlabel("Dimensión Latente 1")
    plt.ylabel("Dimensión Latente 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_mapa_mds.png'))
    plt.close()

def generate_sentiment_phases(df, start_date=None, end_date=None):
    print("🗣️ Generating Fig 7: Word Comparison (Start vs End)...")
    
    def classify_phase(texto):
        texto = texto.lower()
        if any(k in texto for k in KEYWORDS_INICIO): return 'Inicio (Ira/Reporte)'
        if any(k in texto for k in KEYWORDS_FIN): return 'Fin (Alivio/Aviso)'
        return 'Neutro'
    
    df['fase'] = df['text'].apply(classify_phase)
    
    fases = ['Inicio (Ira/Reporte)', 'Fin (Alivio/Aviso)']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, fase in enumerate(fases):
        textos = df[df['fase'] == fase]['text'].str.lower().str.cat(sep=' ')
        stopwords = set(['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'se', 'por', 'con', 'no', 'si', 'del'])
        palabras = [p for p in re.findall(r'\w+', textos) if p not in stopwords and len(p) > 3]
        
        common = Counter(palabras).most_common(10)
        
        if common:
            words, counts = zip(*common)
            sns.barplot(x=list(counts), y=list(words), ax=axes[i], palette='viridis')
            axes[i].set_title(f"Top Palabras: {fase}")
            axes[i].set_xlabel("Frecuencia")
    
    title = "Vocabulario Diferencial según la Fase del Apagón"
    if start_date and end_date:
        title += f"\nPeríodo: {pd.Timestamp(start_date).strftime('%d %b')} - {pd.Timestamp(end_date).strftime('%d %b %Y')}"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_sentimiento_fases.png'))
    plt.close()

def generate_day_night_pattern(df, start_date=None, end_date=None):
    print("☀️🌙 Generating Fig 8: Day/Night Contrast...")
    
    df['hora'] = df['date'].dt.hour
    df['periodo'] = df['hora'].apply(lambda x: 'Noche (19-06h)' if (x >= 19 or x < 6) else 'Día (07-18h)')
    
    top_lugares = df['lugar_principal'].value_counts().head(10).index
    df_top = df[df['lugar_principal'].isin(top_lugares)]
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_top, x='lugar_principal', hue='periodo', palette='coolwarm')
    title = "Distribución de Reportes Día vs Noche por Municipio"
    if start_date and end_date:
        title += f"\nPeríodo: {pd.Timestamp(start_date).strftime('%d %b')} - {pd.Timestamp(end_date).strftime('%d %b %Y')}"
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel("Cantidad de Reportes")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig8_dia_vs_noche.png'))
    plt.close()

def run_complete_analysis(start_date=None, end_date=None):
    print("\n" + "="*60)
    print("🔬 STARTING CONSOLIDATED ANALYSIS")
    print("="*60 + "\n")
    
    df, df_rel = load_and_filter_data(start_date, end_date)
    
    if df is None or df.empty:
        print("❌ No data to analyze.")
        return
    
    generate_temporal_evolution(df, start_date, end_date)
    generate_weekly_heatmap(df, start_date, end_date)
    generate_top_locations(df, start_date, end_date)
    generate_top_blocks(df, start_date, end_date)
    generate_top_words(df, start_date, end_date)
    generate_network_topology(df_rel, start_date, end_date)
    
    generate_transition_matrix(df, start_date, end_date)
    generate_semantic_map_mds(df, start_date, end_date)
    generate_sentiment_phases(df, start_date, end_date)
    generate_day_night_pattern(df, start_date, end_date)
    
    print(f"\n✅ Analysis completed! Generated 10 images in {FIGURES_DIR}/")

if __name__ == "__main__":
    run_complete_analysis(start_date='2024-11-05', end_date='2024-12-05')

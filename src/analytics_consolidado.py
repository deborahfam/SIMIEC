import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
import matplotlib.dates as mdates
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from collections import Counter
import re
import os

from utils.text_processing import extract_block_from_text
from config import (
    RESULTS_DIR, FIGURES_DIR, KEYWORDS_INICIO, KEYWORDS_FIN,
    STOPWORDS_EXTENSAS, STOPWORDS_BASICAS,
    DEFAULT_WEIGHT_THRESHOLD, DEFAULT_TOP_LOCATIONS_MATRIX,
    DEFAULT_TOP_LOCATIONS_MDS, DEFAULT_TIME_BLOCK_MINUTES,
    NIGHT_START_HOUR, NIGHT_END_HOUR, MIN_BLOCK_NUMBER, MAX_BLOCK_NUMBER,
    DEFAULT_TOP_WORDS
)

os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'sans-serif'

def load_and_filter_data(start_date=None, end_date=None):
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, 'datos_georeferenciados.csv'))
        df['date'] = pd.to_datetime(df['date'])
        df_rel = pd.read_csv(os.path.join(RESULTS_DIR, 'relaciones_lugares.csv'))
    except FileNotFoundError:
        print("❌ Error: Missing CSV files in results/")
        return None, None
    
    if start_date is not None or end_date is not None:
        if start_date is not None:
            start_date_ts = pd.Timestamp(start_date)
            df = df[df['date'] >= start_date_ts].copy()
        if end_date is not None:
            end_date_ts = pd.Timestamp(end_date)
            df = df[df['date'] <= end_date_ts].copy()
        
        lugares_periodo = set(df['lugar_principal'].unique())
        df_rel = df_rel[
            (df_rel['Source'].isin(lugares_periodo)) & 
            (df_rel['Target'].isin(lugares_periodo))
        ].copy()
    
    return df, df_rel

def generate_temporal_evolution(df, start_date=None, end_date=None):
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
    """
    Genera top lugares con intervalos de confianza (95%).
    NOTA: Frecuencia de reportes puede no reflejar frecuencia real de apagones
    debido a sesgo de muestreo (más usuarios activos = más reportes).
    """
    df_filtered = df[~df['lugar_principal'].str.lower().str.contains('bloque', na=False)].copy()
    
    if df_filtered.empty:
        return
    
    top_places = df_filtered['lugar_principal'].value_counts().head(15)
    total_reports = len(df_filtered)
    
    place_stats = []
    for lugar, count in top_places.items():
        prop = count / total_reports
        se = np.sqrt(prop * (1 - prop) / total_reports)
        ci_lower = max(0, prop - 1.96 * se)
        ci_upper = min(1, prop + 1.96 * se)
        place_stats.append({
            'lugar': lugar,
            'count': count,
            'proportion': prop,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_lower_count': ci_lower * total_reports,
            'ci_upper_count': ci_upper * total_reports
        })
    
    df_stats = pd.DataFrame(place_stats)
    
    plt.figure(figsize=(12, 8))
    
    y_pos = range(len(df_stats))
    
    bars = plt.barh(y_pos, df_stats['count'].values, 
                   color=sns.color_palette('viridis', len(df_stats)), alpha=0.8)
    
    for i, row in df_stats.iterrows():
        plt.errorbar(row['count'], i,
                    xerr=[[row['count'] - row['ci_lower_count']],
                          [row['ci_upper_count'] - row['count']]],
                    fmt='o', color='black', capsize=3, capthick=1.5, markersize=4)
        plt.text(row['count'] + row['ci_upper_count'] * 0.02, i, 
                f"{int(row['count'])} ({row['proportion']*100:.1f}%)",
                va='center', fontsize=9, fontweight='bold')
    
    plt.yticks(y_pos, df_stats['lugar'].values, fontsize=10)
    plt.xlabel('Cantidad de Reportes (con IC 95%)', fontsize=12, fontweight='bold')
    plt.ylabel('Zona / Municipio Identificado', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    
    title = 'Top 15 Zonas con Mayor Frecuencia de Reportes'
    if start_date and end_date:
        title += f'\nPeríodo: {pd.Timestamp(start_date).strftime("%d %b")} - {pd.Timestamp(end_date).strftime("%d %b %Y")}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.figtext(0.5, 0.02,
               "NOTA: Barras de error muestran intervalo de confianza 95%. Frecuencia de reportes puede no reflejar frecuencia real de apagones (sesgo de muestreo).",
               ha='center', fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_top_lugares.png'))
    plt.close()

def generate_top_blocks(df, start_date=None, end_date=None):
    if df is None or df.empty:
        return
    
    df['block'] = df['text'].apply(extract_block_from_text)
    df_with_blocks = df[df['block'].notna()].copy()
    
    if df_with_blocks.empty:
        return
    
    invalid_blocks = df_with_blocks[~df_with_blocks['block'].between(MIN_BLOCK_NUMBER, MAX_BLOCK_NUMBER)]
    if not invalid_blocks.empty:
        df_with_blocks = df_with_blocks[df_with_blocks['block'].between(MIN_BLOCK_NUMBER, MAX_BLOCK_NUMBER)]
    
    block_counts = df_with_blocks['block'].value_counts().sort_index()
    
    all_blocks = list(range(MIN_BLOCK_NUMBER, MAX_BLOCK_NUMBER + 1))
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

def generate_top_words(df, start_date=None, end_date=None):
    """
    Genera top palabras usando TF-IDF en lugar de frecuencia simple.
    TF-IDF penaliza palabras comunes en todos los documentos y destaca palabras distintivas.
    """
    if df is None or df.empty:
        return
    
    texts = df['text'].astype(str).tolist()
    
    if len(texts) < 2:
        return
    
    try:
        vectorizer = TfidfVectorizer(
            max_features=DEFAULT_TOP_WORDS,
            stop_words=list(STOPWORDS_EXTENSAS),
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            token_pattern=r'\b[a-záéíóúñü]+\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        word_scores = pd.Series(
            tfidf_matrix.sum(axis=0).A1,
            index=feature_names
        ).sort_values(ascending=False)
        
        if len(word_scores) == 0:
            return
        
        top_words = word_scores.head(DEFAULT_TOP_WORDS)
        words_list = top_words.index.tolist()
        scores_list = top_words.values.tolist()
        
        plt.figure(figsize=(12, 14))
        
        y_pos = range(len(words_list))
        bars = plt.barh(y_pos, scores_list, color=sns.color_palette('viridis', len(words_list)))
        
        plt.yticks(y_pos, words_list, fontsize=9)
        plt.xlabel('Score TF-IDF', fontsize=12, fontweight='bold')
        plt.ylabel('Palabra / Bigrama', fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
        
        for i, (word, score) in enumerate(zip(words_list, scores_list)):
            plt.text(score + max(scores_list) * 0.01, i, f'{score:.3f}', 
                    va='center', fontsize=8, fontweight='bold')
        
        title = f'Top {DEFAULT_TOP_WORDS} Palabras Más Importantes (TF-IDF)'
        if start_date and end_date:
            title += f'\nPeríodo: {pd.Timestamp(start_date).strftime("%d %b")} - {pd.Timestamp(end_date).strftime("%d %b %Y")}'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.figtext(0.5, 0.02,
                   "NOTA: TF-IDF destaca palabras distintivas, no solo frecuentes. Incluye bigramas (pares de palabras).",
                   ha='center', fontsize=9, style='italic', wrap=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig3c_top_palabras.png'))
        plt.close()
        
    except Exception as e:
        return

def generate_network_topology(df_rel, start_date=None, end_date=None, weight_threshold=None):
    if weight_threshold is None:
        weight_threshold = DEFAULT_WEIGHT_THRESHOLD
    
    if df_rel is None or df_rel.empty:
        return
    
    plt.figure(figsize=(12, 12))
    
    try:
        G = nx.from_pandas_edgelist(df_rel, 'Source', 'Target', edge_attr='Weight')
    except Exception as e:
        plt.close()
        return
    
    edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['Weight'] >= weight_threshold]
    G_filtered = G.edge_subgraph(edges_to_keep)
    
    if len(G_filtered.nodes()) == 0:
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

def generate_cooccurrence_matrix(df, start_date=None, end_date=None):
    """
    Calcula probabilidad de co-ocurrencia simultánea: P(Lugar B reportado | Lugar A reportado en mismo bloque temporal)
    NOTA: Esto mide co-ocurrencia simultánea, NO transición temporal causal.
    """
    if df is None or df.empty:
        return
    
    df['time_block'] = df['date'].dt.floor(f'{DEFAULT_TIME_BLOCK_MINUTES}T')
    
    top_lugares = df['lugar_principal'].value_counts().head(DEFAULT_TOP_LOCATIONS_MATRIX).index
    df_top = df[df['lugar_principal'].isin(top_lugares)]
    
    if len(df_top) == 0:
        return
    
    pivot = pd.crosstab(df_top['time_block'], df_top['lugar_principal'])
    pivot = (pivot > 0).astype(int)
    
    cooccurrence = pivot.T.dot(pivot)
    diagonal = np.diag(cooccurrence)
    
    prob_matrix = np.divide(
        cooccurrence,
        diagonal[:, None],
        out=np.zeros_like(cooccurrence, dtype=float),
        where=diagonal[:, None] != 0
    )
    
    prob_matrix = np.nan_to_num(prob_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(prob_matrix, annot=False, cmap="rocket_r", linewidths=.5, 
                cbar_kws={'label': 'Probabilidad de Co-ocurrencia'})
    title = "Matriz de Co-ocurrencia Simultánea\nP(Lugar B reportado | Lugar A reportado en mismo bloque temporal)"
    if start_date and end_date:
        title += f"\nPeríodo: {pd.Timestamp(start_date).strftime('%d %b')} - {pd.Timestamp(end_date).strftime('%d %b %Y')}"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Lugar B (Co-ocurre con A)")
    plt.ylabel("Lugar A (Reportado inicialmente)")
    plt.figtext(0.5, 0.02, 
                "NOTA: Mide co-ocurrencia simultánea, no causalidad temporal. Lugares que fallan juntos pueden compartir circuito eléctrico.",
                ha='center', fontsize=9, style='italic', wrap=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_matriz_coocurrencia.png'))
    plt.close()

def generate_transition_matrix(df, start_date=None, end_date=None, max_time_window_minutes=30):
    """
    Calcula matriz de transición temporal REAL: P(Lugar B reportado | Lugar A reportado ANTES)
    Mide si un apagón en A precede a un apagón en B dentro de una ventana temporal.
    """
    if df is None or df.empty:
        return
    
    df_sorted = df.sort_values('date').copy()
    top_lugares = df['lugar_principal'].value_counts().head(DEFAULT_TOP_LOCATIONS_MATRIX).index
    df_top = df_sorted[df_sorted['lugar_principal'].isin(top_lugares)].copy()
    
    if len(df_top) < 2:
        return
    
    transitions = []
    for i in range(len(df_top) - 1):
        lugar_actual = df_top.iloc[i]['lugar_principal']
        lugar_siguiente = df_top.iloc[i+1]['lugar_principal']
        tiempo_diff = (df_top.iloc[i+1]['date'] - df_top.iloc[i]['date']).total_seconds() / 60
        
        if tiempo_diff <= max_time_window_minutes and lugar_actual != lugar_siguiente:
            transitions.append((lugar_actual, lugar_siguiente))
    
    if not transitions:
        return
    
    transition_df = pd.DataFrame(transitions, columns=['From', 'To'])
    transition_counts = pd.crosstab(transition_df['From'], transition_df['To'])
    
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)
    
    transition_matrix = transition_matrix.reindex(
        index=top_lugares, 
        columns=top_lugares, 
        fill_value=0
    )
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(transition_matrix, annot=False, cmap="viridis", linewidths=.5,
                cbar_kws={'label': 'Probabilidad de Transición'})
    title = f"Matriz de Transición Temporal Real\nP(Lugar B reportado | Lugar A reportado ANTES, ventana ≤{max_time_window_minutes} min)"
    if start_date and end_date:
        title += f"\nPeríodo: {pd.Timestamp(start_date).strftime('%d %b')} - {pd.Timestamp(end_date).strftime('%d %b %Y')}"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Lugar B (Reportado después)")
    plt.ylabel("Lugar A (Reportado primero)")
    plt.figtext(0.5, 0.02,
                f"NOTA: Mide transición temporal real. Valores altos sugieren que apagones en A pueden preceder apagones en B (posible efecto dominó).",
                ha='center', fontsize=9, style='italic', wrap=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5b_matriz_transicion_temporal.png'))
    plt.close()

def generate_semantic_map_mds(df, start_date=None, end_date=None):
    """
    Genera mapa semántico usando MDS con distancia de Jaccard.
    La distancia de Jaccard es apropiada para datos binarios (presencia/ausencia).
    """
    if df is None or df.empty:
        return
    
    df['time_block'] = df['date'].dt.floor(f'{DEFAULT_TIME_BLOCK_MINUTES}T')
    
    top_lugares = df['lugar_principal'].value_counts().head(DEFAULT_TOP_LOCATIONS_MDS).index
    
    if len(top_lugares) < 3:
        return
    
    df_top = df[df['lugar_principal'].isin(top_lugares)]
    
    if len(df_top) == 0:
        return
    
    pivot = pd.crosstab(df_top['time_block'], df_top['lugar_principal'])
    pivot_binary = (pivot > 0).astype(int)
    
    if pivot_binary.empty or pivot_binary.shape[1] < 3:
        return
    
    try:
        jaccard_distances = pairwise_distances(
            pivot_binary.T.values,
            metric='jaccard'
        )
        
        jaccard_distances = pd.DataFrame(
            jaccard_distances,
            index=pivot_binary.columns,
            columns=pivot_binary.columns
        )
        
        if jaccard_distances.isna().all().all() or (jaccard_distances == 0).all().all():
            return
        
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
                  normalized_stress='auto', max_iter=300)
        coords = mds.fit_transform(jaccard_distances.values)
        
        stress = mds.stress_
        
    except Exception:
        return
    
    plt.figure(figsize=(12, 10))
    plt.scatter(coords[:, 0], coords[:, 1], color='red', s=100, alpha=0.7)
    
    for i, txt in enumerate(jaccard_distances.index):
        plt.annotate(txt, (coords[i, 0]+0.02, coords[i, 1]+0.02), 
                    fontsize=9, alpha=0.8)
    
    title = "Mapa Semántico de la Red Eléctrica\n(Distancia de Jaccard basada en Patrones de Co-ocurrencia)"
    if start_date and end_date:
        title += f"\nPeríodo: {pd.Timestamp(start_date).strftime('%d %b')} - {pd.Timestamp(end_date).strftime('%d %b %Y')}"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Dimensión Latente 1")
    plt.ylabel("Dimensión Latente 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    stress_text = f"MDS Stress: {stress:.4f}"
    if stress < 0.1:
        stress_text += " (Excelente)"
    elif stress < 0.2:
        stress_text += " (Bueno)"
    else:
        stress_text += " (Aceptable)"
    
    plt.figtext(0.5, 0.02,
               f"{stress_text}. Lugares cercanos tienen patrones similares de fallos simultáneos. NO validado como distancia eléctrica real.",
               ha='center', fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_mapa_mds.png'))
    plt.close()

def calculate_sentiment_score(texto):
    """
    Calcula score de sentimiento basado en keywords y palabras emocionales.
    Retorna score entre -1 (muy negativo) y 1 (muy positivo).
    """
    if not isinstance(texto, str):
        return 0.0
    
    texto_lower = texto.lower()
    score = 0.0
    
    # Keywords de inicio (negativo)
    inicio_weights = {
        'se fue': -0.8, 'quito': -0.7, 'apagon': -0.6, 'apagón': -0.6,
        'ñooo': -0.5, 'pinga': -0.4, 'coño': -0.4, 'otra vez': -0.7,
        'sin luz': -0.6, 'se cortó': -0.7, 'se fue la luz': -0.8
    }
    
    # Keywords de fin (positivo)
    fin_weights = {
        'llego': 0.7, 'llegó': 0.7, 'vino': 0.7, 'pusieron': 0.6,
        'gracias': 0.5, 'al fin': 0.6, 'ya hay': 0.6, 'regresó': 0.7,
        'volvió': 0.7, 'ya está': 0.5
    }
    
    # Palabras emocionales adicionales
    negative_words = ['mal', 'horrible', 'terrible', 'frustrado', 'molesto', 
                     'cansado', 'harto', 'fastidiado', 'problema', 'falla']
    positive_words = ['bien', 'bueno', 'excelente', 'aliviado', 'contento',
                     'feliz', 'satisfecho', 'resuelto', 'solucionado']
    
    # Calcular score
    for keyword, weight in inicio_weights.items():
        if keyword in texto_lower:
            score += weight
    
    for keyword, weight in fin_weights.items():
        if keyword in texto_lower:
            score += weight
    
    for word in negative_words:
        if word in texto_lower:
            score -= 0.2
    
    for word in positive_words:
        if word in texto_lower:
            score += 0.2
    
    # Normalizar entre -1 y 1
    return max(-1.0, min(1.0, score))

def classify_phase_improved(texto):
    """
    Clasifica fase del apagón usando score de sentimiento + keywords.
    Método más robusto que solo keywords.
    """
    texto_lower = texto.lower() if isinstance(texto, str) else ""
    sentiment_score = calculate_sentiment_score(texto)
    
    # Combinar keywords con sentimiento
    inicio_keywords = any(k in texto_lower for k in KEYWORDS_INICIO)
    fin_keywords = any(k in texto_lower for k in KEYWORDS_FIN)
    
    if inicio_keywords and sentiment_score < -0.1:
        return 'Inicio (Ira/Reporte)'
    elif fin_keywords and sentiment_score > 0.1:
        return 'Fin (Alivio/Aviso)'
    elif sentiment_score < -0.3:
        return 'Inicio (Ira/Reporte)'
    elif sentiment_score > 0.3:
        return 'Fin (Alivio/Aviso)'
    else:
        return 'Neutro'

def generate_sentiment_phases(df, start_date=None, end_date=None):
    if df is None or df.empty:
        return
    
    df['fase'] = df['text'].apply(classify_phase_improved)
    df['sentiment_score'] = df['text'].apply(calculate_sentiment_score)
    
    fases = ['Inicio (Ira/Reporte)', 'Fin (Alivio/Aviso)']
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.8])
    
    # Subplot 1 y 2: Top palabras por fase
    for i, fase in enumerate(fases):
        ax = fig.add_subplot(gs[0, i])
        df_fase = df[df['fase'] == fase]
        
        if len(df_fase) == 0:
            ax.text(0.5, 0.5, f'No hay datos para {fase}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Top Palabras: {fase}")
            continue
        
        textos = df_fase['text'].str.lower().str.cat(sep=' ')
        palabras = [p for p in re.findall(r'\w+', textos) 
                   if p not in STOPWORDS_BASICAS and len(p) > 3]
        
        common = Counter(palabras).most_common(10)
        
        if common:
            words, counts = zip(*common)
            sns.barplot(x=list(counts), y=list(words), ax=ax, palette='viridis')
            ax.set_title(f"Top Palabras: {fase}\n(n={len(df_fase)} mensajes)")
            ax.set_xlabel("Frecuencia")
        else:
            ax.text(0.5, 0.5, 'No hay palabras suficientes', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Subplot 3: Distribución de sentimiento
    ax3 = fig.add_subplot(gs[0, 2])
    df_phases = df[df['fase'].isin(fases)]
    if len(df_phases) > 0:
        sns.histplot(data=df_phases, x='sentiment_score', hue='fase', 
                    bins=20, ax=ax3, alpha=0.7, palette=['red', 'green'])
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Score de Sentimiento')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribución de Sentimiento')
        ax3.legend(title='Fase')
    
    title = "Análisis de Sentimiento por Fase del Apagón"
    if start_date and end_date:
        title += f"\nPeríodo: {pd.Timestamp(start_date).strftime('%d %b')} - {pd.Timestamp(end_date).strftime('%d %b %Y')}"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    # Estadísticas de clasificación
    phase_counts = df['fase'].value_counts()
    total_classified = phase_counts.sum()
    if total_classified > 0:
        stats_text = f"Clasificación: Inicio={phase_counts.get('Inicio (Ira/Reporte)', 0)}, "
        stats_text += f"Fin={phase_counts.get('Fin (Alivio/Aviso)', 0)}, "
        stats_text += f"Neutro={phase_counts.get('Neutro', 0)}"
        plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_sentimiento_fases.png'))
    plt.close()

def generate_day_night_pattern(df, start_date=None, end_date=None):
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
    df, df_rel = load_and_filter_data(start_date, end_date)
    
    if df is None or df.empty:
        return
    
    generate_temporal_evolution(df, start_date, end_date)
    generate_weekly_heatmap(df, start_date, end_date)
    generate_top_locations(df, start_date, end_date)
    generate_top_blocks(df, start_date, end_date)
    generate_top_words(df, start_date, end_date)
    generate_network_topology(df_rel, start_date, end_date)
    
    generate_cooccurrence_matrix(df, start_date, end_date)
    generate_transition_matrix(df, start_date, end_date)
    generate_semantic_map_mds(df, start_date, end_date)
    generate_sentiment_phases(df, start_date, end_date)
    generate_day_night_pattern(df, start_date, end_date)

if __name__ == "__main__":
    run_complete_analysis(start_date='2024-11-05', end_date='2024-12-05')

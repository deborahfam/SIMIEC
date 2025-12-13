import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
import sys
import os
import calendar

# Agregar src al path para importar módulos
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from analytics_consolidado import (
    calculate_sentiment_score,
    classify_phase_improved
)
from utils.text_processing import extract_block_from_text
from config import (
    RESULTS_DIR, KEYWORDS_INICIO, KEYWORDS_FIN,
    STOPWORDS_EXTENSAS, STOPWORDS_BASICAS,
    DEFAULT_WEIGHT_THRESHOLD, DEFAULT_TOP_LOCATIONS_MATRIX,
    DEFAULT_TOP_LOCATIONS_MDS, DEFAULT_TIME_BLOCK_MINUTES,
    MIN_BLOCK_NUMBER, MAX_BLOCK_NUMBER, DEFAULT_TOP_WORDS
)

# Ajustar RESULTS_DIR para que sea relativo a la raíz del proyecto
RESULTS_DIR_ABS = os.path.join(os.path.dirname(__file__), 'src', RESULTS_DIR)
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

st.set_page_config(
    page_title="SIMIEC Dashboard", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.title("⚡ SIMIEC")
st.markdown("""
**Sistema Integral de Monitorización de Incidencias Eléctricas Cubano.**
*Visualización interactiva de datos extraídos mediante NLP de reportes ciudadanos.*
""")

@st.cache_data
def load_data_cached():
    """Carga datos desde los archivos CSV - TODO calculado en tiempo real"""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR_ABS, 'datos_georeferenciados.csv'))
        df['date'] = pd.to_datetime(df['date'])
        
        try:
            df_rel = pd.read_csv(os.path.join(RESULTS_DIR_ABS, 'relaciones_lugares.csv'))
        except FileNotFoundError:
            df_rel = pd.DataFrame()
            
        return df, df_rel
    except FileNotFoundError:
        return None, None

def filter_data_by_date(df, df_rel, start_date=None, end_date=None):
    """
    Filtra datos por rango de fechas - TODO calculado en tiempo real
    Replica la lógica de load_and_filter_data pero sin depender de la función
    IMPORTANTE: Incluye todo el día de inicio y fin (hasta 23:59:59)
    """
    if df is None or df.empty:
        return df, df_rel
    
    df_filtered = df.copy()
    
    if start_date is not None:
        # Incluir desde el inicio del día (00:00:00)
        start_date_ts = pd.Timestamp(start_date).normalize()
        df_filtered = df_filtered[df_filtered['date'] >= start_date_ts].copy()
    
    if end_date is not None:
        # Incluir hasta el final del día (23:59:59.999999)
        end_date_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df_filtered = df_filtered[df_filtered['date'] <= end_date_ts].copy()
    
    if df_rel is not None and not df_rel.empty:
        lugares_periodo = set(df_filtered['lugar_principal'].unique())
        df_rel_filtered = df_rel[
            (df_rel['Source'].isin(lugares_periodo)) & 
            (df_rel['Target'].isin(lugares_periodo))
        ].copy()
    else:
        df_rel_filtered = df_rel
    
    return df_filtered, df_rel_filtered

df_raw, df_rel_raw = load_data_cached()

if df_raw is None:
    st.error("❌ No se encontraron los archivos CSV. Ejecuta los scripts de procesamiento primero.")
    st.stop()

# ========== SIDEBAR - FILTROS ==========
st.sidebar.header("🔧 Filtros de Análisis")

min_date = df_raw['date'].min().date()
max_date = df_raw['date'].max().date()

st.sidebar.markdown("### 📅 Selección de Período")

col_date1, col_date2 = st.sidebar.columns(2)

with col_date1:
    st.markdown("**Fecha Inicio**")
    start_year = st.selectbox(
        "Año Inicio",
        options=range(min_date.year, max_date.year + 1),
        index=0,
        key="start_year"
    )
    start_month = st.selectbox(
        "Mes Inicio",
        options=range(1, 13),
        index=min_date.month - 1 if start_year == min_date.year else 0,
        format_func=lambda x: ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'][x-1],
        key="start_month"
    )
    
    # Calcular días disponibles según mes y año
    days_in_month = calendar.monthrange(start_year, start_month)[1]
    if start_year == min_date.year and start_month == min_date.month:
        min_day = min_date.day
    else:
        min_day = 1
    
    start_day = st.selectbox(
        "Día Inicio",
        options=range(min_day, days_in_month + 1),
        index=0,
        key="start_day"
    )

with col_date2:
    st.markdown("**Fecha Fin**")
    end_year = st.selectbox(
        "Año Fin",
        options=range(min_date.year, max_date.year + 1),
        index=len(range(min_date.year, max_date.year + 1)) - 1,
        key="end_year"
    )
    end_month = st.selectbox(
        "Mes Fin",
        options=range(1, 13),
        index=max_date.month - 1 if end_year == max_date.year else 11,
        format_func=lambda x: ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'][x-1],
        key="end_month"
    )
    
    # Calcular días disponibles según mes y año
    days_in_month_end = calendar.monthrange(end_year, end_month)[1]
    if end_year == max_date.year and end_month == max_date.month:
        max_day = max_date.day
    else:
        max_day = days_in_month_end
    
    end_day = st.selectbox(
        "Día Fin",
        options=range(1, max_day + 1),
        index=min(max_day - 1, max_date.day - 1) if end_year == max_date.year and end_month == max_date.month else max_day - 1,
        key="end_day"
    )

# Construir fechas
try:
    start_date_obj = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    end_date_obj = pd.Timestamp(year=end_year, month=end_month, day=end_day)
    
    # Validar que fecha inicio <= fecha fin
    if start_date_obj > end_date_obj:
        st.sidebar.error("⚠️ La fecha de inicio debe ser anterior a la fecha de fin.")
        start_date = None
        end_date = None
    else:
        start_date = start_date_obj.strftime('%Y-%m-%d')
        end_date = end_date_obj.strftime('%Y-%m-%d')
        
        st.sidebar.info(f"📊 Período: {start_date_obj.strftime('%d/%m/%Y')} - {end_date_obj.strftime('%d/%m/%Y')}")
except Exception as e:
    st.sidebar.error(f"Error en fechas: {e}")
    start_date = None
    end_date = None

# Filtrar datos en tiempo real - se recalcula automáticamente cuando cambian las fechas
df, df_rel = filter_data_by_date(df_raw, df_rel_raw, start_date, end_date)

if df is None or df.empty:
    st.error("❌ No hay datos en el rango de fechas seleccionado.")
    st.stop()

# ========== MÉTRICAS PRINCIPALES ==========
st.header("📊 Resumen Ejecutivo")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Reportes", f"{len(df):,}")

with col2:
    st.metric("Lugares Únicos", f"{df['lugar_principal'].nunique()}")

with col3:
    df['hour'] = df['date'].dt.hour
    hora_pico = df['hour'].mode()
    hora_pico_val = hora_pico[0] if len(hora_pico) > 0 else 0
    st.metric("Hora Pico", f"{hora_pico_val}:00")

with col4:
    df['block'] = df['text'].apply(extract_block_from_text)
    bloques_detectados = df['block'].notna().sum()
    st.metric("Reportes con Bloque", f"{bloques_detectados}")

# ========== TABS PRINCIPALES ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⏰ Temporal", 
    "📍 Espacial", 
    "🕸️ Red", 
    "📝 Texto", 
    "📈 Matrices"
])

# ========== TAB 1: ANÁLISIS TEMPORAL ==========
with tab1:
    st.header("⏰ Análisis Temporal")
    
    col_t1_1, col_t1_2 = st.columns(2)
    
    with col_t1_1:
        st.subheader("Evolución Temporal (Por Hora)")
        # CALCULADO EN TIEMPO REAL: Resample por hora desde datos filtrados
        timeline = df.set_index('date').resample('h')['text'].count().reset_index()
        timeline.columns = ['Fecha', 'Reportes']
        
        fig_timeline = px.line(
            timeline, 
            x='Fecha', 
            y='Reportes',
            title="Frecuencia de Reportes por Hora",
            labels={'Reportes': 'Cantidad de Reportes', 'Fecha': 'Fecha y Hora'}
        )
        fig_timeline.update_traces(line_color='#d62728', line_width=2)
        fig_timeline.update_layout(hovermode='x unified')
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col_t1_2:
        st.subheader("Mapa de Calor Semanal")
        # CALCULADO EN TIEMPO REAL: Agrupación por día y hora desde datos filtrados
        df['day_name'] = df['date'].dt.day_name()
        dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        df['day_name'] = pd.Categorical(df['day_name'], categories=dias_orden, ordered=True)
        
        heatmap_data = df.groupby(['day_name', 'hour']).size().reset_index(name='counts')
        heatmap_data['day_name_es'] = heatmap_data['day_name'].map(dict(zip(dias_orden, dias_es)))
        
        fig_heat = px.density_heatmap(
            heatmap_data,
            x='hour',
            y='day_name_es',
            z='counts',
            title="Concentración de Reportes: Día vs Hora",
            color_continuous_scale='YlOrRd',
            category_orders={"day_name_es": dias_es},
            labels={'hour': 'Hora del Día', 'day_name_es': 'Día de la Semana', 'counts': 'Nº Reportes'}
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    
    st.subheader("Patrón Día vs Noche")
    df['periodo'] = df['hour'].apply(
        lambda x: 'Noche (19-06h)' if (x >= 19 or x < 6) else 'Día (07-18h)'
    )
    
    top_lugares_dn = df['lugar_principal'].value_counts().head(10).index
    df_dn = df[df['lugar_principal'].isin(top_lugares_dn)]
    
    fig_dn = px.histogram(
        df_dn,
        x='lugar_principal',
        color='periodo',
        title="Distribución de Reportes Día vs Noche por Municipio",
        labels={'lugar_principal': 'Municipio', 'count': 'Cantidad de Reportes'},
        color_discrete_map={'Día (07-18h)': '#FFA500', 'Noche (19-06h)': '#4169E1'},
        barmode='group'
    )
    fig_dn.update_xaxes(tickangle=45)
    st.plotly_chart(fig_dn, use_container_width=True)

# ========== TAB 2: ANÁLISIS ESPACIAL ==========
with tab2:
    st.header("📍 Análisis Espacial")
    
    col_t2_1, col_t2_2 = st.columns([2, 1])
    
    with col_t2_1:
        st.subheader("Top Lugares con Mayor Frecuencia de Reportes")
        
        # CALCULADO EN TIEMPO REAL: Filtrado y conteo desde datos filtrados
        df_filtered_locs = df[~df['lugar_principal'].str.lower().str.contains('bloque', na=False)].copy()
        top_places = df_filtered_locs['lugar_principal'].value_counts().head(15)
        total_reports = len(df_filtered_locs)
        
        # CALCULADO EN TIEMPO REAL: Intervalos de confianza calculados dinámicamente
        place_stats = []
        for lugar, count in top_places.items():
            prop = count / total_reports
            se = np.sqrt(prop * (1 - prop) / total_reports)
            ci_lower = max(0, prop - 1.96 * se)
            ci_upper = min(1, prop + 1.96 * se)
            place_stats.append({
                'Lugar': lugar,
                'Reportes': count,
                'Proporción': prop,
                'IC_Lower': ci_lower * total_reports,
                'IC_Upper': ci_upper * total_reports
            })
        
        df_stats = pd.DataFrame(place_stats)
        
        fig_locs = go.Figure()
        
        fig_locs.add_trace(go.Bar(
            y=df_stats['Lugar'],
            x=df_stats['Reportes'],
            orientation='h',
            name='Reportes',
            marker_color='#2E86AB',
            error_x=dict(
                type='data',
                array=df_stats['IC_Upper'] - df_stats['Reportes'],
                arrayminus=df_stats['Reportes'] - df_stats['IC_Lower'],
                visible=True
            )
        ))
        
        fig_locs.update_layout(
            title="Top 15 Zonas con Mayor Frecuencia de Reportes (con IC 95%)",
            xaxis_title="Cantidad de Reportes",
            yaxis_title="Zona / Municipio",
            yaxis={'categoryorder': 'total ascending'},
            height=600
        )
        
        st.plotly_chart(fig_locs, use_container_width=True)
        st.caption("⚠️ NOTA: Barras de error muestran intervalo de confianza 95%. Frecuencia de reportes puede no reflejar frecuencia real de apagones (sesgo de muestreo).")
    
    with col_t2_2:
        st.subheader("Datos Detallados")
        st.dataframe(
            df_stats[['Lugar', 'Reportes', 'Proporción']].style.format({
                'Proporción': '{:.2%}'
            }),
            use_container_width=True,
            height=600
        )
    
    st.subheader("Frecuencia de Reportes por Bloque Eléctrico")
    # CALCULADO EN TIEMPO REAL: Extracción de bloques desde texto de datos filtrados
    df['block'] = df['text'].apply(extract_block_from_text)
    df_with_blocks = df[df['block'].notna()].copy()
    
    if not df_with_blocks.empty:
        invalid_blocks = df_with_blocks[~df_with_blocks['block'].between(MIN_BLOCK_NUMBER, MAX_BLOCK_NUMBER)]
        if not invalid_blocks.empty:
            df_with_blocks = df_with_blocks[df_with_blocks['block'].between(MIN_BLOCK_NUMBER, MAX_BLOCK_NUMBER)]
        
        block_counts = df_with_blocks['block'].value_counts().sort_index()
        all_blocks = list(range(MIN_BLOCK_NUMBER, MAX_BLOCK_NUMBER + 1))
        block_counts_complete = pd.Series({b: block_counts.get(b, 0) for b in all_blocks})
        
        fig_blocks = px.bar(
            x=[f'Bloque {b}' for b in all_blocks],
            y=block_counts_complete.values,
            title="Frecuencia de Reportes por Bloque",
            labels={'x': 'Bloque', 'y': 'Cantidad de Reportes'},
            color=block_counts_complete.values,
            color_continuous_scale='viridis'
        )
        fig_blocks.update_layout(showlegend=False)
        st.plotly_chart(fig_blocks, use_container_width=True)
    else:
        st.info("No se encontraron reportes con información de bloque.")

# ========== TAB 3: ANÁLISIS DE RED ==========
with tab3:
    st.header("🕸️ Análisis de Red")
    
    if df_rel is not None and not df_rel.empty:
        weight_threshold = st.slider(
            "Filtrar conexiones débiles (Peso mínimo)",
            1, 20, DEFAULT_WEIGHT_THRESHOLD
        )
        
        rel_filtered = df_rel[df_rel['Weight'] >= weight_threshold].copy()
        
        if not rel_filtered.empty:
            G = nx.from_pandas_edgelist(rel_filtered, 'Source', 'Target', edge_attr='Weight')
            
            if len(G.nodes()) > 0:
                pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
                
                edge_x = []
                edge_y = []
                edge_info = []
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_info.append(edge[2]['Weight'])
                
                node_x = []
                node_y = []
                node_text = []
                node_sizes = []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    node_sizes.append(G.degree(node) * 10)
                
                fig_net = go.Figure()
                
                fig_net.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                ))
                
                fig_net.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    name='Nodos',
                    marker=dict(
                        size=node_sizes,
                        color='#3498db',
                        line=dict(width=2, color='white')
                    ),
                    text=node_text,
                    textposition="middle center",
                    hoverinfo='text'
                ))
                
                fig_net.update_layout(
                    title=f'Grafo de Co-ocurrencia de Cortes<br>Filtro: Conexiones con >= {weight_threshold} reportes conjuntos',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=700
                )
                
                st.plotly_chart(fig_net, use_container_width=True)
                st.caption("⚠️ NOTA: Co-ocurrencia en mensajes, no necesariamente conexión eléctrica real.")
            else:
                st.warning("No hay nodos después del filtrado.")
        else:
            st.warning("No hay relaciones que cumplan el criterio de filtrado.")
    else:
        st.warning("No hay datos de relaciones disponibles.")

# ========== TAB 4: ANÁLISIS DE TEXTO ==========
with tab4:
    st.header("📝 Análisis de Texto y Sentimiento")
    
    col_t4_1, col_t4_2 = st.columns(2)
    
    with col_t4_1:
        st.subheader("Top Palabras (TF-IDF)")
        
        # CALCULADO EN TIEMPO REAL: TF-IDF calculado desde textos de datos filtrados
        texts = df['text'].astype(str).tolist()
        
        if len(texts) >= 2:
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
                ).sort_values(ascending=False).head(DEFAULT_TOP_WORDS)
                
                fig_words = px.bar(
                    x=word_scores.values,
                    y=word_scores.index,
                    orientation='h',
                    title=f"Top {DEFAULT_TOP_WORDS} Palabras Más Importantes (TF-IDF)",
                    labels={'x': 'Score TF-IDF', 'y': 'Palabra / Bigrama'},
                    color=word_scores.values,
                    color_continuous_scale='viridis'
                )
                fig_words.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                st.plotly_chart(fig_words, use_container_width=True)
                st.caption("ℹ️ TF-IDF destaca palabras distintivas, no solo frecuentes. Incluye bigramas.")
            except Exception as e:
                st.error(f"Error calculando TF-IDF: {e}")
        else:
            st.info("Se necesitan al menos 2 documentos para calcular TF-IDF.")
    
    with col_t4_2:
        st.subheader("Análisis de Sentimiento por Fase")
        
        # CALCULADO EN TIEMPO REAL: Clasificación y scoring desde textos de datos filtrados
        df['fase'] = df['text'].apply(classify_phase_improved)
        df['sentiment_score'] = df['text'].apply(calculate_sentiment_score)
        
        phase_counts = df['fase'].value_counts()
        
        fig_sent = px.pie(
            values=phase_counts.values,
            names=phase_counts.index,
            title="Distribución de Fases del Apagón",
            color_discrete_map={
                'Inicio (Ira/Reporte)': '#FF4444',
                'Fin (Alivio/Aviso)': '#44FF44',
                'Neutro': '#CCCCCC'
            }
        )
        st.plotly_chart(fig_sent, use_container_width=True)
        
        df_phases = df[df['fase'].isin(['Inicio (Ira/Reporte)', 'Fin (Alivio/Aviso)'])]
        if not df_phases.empty:
            fig_sent_dist = px.histogram(
                df_phases,
                x='sentiment_score',
                color='fase',
                nbins=20,
                title="Distribución de Sentimiento",
                labels={'sentiment_score': 'Score de Sentimiento', 'count': 'Frecuencia'},
                color_discrete_map={
                    'Inicio (Ira/Reporte)': '#FF4444',
                    'Fin (Alivio/Aviso)': '#44FF44'
                }
            )
            fig_sent_dist.add_vline(x=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_sent_dist, use_container_width=True)

# ========== TAB 5: MATRICES ==========
with tab5:
    st.header("📈 Matrices de Análisis")
    
    tab5_1, tab5_2 = st.tabs(["Co-ocurrencia", "Transición Temporal"])
    
    with tab5_1:
        st.subheader("Matriz de Co-ocurrencia Simultánea")
        st.caption("P(Lugar B reportado | Lugar A reportado en mismo bloque temporal)")
        
        # CALCULADO EN TIEMPO REAL: Matriz de co-ocurrencia calculada desde datos filtrados
        df['time_block'] = df['date'].dt.floor(f'{DEFAULT_TIME_BLOCK_MINUTES}T')
        top_lugares_mat = df['lugar_principal'].value_counts().head(DEFAULT_TOP_LOCATIONS_MATRIX).index
        df_top_mat = df[df['lugar_principal'].isin(top_lugares_mat)]
        
        if not df_top_mat.empty:
            pivot = pd.crosstab(df_top_mat['time_block'], df_top_mat['lugar_principal'])
            pivot = (pivot > 0).astype(int)
            
            cooccurrence = pivot.T.dot(pivot)
            diagonal = np.diag(cooccurrence)
            
            # Convertir a numpy arrays para evitar problemas con pandas
            cooccurrence_arr = cooccurrence.values
            diagonal_arr = diagonal  # np.diag ya retorna un array, no DataFrame
            
            # Calcular matriz de probabilidad evitando división por cero
            prob_matrix = np.zeros_like(cooccurrence_arr, dtype=float)
            for i in range(len(diagonal_arr)):
                if diagonal_arr[i] > 0:
                    prob_matrix[i, :] = cooccurrence_arr[i, :] / diagonal_arr[i]
            
            prob_matrix = np.nan_to_num(prob_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            prob_df = pd.DataFrame(prob_matrix, index=cooccurrence.index, columns=cooccurrence.columns)
            
            # Usar colorscale válido de Plotly (rocket no existe, usar 'hot' o 'inferno' que son similares)
            fig_cooc = px.imshow(
                prob_df,
                color_continuous_scale='hot',
                title="Matriz de Co-ocurrencia Simultánea",
                labels=dict(x="Lugar B (Co-ocurre con A)", y="Lugar A (Reportado inicialmente)", color="Probabilidad")
            )
            fig_cooc.update_layout(height=700)
            st.plotly_chart(fig_cooc, use_container_width=True)
            st.caption("⚠️ NOTA: Mide co-ocurrencia simultánea, no causalidad temporal.")
        else:
            st.warning("No hay datos suficientes para calcular la matriz de co-ocurrencia.")
    
    with tab5_2:
        st.subheader("Matriz de Transición Temporal Real")
        st.caption("P(Lugar B reportado | Lugar A reportado ANTES)")
        
        max_window = st.slider("Ventana temporal máxima (minutos)", 10, 60, 30)
        
        # CALCULADO EN TIEMPO REAL: Matriz de transición calculada desde datos filtrados ordenados
        df_sorted = df.sort_values('date').copy()
        top_lugares_trans = df['lugar_principal'].value_counts().head(DEFAULT_TOP_LOCATIONS_MATRIX).index
        df_top_trans = df_sorted[df_sorted['lugar_principal'].isin(top_lugares_trans)].copy()
        
        if len(df_top_trans) >= 2:
            transitions = []
            for i in range(len(df_top_trans) - 1):
                lugar_actual = df_top_trans.iloc[i]['lugar_principal']
                lugar_siguiente = df_top_trans.iloc[i+1]['lugar_principal']
                tiempo_diff = (df_top_trans.iloc[i+1]['date'] - df_top_trans.iloc[i]['date']).total_seconds() / 60
                
                if tiempo_diff <= max_window and lugar_actual != lugar_siguiente:
                    transitions.append((lugar_actual, lugar_siguiente))
            
            if transitions:
                transition_df = pd.DataFrame(transitions, columns=['From', 'To'])
                transition_counts = pd.crosstab(transition_df['From'], transition_df['To'])
                transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)
                transition_matrix = transition_matrix.reindex(
                    index=top_lugares_trans,
                    columns=top_lugares_trans,
                    fill_value=0
                )
                
                fig_trans = px.imshow(
                    transition_matrix,
                    color_continuous_scale='viridis',
                    title=f"Matriz de Transición Temporal (ventana ≤{max_window} min)",
                    labels=dict(x="Lugar B (Reportado después)", y="Lugar A (Reportado primero)", color="Probabilidad")
                )
                fig_trans.update_layout(height=700)
                st.plotly_chart(fig_trans, use_container_width=True)
                st.caption("ℹ️ Mide transición temporal real. Valores altos sugieren posible efecto dominó.")
            else:
                st.warning("No se encontraron transiciones en la ventana temporal especificada.")
        else:
            st.warning("Se necesitan al menos 2 reportes para calcular transiciones.")

# ========== FOOTER ==========
st.markdown("---")
st.caption("⚡ Generado automáticamente por el pipeline SIMIEC. Proyecto de investigación. Todos los datos se calculan en tiempo real.")

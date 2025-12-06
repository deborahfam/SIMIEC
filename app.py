import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(page_title="SIMIEC Dashboard", layout="wide", page_icon="⚡")

st.title("⚡ SIMIEC: Monitorización de Incidencias Eléctricas")
st.markdown("""
**Sistema Integral de Monitorización de Incidencias Eléctricas basado en Crowdsourcing.**
*Visualización de datos extraídos mediante NLP de reportes ciudadanos.*
""")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('datos_georeferenciados.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        try:
            df_rel = pd.read_csv('relaciones_lugares.csv')
        except FileNotFoundError:
            df_rel = pd.DataFrame()
            
        return df, df_rel
    except FileNotFoundError:
        return None, None

df, df_rel = load_data()

if df is None:
    st.error("❌ No se encontraron los archivos CSV. Ejecuta los scripts de procesamiento primero.")
    st.stop()

st.sidebar.header("Filtros de Análisis")
min_date = df['date'].min()
max_date = df['date'].max()

date_range = st.sidebar.date_input(
    "Rango de Fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
df_filtered = df.loc[mask]

col1, col2, col3 = st.columns(3)
col1.metric("Total Reportes (Filtrados)", f"{len(df_filtered)}")
col2.metric("Lugares Únicos Detectados", f"{df_filtered['lugar_principal'].nunique()}")
df_filtered['hour'] = df_filtered['date'].dt.hour
hora_pico = df_filtered['hour'].mode()[0]
col3.metric("Hora Pico de Reportes", f"{hora_pico}:00 hrs")

tab1, tab2, tab3 = st.tabs(["📈 Dinámica Temporal", "🗺️ Distribución Espacial", "🕸️ Topología de Red"])

with tab1:
    st.subheader("Evolución Temporal de Incidencias")
    
    timeline = df_filtered.resample('h', on='date').count()['text'].reset_index()
    timeline.columns = ['Fecha', 'Reportes']
    
    fig_line = px.line(timeline, x='Fecha', y='Reportes', title="Frecuencia de Reportes por Hora")
    fig_line.update_traces(line_color='#FF4B4B')
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.subheader("Mapa de Calor: Patrones Semanales")
    df_filtered['day_name'] = df_filtered['date'].dt.day_name()
    dias_es = {'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles', 'Thursday': 'Jueves', 
               'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
    df_filtered['day_name'] = df_filtered['day_name'].map(dias_es)
    
    heatmap_data = df_filtered.groupby(['day_name', 'hour']).size().reset_index(name='counts')
    
    fig_heat = px.density_heatmap(
        heatmap_data, 
        x='hour', 
        y='day_name', 
        z='counts', 
        title="Intensidad de Reportes (Día vs Hora)",
        color_continuous_scale='Viridis',
        category_orders={"day_name": ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]}
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab2:
    col_map1, col_map2 = st.columns([2, 1])
    
    with col_map1:
        st.subheader("Lugares con Mayor Afectación")
        top_places = df_filtered['lugar_principal'].value_counts().reset_index()
        top_places.columns = ['Lugar', 'Reportes']
        
        fig_bar = px.bar(
            top_places.head(15), 
            x='Reportes', 
            y='Lugar', 
            orientation='h', 
            title="Top 15 Zonas Reportadas",
            color='Reportes',
            color_continuous_scale='Reds'
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_map2:
        st.write("### Datos Crudos")
        st.dataframe(top_places, height=400)

with tab3:
    st.subheader("Grafo de Co-ocurrencia (Conectividad de Circuitos)")
    st.markdown("""
    Este grafo muestra qué lugares se reportan **juntos** en el mismo mensaje. 
    *Una conexión fuerte indica que probablemente comparten circuito eléctrico.*
    """)
    
    if not df_rel.empty:
        min_weight = st.slider("Filtrar conexiones débiles (Peso mínimo)", 1, 20, 2)
        rel_filtered = df_rel[df_rel['Weight'] >= min_weight]
        
        import graphviz
        graph = graphviz.Digraph()
        graph.attr(rankdir='LR', size='10')
        
        for idx, row in rel_filtered.iterrows():
            penwidth = str(max(1, row['Weight'] / 2)) 
            graph.edge(row['Source'], row['Target'], label=str(row['Weight']), penwidth=penwidth)
            
        st.graphviz_chart(graph)
        
    else:
        st.warning("No hay suficientes datos de relaciones para generar el grafo.")

st.markdown("---")
st.caption("Generado automáticamente por el pipeline SIMIEC. Proyecto de investigación.")
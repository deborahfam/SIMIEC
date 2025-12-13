import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import re
from datetime import datetime

INPUT_FILE = 'results/datos_georeferenciados.csv'
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="white")

KEYWORDS_INICIO = ['se fue', 'quito', 'apagon', 'ñooo', 'pinga', 'coño', 'otra vez']
KEYWORDS_FIN = ['llego', 'vino', 'pusieron', 'gracias', 'al fin', 'llegó']

def cargar_datos():
    try:
        df = pd.read_csv(INPUT_FILE)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filtrar por último mes (5 de noviembre a 5 de diciembre)
        fecha_inicio = pd.Timestamp('2024-11-05')
        fecha_fin = pd.Timestamp('2024-12-05')
        df = df[(df['date'] >= fecha_inicio) & (df['date'] <= fecha_fin)].copy()
        
        df['time_block'] = df['date'].dt.floor('10T') 
        return df
    except FileNotFoundError:
        print("❌ Faltan datos.")
        return None

def generar_matriz_transicion(df):
    """
    Calcula: Dado que se reportó el Lugar A, ¿qué tan probable es que se reporte el Lugar B en el mismo bloque temporal?
    """
    top_lugares = df['lugar_principal'].value_counts().head(20).index
    df_top = df[df['lugar_principal'].isin(top_lugares)]
    
    pivot = pd.crosstab(df_top['time_block'], df_top['lugar_principal'])

    pivot = (pivot > 0).astype(int)
    
    cooccurrence = pivot.T.dot(pivot)
    
    diagonal = np.diag(cooccurrence)
    prob_matrix = cooccurrence / diagonal[:, None]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(prob_matrix, annot=False, cmap="rocket_r", linewidths=.5)
    plt.title("Matriz de Probabilidad Condicional: Si cae X, ¿cae Y?\nPeríodo: 5 Nov - 5 Dic 2024", fontsize=16)
    plt.xlabel("Lugar Afectado (Consecuencia)")
    plt.ylabel("Lugar Reportado Inicialmente (Causa Potencial)")
    plt.tight_layout()
    plt.savefig('fig5_matriz_domino.png')
    plt.close()

def mapa_semantico_mds(df):
    """
    Usa Multidimensional Scaling para dibujar un 'Mapa' sin tener coordenadas GPS,
    basado puramente en la 'distancia eléctrica' (co-ocurrencia).
    """
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
        
    plt.title("Mapa Semántico de la Red Eléctrica (Distancia basada en Fallos Simultáneos)\nPeríodo: 5 Nov - 5 Dic 2024", fontsize=14)
    plt.xlabel("Dimensión Latente 1")
    plt.ylabel("Dimensión Latente 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('fig6_mapa_mds.png')
    plt.close()

def analisis_sentimiento_fases(df):
    """
    Analiza qué palabras se usan más al 'Inicio' vs al 'Final' del apagón.
    """
    def clasificar_fase(texto):
        texto = texto.lower()
        if any(k in texto for k in KEYWORDS_INICIO): return 'Inicio (Ira/Reporte)'
        if any(k in texto for k in KEYWORDS_FIN): return 'Fin (Alivio/Aviso)'
        return 'Neutro'

    df['fase'] = df['text'].apply(clasificar_fase)
    
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
    
    plt.suptitle("Vocabulario Diferencial según la Fase del Apagón\nPeríodo: 5 Nov - 5 Dic 2024", fontsize=16)
    plt.tight_layout()
    plt.savefig('fig7_sentimiento_fases.png')
    plt.close()

def patron_dia_noche(df):
    """
    Contraste simple: ¿Quién reporta más de día vs de noche?
    """
    df['hora'] = df['date'].dt.hour
    df['periodo'] = df['hora'].apply(lambda x: 'Noche (19-06h)' if (x >= 19 or x < 6) else 'Día (07-18h)')
    
    top_lugares = df['lugar_principal'].value_counts().head(10).index
    df_top = df[df['lugar_principal'].isin(top_lugares)]
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_top, x='lugar_principal', hue='periodo', palette='coolwarm')
    plt.title("Distribución de Reportes Día vs Noche por Municipio\nPeríodo: 5 Nov - 5 Dic 2024", fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel("Cantidad de Reportes")
    plt.tight_layout()
    plt.savefig('fig8_dia_vs_noche.png')
    plt.close()

if __name__ == "__main__":
    df = cargar_datos()
    if df is not None:
        generar_matriz_transicion(df)
        mapa_semantico_mds(df)
        analisis_sentimiento_fases(df)
        patron_dia_noche(df)
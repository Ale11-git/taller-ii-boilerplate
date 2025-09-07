import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
import re


# Cargar tu CSV de reseñas
csv_path = r"C:\Users\MSI KATANA\IdeaProjects\taller-ii-boilerplate\data\raw\reviews.csv"
df = pd.read_csv(csv_path)

from datetime import datetime, timedelta


# Función para convertir "x days ago", "x hours ago", etc.
from datetime import datetime, timedelta
import pandas as pd

# Normalizar nombres de columnas por si vienen con espacios raros
df.columns = df.columns.str.strip()

# (Opcional) Ver qué columnas hay
# print(df.columns.tolist())

# 2) Función robusta para fechas ("9 hours ago", "2 days ago", o fechas tipo "August 3, 2025")
def parse_relative_date(s: str):
    if not isinstance(s, str) or not s.strip() or s == "No date":
        return pd.NaT

    s_low = s.lower().strip()

    # Caso relativo: "X [minute|hour|day|month|year]s ago"
    m = re.match(r"^(\d+)\s+(minute|hour|day|month|year)s?\s+ago$", s_low)
    if m:
        num = int(m.group(1))
        unit = m.group(2)
        if unit == "minute":
            delta = timedelta(minutes=num)
        elif unit == "hour":
            delta = timedelta(hours=num)
        elif unit == "day":
            delta = timedelta(days=num)
        elif unit == "month":
            delta = timedelta(days=30*num)   # aprox
        else:  # year
            delta = timedelta(days=365*num)  # aprox
        return pd.Timestamp.now() - delta

    # Caso absoluto: intentar parseo directo (e.g., "August 6, 2025")
    return pd.to_datetime(s, errors="coerce")

# 3) Crear columnas de fecha
if 'Date' not in df.columns:
    raise KeyError(f'No encuentro la columna "Date". Columnas disponibles: {df.columns.tolist()}')

df['Date_parsed']   = df['Date'].apply(parse_relative_date)
df['Date_formatted'] = df['Date_parsed'].dt.strftime("%d/%m/%y")  # puede quedar NaN si Date_parsed es NaT

# 4) Convertir rating "Rated 4 out of 5 stars" -> 4.0
df['Rating_num'] = (
    df['Rating']
    .str.extract(r'Rated\s+(\d+)\s+out of\s+5', expand=False)
    .astype('float')
)

# 5) Ver resultado
df = df.drop(columns=['Date', 'Date_parsed'])
print(df.head(10))
df.to_csv(
    r"C:\Users\MSI KATANA\IdeaProjects\taller-ii-boilerplate\data\preprocessed\reviews_clean.csv",
    index=False,
    encoding="utf-8-sig"
)

# Reemplazar "No text" por NaN y quitar espacios en blanco
df['Review_text'] = df['Review_text'].replace("No text", pd.NA).str.strip()

csv_path = r"C:\Users\MSI KATANA\IdeaProjects\taller-ii-boilerplate\data\preprocessed\reviews_clean.csv"
df = pd.read_csv(csv_path)

# --- LIMPIEZA ---

# 1. Reemplazar "No text" por NaN y quitar espacios extra
df['Review_text'] = df['Review_text'].replace("No text", pd.NA).str.strip()

# 2. Eliminar filas con Review_text vacío o NaN
df = df.dropna(subset=['Review_text'])

# 3. Eliminar reseñas duplicadas (basadas en el texto)
df = df.drop_duplicates(subset=['Review_text'], keep='first')

# --- GUARDAR (sobrescribir el archivo existente) ---
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"✅ CSV sobrescrito en: {csv_path} con {len(df)} reseñas únicas y limpias")


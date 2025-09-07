import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Encabezados (simula navegador normal)
headers = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/140.0.0.0 Safari/537.36")
}

all_reviews = defaultdict(list)

page = 1
while True:
    url = f"https://www.trustpilot.com/review/www.airbnb.com?page={page}"
    print(f"📥 Descargando página {page}...")

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    reviews = soup.find_all("article")

    if not reviews:   # si no hay más reseñas, cortamos el bucle
        break

    for review in reviews:
        # Texto
        text_tag = review.find("p")
        review_text = text_tag.text.strip() if text_tag else "No text"

        # Rating
        rating_tag = review.find("img")
        rating = rating_tag.get("alt") if rating_tag else "No rating"

        # Fecha
        date_tag = review.find("time")
        date = date_tag.text.strip() if date_tag else "No date"

        all_reviews['Review_text'].append(review_text)
        all_reviews['Rating'].append(rating)
        all_reviews['Date'].append(date)

    page += 1  # pasamos a la siguiente página

# Guardar en DataFrame
df = pd.DataFrame(all_reviews)

# 📂 Ruta donde guardar el CSV
csv_path = Path(r"C:\Users\MSI KATANA\IdeaProjects\taller-ii-boilerplate\data\raw\reviews.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"✅ Se guardaron {len(df)} reseñas en: {csv_path}")
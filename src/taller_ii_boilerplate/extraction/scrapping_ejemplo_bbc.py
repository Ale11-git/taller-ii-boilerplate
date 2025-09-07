import requests
from bs4 import BeautifulSoup

url = "https://www.bbc.com/mundo"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

titulos = soup.find_all("h3")

for titulo in titulos:
    print(titulo.get_text())
    print("----------------")

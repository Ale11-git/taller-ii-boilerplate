import requests

#TODO extraer datos desde https://dummyjson.com/products
response = requests.get('https://fakestoreapi.com/products')

if response.status_code == 200:
    data = response.json()
    print(data)
    product = data[0]
    print("Nombre del Producto:", product['title'])
else:
    print('Error:', response.status_code)


import requests

response = requests.get('https://dummyjson.com/products')

if response.status_code == 200:
    data = response.json()
    products = data['products']   # lista de productos

    for product in products:
        print("ID:", product.get('id', 'Sin ID'))
        print("Nombre:", product.get('title', 'Sin nombre'))
        print("Precio:", product.get('price', 'Sin precio'))
        print("Marca:", product.get('brand', 'Sin marca'))   # <- cambio acá
        print("Categoría:", product.get('category', 'Sin categoría'))

        reviews = product.get("reviews", [])
        if reviews:
            print("Reseñas:")

            for review in reviews:
                print("-- Usuario: ", review.get('reviewerName'))
                print("-- Valoracion: ", review.get('rating'))
                print("-- Commentario: ", review.get('comment'))
        else:
            print("Reseñas: (sin reseñas)")

        print("-" * 40)

else:
    print("Error:", response.status_code)
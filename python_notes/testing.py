import pandas as pd

data = {
    "Tarih": ["2025-09-01", "2025-09-02", "2025-09-03", None, "2025-09-05"],
    "Ürün": ["Laptop", "Mouse", None, "Monitör", "Kulaklık"],
    "Kategori": ["Elektronik", "Aksesuar", "Aksesuar", "Elektronik", None],
    "Adet": [5, None, 7, 3, 15],
    "Fiyat": [12000, 150, 300, 450, 200],
}


df = pd.DataFrame(data)

f = df.query("Fiyat > 600")
print(f)
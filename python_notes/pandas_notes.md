# Pandas

## 📈 1- Pandas Series (Tek boyutlu veri (1D))

- Pandas’ta Series, tek boyutlu, etiketlenmiş bir veri yapısıdır.
- Hem veri hem de index (etiket) içerir.
- Python’daki list veya NumPy array gibi düşünülebilir, ama etiketleme özelliği vardır.

**Özellikleri:**
- Tek boyutludur.
- Farklı veri tiplerini içerebilir: int, float, string, bool vb.
- Index ile her elemana isim verilebilir.

### 1) Series oluşturma

#### a) Listeden series
``` python
import pandas as pd

veriler = [10,20,30,40]
result = pd.Series(veriler)

print(result)

# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64
```

#### b) Index belirleyerek series
``` python
import pandan as pd

result = pd.Series(veriler,index=['a','b','c','d'])
print(result)

# a    10
# b    20
# c    30
# d    40
# dtype: int64
```
- Artık index isimleri 0,1,2,3 yerine 'a','b','c','d'.

#### c) Sözlükten series
``` python
sozluk = {"Elma":7,"Kayısı":8,"Kiraz":6}

result = pd.Series(sozluk)
print(result)

# elma      5
# armut    10
# kiraz     7
# dtype: int64
```
- Anahtarlar index, değerler veri olarak kullanılır.

#### d) Tek değer ile series
``` python
result = pd.Series(4,"a","b","c")
print(result)

# a    7
# b    7
# c    7
# dtype: int64
```
### 2) Series'e erişim

**Index ile irişim**
``` python
print(result["a"]) # 4
```
**Konum ile irişim**
``` python
print(result[0]) # 4
```
**Dilimleme (slice)**
``` python
print(result[0:2])
```

### 3) Series üzerinde işlemler
``` python
a = pd.Series([1,2,3,4])
b = pd.Series([20,25,30,35])

print(s1 + s2)  # Toplama 
print(s1 * 2)   # Çarpma
print(s1 > 2)   # Karşılaştırma
```


### 4) Faydalı fonksiyonlar

#### Temel fonksiyonlar
| Fonksiyon          | Açıklama                           | Örnek Kullanım         |
|--------------------|-----------------------------------|------------------------|
| `head(n)`          | İlk `n` elemanı döner              | `s.head(3)`            |
| `tail(n)`          | Son `n` elemanı döner              | `s.tail(2)`            |
| `sum()`            | Elemanların toplamı                | `s.sum()`              |
| `mean()`           | Ortalama                          | `s.mean()`             |
| `max()`            | En büyük değer                     | `s.max()`              |
| `min()`            | En küçük değer                     | `s.min()`              |
| `describe()`       | İstatistiksel özet                 | `s.describe()`         |

#### Eksik Veri (NaN) Kontrol Fonksiyonları
| Fonksiyon     | Açıklama                                        | Örnek Kullanım      |
|---------------|------------------------------------------------|---------------------|
| `isnull()`    | NaN olan değerleri `True`, diğerlerini `False` döner | `s.isnull()`       |
| `notnull()`   | NaN olmayanları `True`, NaN olanları `False` döner | `s.notnull()`      |

**Filtreleme Örnekleri:**
```python
s[s.notnull()]   # Sadece boş olmayan değerler
s[s.isnull()]    # Sadece boş değerler
```

## 📊 2- Pandas Dataframe (İki boyutlu veri (2D))

- pandas kütüphanesindeki en temel veri yapılarından biridir.
- Excel tablosuna veya SQL tablosuna çok benzer.
- Satırlar (index) ve sütunlar (columns) içerir.
- Farklı veri tiplerini (sayı, string, tarih vs.) aynı tabloda tutabilir.
- Kısaca: DataFrame = 2 boyutlu etiketlenmiş tablo yapısıdır.

### 1) Sözlükten Dataframe
``` python
import pandas as pd

data = {
    "Ad":["Ali", "Ayşe", "Mehmet", "Zeynep"],
    "Yaş": [25, 30, 22, 28],
    "Şehir": ["İstanbul", "Ankara", "İzmir", "Bursa"]
}

result = pd.DataFrame(data)
print(result)

#        Ad  Yaş    Şehir
# 0     Ali   25  İstanbul
# 1    Ayşe   30    Ankara
# 2  Mehmet   22     İzmir
# 3  Zeynep   28     Bursa
```
### 2) Liste Listesinden DataFrame
``` python
data = [
    ["Ali", 25, "İstanbul"],
    ["Ayşe", 30, "Ankara"],
    ["Mehmet", 22, "İzmir"],
    ["Zeynep", 28, "Bursa"]
]

result = pd.DataFrame(data, columns=["Ad", "Yaş", "Şehir"])
print(result)

#        Ad  Yaş    Şehir
# 0     Ali   25  İstanbul
# 1    Ayşe   30    Ankara
# 2  Mehmet   22     İzmir
# 3  Zeynep   28     Bursa
```
### 3) CSV Dosyasından DataFrame
``` python
result=pd.read_csv("veriler.csv")
print(result.head()) # ilk 5 satırı yazdırır.
```

### 4) DataFrame Üzerinde Temel İşlemler
``` python
print(df.head())        # İlk 5 satır
print(df.tail())        # Son 5 satır
print(df.info())        # Veri tipi bilgileri
print(df.describe())    # Sayısal sütunların istatistikleri

print(df["Ad"])         # Tek bir sütun seçme
print(df[["Ad","Şehir"]]) # Birden fazla sütun seçme

print(df.iloc[0])       # İlk satırı index ile seçme
print(df.loc[2])        # 2. indexteki satırı seçme
```

### 5) Veri Filtreleme
``` python
# Yaşı 25’ten büyük olanları seç
print(result[result["yaş"] > 25])

# İstanbul’da yaşayanları seç
print(result[result["Şehir"]=="İstanbul"])
```


## 📊 3- Pandas Veri Düzenleme Metodları

### 1. `rename()` – Sütun veya indeks adlarını değiştirme

```python
import pandas as pd

df = pd.DataFrame({
    "isim": ["Ali", "Ayşe", "Mehmet"],
    "yas": [25, 30, 35]
})

# Sütun adlarını değiştirelim
df_renamed = df.rename(columns={"isim": "ad", "yas": "yaş"})
print(df_renamed)
```

---

### 2. `drop()` – Satır veya sütun silme

```python
# Sütun silme
df_drop_col = df.drop(columns=["yas"])

# Satır silme (index numarasına göre)
df_drop_row = df.drop(index=[1])  
print(df_drop_col)
print(df_drop_row)
```

---

### 3. `sort_values()` – Değerleri sıralama

```python
# Yaşa göre küçükten büyüğe sıralama
df_sorted = df.sort_values(by="yas")

# İsme göre ters sıralama
df_sorted_name = df.sort_values(by="isim", ascending=False)
print(df_sorted)
print(df_sorted_name)
```

---

### 4. `set_index()` ve `reset_index()` – İndeksi ayarlama / sıfırlama

```python
# "isim" sütununu index yapalım
df_indexed = df.set_index("isim")

# İndeksi sıfırlama
df_reset = df_indexed.reset_index()
print(df_indexed)
print(df_reset)
```

---

### 5. `fillna()` – Eksik değerleri doldurma

```python
df_nan = pd.DataFrame({
    "isim": ["Ali", "Ayşe", "Mehmet"],
    "yas": [25, None, 35]
})

# Eksik değerleri ortalama ile dolduralım
df_filled = df_nan.fillna(df_nan["yas"].mean())
print(df_filled)
```

---

### 6. `dropna()` – Eksik değerleri silme

```python
# Eksik değer içeren satırları sil
df_dropna = df_nan.dropna()
print(df_dropna)
```

---

### 7. `replace()` – Değer değiştirme

```python
# "Ali" ismini "Ahmet" ile değiştir
df_replaced = df.replace({"Ali": "Ahmet"})
print(df_replaced)
```

---

### 8. `apply()` – Fonksiyon uygulama

```python
# Yaş değerlerini 2 ile çarp
df_apply = df.assign(yas = df["yas"].apply(lambda x: x * 2))
print(df_apply)
```

---

### 9. `astype()` – Veri tipini değiştirme

```python
# Yaş sütununu float tipine dönüştürme
df["yas"] = df["yas"].astype(float)
print(df.dtypes)
```

---

### 10. `query()` – Koşula göre filtreleme

```python
# Yaşı 30'dan büyük olanları seç
df_query = df.query("yas > 30")
print(df_query)
```

---

### 11. `assign()` – Yeni sütun ekleme

```python
# Yeni sütun ekleyelim (doğum yılı)
df_new = df.assign(dogum_yili = 2025 - df["yas"])
print(df_new)
```

---

### 12. `melt()` – Geniş formattan uzun formata dönüştürme

```python
df_melt = pd.melt(df, id_vars=["isim"], value_vars=["yas"])
print(df_melt)
```

---

### 13. `pivot_table()` – Veri özetleme (pivot tablo)

```python
df_sales = pd.DataFrame({
    "mağaza": ["A", "A", "B", "B"],
    "ürün": ["Elma", "Armut", "Elma", "Armut"],
    "satış": [10, 15, 5, 20]
})

df_pivot = df_sales.pivot_table(values="satış", index="mağaza", columns="ürün", aggfunc="sum")
print(df_pivot)
```

---

## 📊 4- Pandas ile Veri Korelasyonu

Veri korelasyonu, iki veya daha fazla değişken arasındaki ilişkinin **güçlü veya zayıf** olup olmadığını ölçer.

* Pozitif korelasyon: Bir değişken artarken diğer değişken de artar.
* Negatif korelasyon: Bir değişken artarken diğer değişken azalır.
* Korelasyon katsayısı: `-1` ile `1` arasında değer alır.

  * `1` → tam pozitif korelasyon
  * `-1` → tam negatif korelasyon
  * `0` → ilişki yok

---

### 1️⃣ Pandas ile Korelasyon Hesaplama

Pandas’ta `.corr()` fonksiyonu kullanılır:

```python
import pandas as pd

# Örnek DataFrame
data = {
    "yas": [20, 25, 30, 35, 40],
    "maas": [2000, 2500, 3000, 3500, 4000],
    "tecrube": [1, 3, 5, 7, 10]
}

df = pd.DataFrame(data)

# Korelasyon matrisi
correlation_matrix = df.corr()
print(correlation_matrix)
```

---

### 2️⃣ Çıktı Örneği

|         | yas  | maas | tecrube |
| ------- | ---- | ---- | ------- |
| yas     | 1.0  | 1.0  | 0.99    |
| maas    | 1.0  | 1.0  | 0.99    |
| tecrube | 0.99 | 0.99 | 1.0     |

* `yas` ve `maas` arasında **çok güçlü pozitif korelasyon** var (1.0).
* `tecrube` de diğer değişkenlerle yüksek pozitif korelasyona sahip.

---

### 3️⃣ Tek Bir Korelasyon Değeri

Belirli iki sütun arasındaki korelasyonu şöyle alabiliriz:

```python
yas_maas_corr = df["yas"].corr(df["maas"])
print("Yaş ve Maaş Korelasyonu:", yas_maas_corr)
```

**Çıktı:**

```
Yaş ve Maaş Korelasyonu: 1.0
```

---

💡 **Not:**

* Korelasyon **nedensellik anlamına gelmez**. Yani iki değişken yüksek korelasyona sahip olsa bile, biri diğerine sebep olmaz.
* Korelasyon sayısal sütunlar için uygundur. Kategorik veriler için farklı yöntemler (örn. `chi-square`) kullanılır.

---

## 5- Pandas İle Veri Birleştirme

### 1) concat()
- DataFrame veya Series’leri alt alta (row-wise) veya yan yana (column-wise) birleştirmek için kullanılır
```python
import pandas as pd

# Örnek DataFrame'ler
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})

# Alt alta birleştirme
print(pd.concat([df1, df2]))

# Yan yana birleştirme
print(pd.concat([df1, df2], axis=1))

çıktı:

Alt alta:
   A  B
0  1  3
1  2  4
0  5  7
1  6  8

Yan yana:
   A  B  A  B
0  1  3  5  7
1  2  4  6  8
```
---
### 2) merge()
- SQL’deki JOIN işlemlerine benzer. Ortak kolon(lar) üzerinden iki DataFrame’i birleştirir.
```python
import pandas as pd

df1 = pd.DataFrame({
    "id": [1, 2, 3],
    "isim": ["Ali", "Ayşe", "Mehmet"]
})

df2 = pd.DataFrame({
    "id": [1, 2, 4],
    "yas": [23, 25, 30]
})

# Ortak sütun 'id' üzerinden INNER JOIN
print(df1.merge(df2, on="id", how="inner"))

# LEFT JOIN
print(df1.merge(df2, on="id", how="left"))

# OUTER JOIN
print(df1.merge(df2, on="id", how="outer"))

çıktı:

Inner Join:
   id  isim  yas
0   1   Ali   23
1   2  Ayşe   25

Left Join:
   id   isim   yas
0   1    Ali  23.0
1   2   Ayşe  25.0
2   3  Mehmet   NaN

Outer Join:
   id   isim   yas
0   1    Ali  23.0
1   2   Ayşe  25.0
2   3  Mehmet   NaN
3   4    NaN  30.0
```
---
### 3) join()
- Index üzerinden veya belirli kolonlara göre DataFrame’leri birleştirir.
```python
df1 = pd.DataFrame({"isim": ["Ali", "Ayşe", "Mehmet"]}, index=[1, 2, 3])
df2 = pd.DataFrame({"yas": [23, 25, 30]}, index=[1, 2, 4])

print(df1.join(df2, how="inner"))   # ortak index'ler
print(df1.join(df2, how="outer"))   # tüm index'ler

çıktı:

Inner Join:
     isim  yas
1     Ali   23
2    Ayşe   25

Outer Join:
     isim   yas
1     Ali  23.0
2    Ayşe  25.0
3  Mehmet   NaN
4     NaN  30.0
```
## Pandas String Metotları
```python
data = pd.DataFrame({
    "isim": [" ali ", "AYŞE", "mehmet", "Ahmet"],
    "sehir": ["istanbul", "ANKARA", "izmir", "Bursa"]
})

lower = data["isim"].str.lower()
upper = data["isim"].str.upper()
strip = data["isim"].str.strip()
replace_a = data["sehir"].str.replace("a", "e", case=False)
contains_an = data["sehir"].str.contains("an", case=False) # 'an' geçenleri True yapar
length = data["isim"].str.len() # her bir stringin karakter sayısını verir
title = data["isim"].str.title() # kelimelerin baş harfini büyütür
first_letter = data["sehir"].str[0] # her bir stringin ilk harfini döndürür
```
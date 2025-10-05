# Python Hızlı Başvuru / Cheat Sheet

Bu cheat sheet, Python'da sık kullanılan fonksiyonları, metodları ve örneklerini hızlıca görmeniz için hazırlandı.

---

## 1️⃣ String (Metin) Metodları

| Metod | Açıklama | Örnek |
|-------|----------|-------|
| `join()` | Listeyi stringe çevirir | `"-".join(['a','b'])` → `"a-b"` |
| `split()` | String’i parçalara ayırır | `"a-b".split("-")` → `['a','b']` |
| `replace()` | Metni değiştirir | `"merhaba dünya".replace("dünya","Python")` → `"merhaba Python"` |
| `strip()` | Baş ve sondaki boşlukları temizler | `"  hello  ".strip()` → `"hello"` |
| `lstrip()` | Baştaki boşlukları siler | `"  hello".lstrip()` → `"hello"` |
| `rstrip()` | Sondaki boşlukları siler | `"hello  ".rstrip()` → `"hello"` |
| `lower()` | Küçük harfe çevirir | `"HELLO".lower()` → `"hello"` |
| `upper()` | Büyük harfe çevirir | `"hello".upper()` → `"HELLO"` |
| `capitalize()` | İlk harfi büyük yapar | `"hello".capitalize()` → `"Hello"` |
| `title()` | Her kelimenin baş harfi büyük | `"hello world".title()` → `"Hello World"` |
| `startswith()` | Başlangıç kontrolü | `"hello".startswith("he")` → `True` |
| `endswith()` | Bitiş kontrolü | `"hello".endswith("lo")` → `True` |
| `find()` | Aranan karakterin indexi | `"hello".find("e")` → `1` |
| `count()` | Karakter sayısı | `"hello".count("l")` → `2` |

---

## 2️⃣ List (Liste) Metodları

| Metod | Açıklama | Örnek |
|-------|----------|-------|
| `append()` | Eleman ekler | `[1,2].append(3)` → `[1,2,3]` |
| `extend()` | Listeyi başka liste ile birleştirir | `[1].extend([2,3])` → `[1,2,3]` |
| `insert(index, value)` | Belirli yere ekler | `[1,3].insert(1,2)` → `[1,2,3]` |
| `pop()` | Son elemanı çıkarır | `[1,2,3].pop()` → `3` |
| `remove(value)` | Belirli değeri siler | `[1,2,3].remove(2)` → `[1,3]` |
| `clear()` | Listeyi temizler | `[1,2].clear()` → `[]` |
| `index(value)` | Elemanın indexini verir | `[1,2,3].index(2)` → `1` |
| `count(value)` | Eleman sayısını verir | `[1,2,2,3].count(2)` → `2` |
| `sort()` | Sıralar | `[3,1,2].sort()` → `[1,2,3]`  Listenin kendisini sıralanmış olarak değiştirir. | 
| `reverse()` | Listeyi tersine çevirir | `[1,2,3].reverse()` → `[3,2,1]`  Listenin kendisini tersine çevirir.|

---

## 3️⃣ Set (Küme) Metodları

| Metod | Açıklama | Örnek |
|-------|----------|-------|
| `add()` | Eleman ekler | `{1,2}.add(3)` → `{1,2,3}` |
| `remove()` | Eleman siler | `{1,2,3}.remove(2)` → `{1,3}` |
| `discard()` | Eleman varsa siler | `{1,2}.discard(3)` → `{1,2}` |
| `union()` | Birleşim | `{1,2}.union({2,3})` → `{1,2,3}` |
| `intersection()` | Kesişim | `{1,2}.intersection({2,3})` → `{2}` |
| `difference()` | Fark | `{1,2,3}.difference({2,3})` → `{1}` |
| `clear()` | Tüm elemanları siler | `{1,2}.clear()` → `set()` |

---

## 4️⃣ Dictionary (Sözlük) Metodları

| Metod | Açıklama | Örnek |
|-------|----------|-------|
| `keys()` | Anahtarları döndürür | `{'a':1,'b':2}.keys()` → `dict_keys(['a','b'])` |
| `values()` | Değerleri döndürür | `{'a':1,'b':2}.values()` → `dict_values([1,2])` |
| `items()` | (anahtar, değer) ikilileri | `{'a':1}.items()` → `dict_items([('a',1)])` |
| `get(key, default)` | Anahtar değeri alır, yoksa default | `d.get('c',0)` → `0` |
| `pop(key)` | Anahtar ve değerini çıkarır | `d.pop('a')` → `1` |
| `update(dict)` | Sözlüğü günceller | `d.update({'c':3})` → `{'a':1,'b':2,'c':3}` |
| `clear()` | Sözlüğü temizler | `d.clear()` → `{}` |

---

## 5️⃣ Built-in Fonksiyonlar (Yerleşik)

| Fonksiyon | Açıklama | Örnek |
|-----------|----------|-------|
| `len()` | Uzunluk | `len([1,2,3])` → `3` |
| `type()` | Tip öğrenme | `type(5)` → `<class 'int'>` |
| `int()` | Tamsayıya çevirir | `int("5")` → `5` |
| `float()` | Ondalıklı sayıya çevirir | `float("5.5")` → `5.5` |
| `str()` | Stringe çevirir | `str(5)` → `"5"` |
| `range()` | Sayı dizisi oluşturur | `list(range(3))` → `[0,1,2]` |
| `enumerate()` | İndex ile birlikte döndürür | `list(enumerate(['a','b']))` → `[(0,'a'),(1,'b')]` |
| `zip()` | Listeleri eşleştirir | `list(zip([1,2],[3,4]))` → `[(1,3),(2,4)]` |
| `map(func, iterable)` | Fonksiyonu uygular | `list(map(str,[1,2]))` → `['1','2']` |
| `filter(func, iterable)` | Şarta uyanları alır | `list(filter(lambda x:x>1,[0,1,2]))` → `[2]` |
| `sorted()` | Sıralama | `sorted([3,1,2])` → `[1,2,3]` |
| `max()` | Maksimum değer | `max([1,5,3])` → `5` |
| `min()` | Minimum değer | `min([1,5,3])` → `1` |
| `sum()` | Toplam | `sum([1,2,3])` → `6` |
| `abs()` | Mutlak değer | `abs(-5)` → `5` |
| `round()` | Yuvarlama | `round(3.1415,2)` → `3.14` |
| `all()` | Tüm eleman True mu | `all([True,False])` → `False` |
| `any()` | En az bir eleman True mu | `any([True,False])` → `True` |

---

**Not:** Bu cheat sheet, Python’da günlük olarak en sık kullandığınız fonksiyonları kapsar.

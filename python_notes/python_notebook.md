### Python notebook

## join()
- listeyi veya tuple’ı tek bir string hâline getirmek için kullanılır.

Örnek:
``` python
- liste = ['A', 'B', 'C']
- print(''.join(liste))    # "ABC"
- print('-'.join(liste))   # "A-B-C"
```
## set()
- Python’da tekrarsız ve sırasız öğelerden oluşan bir küme oluşturur.
- Tekrarlayan öğeleri otomatik kaldırır.

Örnek:
``` python
- my_list = [1, 2, 2, 3]
- unique = set(my_list)
- print(unique)  # Çıktı: {1, 2, 3}
```
## split()
- split() → String’i parçalara ayırır.

Örnek:
``` python
s = "a-b-c"
lst = s.split('-')  # ['a', 'b', 'c']
```
## namedtuple
- tuple gibi davranır ama elemanlara isimle erişebilirsin.
- collections modülünden gelir.
- Daha okunabilir ve düzenli veri saklamaya yarar.

Örnek:
``` python
from collections import namedtuple

# Sınıf tanımı
Student = namedtuple("Student", ["ID", "Name", "Marks"])

# Nesne oluşturma
s1 = Student(1, "Alice", 95)

# Erişim
print(s1.ID)     # 1
print(s1.Name)   # Alice
print(s1.Marks)  # 95
```
Özellikler:
- Alanlara isimle erişim: s1.Name
- Normal tuple gibi indeksle de erişim: s1[1] → "Alice"
- Immutable (değiştirilemez), tıpkı tuple gibi.

## eval()
- Tanım: Python’da eval() verilen string ifadeyi Python koduymuş gibi çalıştırır.
- Kullanım Alanı: Matematiksel ifadeleri veya dinamik kod parçalarını string olarak çalıştırmak için.

Örnek:
``` python
expr = "3 * (2 + 5)"
print(eval(expr))   # 21
```
## itertools.product()
- Amaç: Birden fazla liste veya iterable’ın tüm kombinasyonlarını üretmek.
- Her kombinasyon bir tuple olarak döner.
- *lists şeklinde argüman vererek, istediğin kadar listeyi tek seferde kullanabilirsin.

Örnek:
``` python
from itertools import product

A = [1, 2]
B = ['a', 'b']

for p in product(A, B):
    print(p)

çıktı:
(1, 'a')
(1, 'b')
(2, 'a')
(2, 'b')
```
## rpartition(' ')
- Bir stringi sağdan (en sondan), verilen ayırıcıya göre 3 parçaya böler:

- 1. Ayırıcının solundaki kısım

- 2. Ayırıcı (mesela ' ')

- 3. Ayırıcının sağındaki kısım

Örnek:
``` python
metin = "APPLE JUICE 10"
item, space, price = metin.rpartition(" ")
print(item)   # APPLE JUICE
print(space)  # ' '  (boşluk)
print(price)  # 10

```
- Eğer boşluğu kullanmayacaksan _ koyabilirsin:
``` python
item, _,price = metin.rpartition(" ")
```

## .get(key, default)
- Sözlüklerde (dict) kullanılan bir fonksiyondur.

- key → değerini almak istediğin anahtar

- default → eğer o anahtar sözlükte yoksa, dönecek olan yedek değer (varsayılan olarak None)

🔹 Normal erişim ile farkı :

#### - Eğer normal d[key] ile erişirsen:

- Anahtar varsa → değeri döner

- Anahtar yoksa → hata (KeyError) verir

#### - Ama d.get(key, default) ile:

- Anahtar varsa → değeri döner

- Anahtar yoksa → hata yerine verdiğin default döner (eğer yazmazsan None).

Örnek:
``` python
# Bir sözlük oluşturalım
d = {"elma": 3, "armut": 5}

# 1. Var olan bir anahtarı almak
print(d.get("elma", 0))   # 3

# 2. Olmayan bir anahtarı almak (default = 0)
print(d.get("muz", 0))    # 0

# 3. Olmayan bir anahtarı almak (default = None çünkü yazmadık)
print(d.get("kiraz"))     # None
```
Örnek kullanım:
``` python
sayilar = {}

for s in [1,2,2,3,1]:
    sayilar[s]= sayilar.get(s, 0) + 1

print(sayilar)
# {1: 2, 2: 2, 3: 1}
```
- Burada sayilar.get(s, 0) sayesinde anahtar yoksa 0’dan başlıyor, varsa üzerine ekleniyor.

## Defaultdict

**Ne işe yarar?**

- Normal dict gibi çalışır.
- Farkı: olmayan bir anahtara erişildiğinde otomatik olarak varsayılan değer üretir.
- defaultdict(list) → varsayılan değer boş liste []
- defaultdict(int) → varsayılan değer 0
- Böylece KeyError hatası almazsın.

Örnek:
``` python
from collections import defaultdict

# Normal dict
d = {}
# d["a"].append(1)  # KeyError! çünkü "a" yok

# defaultdict
A = defaultdict(list)
A["a"].append(1)   # otomatik olarak A["a"] = [] oluşturur, sonra 1 ekler
A["b"].append(2)
print(A)  # defaultdict(<class 'list'>, {'a': [1], 'b': [2]})
```
## Counter
- Counter, Python’un collections modülünde bulunan bir sınıftır.
- Aslında dict’in alt sınıfıdır, yani anahtar–değer çiftleri saklar.
- Ama özellikle eleman sayımı ve frekans analizi için tasarlanmıştır.
- Eksik bir anahtara erişmeye çalışırsan 0 döndürür, KeyError vermez.
``` python
from collections import Counter

words = ["apple", "banana", "apple", "orange", "banana", "apple"]
c = Counter(words)

print("Farklı kelime sayısı:", len(c))
print("Kelime frekansları:", *c.values())
print("En sık kelime:", c.most_common(1)[0][0])
```
**çıktı :** 
```python
Farklı kelime sayısı: 3
Kelime frekansları: 3 2 1
En sık kelime: apple
```

## Groupby
-groupby, Python’un itertools modülünde bulunan bir fonksiyondur.
- Amaç: Ardışık tekrarlayan öğeleri gruplayarak işlem yapmaktır.
- Önemli: groupby sadece ardışık aynı değerleri gruplar. Eğer tüm listeyi gruplayıp benzerleri birleştirmek istiyorsan önce listeyi sorted() ile sıralamalısın.
```python
from itertools import groupby

data = [1, 1, 2, 2, 2, 3, 1, 1]

for key, group in groupby(data):
    print(key, list(group))
```
**çıktı :**
```python
1 [1, 1]
2 [2, 2, 2]
3 [3]
1 [1, 1]
```

## Deque
- deque, Python’un collections modülünde bulunan bir veri yapısıdır.
- Açılımı: double-ended queue (çift uçlu kuyruk).

**Özellikleri:**

- Hem başından hem de sonundan hızlı bir şekilde ekleme ve çıkarma yapabilirsin (O(1) zaman).

- Normal list’e göre baştan ekleme/çıkarma çok daha hızlıdır.

```python
from collections import deque

d = deque([1, 2, 3])
print(d)  # deque([1, 2, 3])

```
| Metot              | Açıklama                        | Örnek                                    |
|-------------------|---------------------------------|-----------------------------------------|
| `append(x)`        | Sona ekler                      | `d.append(4)` → `deque([1,2,3,4])`     |
| `appendleft(x)`    | Başa ekler                      | `d.appendleft(0)` → `deque([0,1,2,3,4])`|
| `pop()`            | Sondan çıkarır                  | `d.pop()` → 4, `deque([0,1,2,3])`      |
| `popleft()`        | Baştan çıkarır                  | `d.popleft()` → 0, `deque([1,2,3])`    |
| `extend(iterable)` | Sona birden fazla ekler         | `d.extend([4,5])` → `deque([1,2,3,4,5])`|
| `extendleft(iterable)` | Başa birden fazla ekler (ters sırayla) | `d.extendleft([0,-1])` → `deque([-1,0,1,2,3,4,5])` |

## getattr
- getattr() bir Python yerleşik fonksiyonudur ve bir objenin özellik veya metoduna isimle erişmek için kullanılır.
```python
getattr(object, "attribute_name")
```
- object → erişmek istediğin obje
- "attribute_name" → string olarak attribute/metod ismi
- Döndürür → objenin o attribute/metodu (çağırılabilir bir fonksiyon olabilir)

**Örnek :**
```python
class MyClass:
    def greet(self, name):
        print(f"Hello {name}")

obj = MyClass()

# Metodu almak
meth = getattr(obj, "greet")  # obj.greet ile aynı
meth("Alice")                  # Hello Alice
```
**Direkt Çağırmak**
- Metodu alıp bir değişkene atamak zorunda değilsin, hemen çağırabilirsin :
```python
getattr(obj, "greet")("Bob")  # Hello Bob
```
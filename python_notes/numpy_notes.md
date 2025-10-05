## 1- Array oluşturma 
``` python
np.array([1, 2, 3])            # Python listesinden array
np.zeros((2,3))                # 2x3 sıfır matrisi
np.ones((3,3))                 # 3x3 bir matrisi
np.eye(4)                      # 4x4 birim matris
np.arange(0,10,2)              # [0, 2, 4, 6, 8]
np.linspace(0,1,5)             # [0, 0.25, 0.5, 0.75, 1]
```

### 1) Birim martisi(np.eye):

- Kare matris olup köşegen üzerindeki elemanlar 1, diğerleri 0’dır.
#### Örnek: 
``` python
np.eye(3)

[[1. 0. 0.]
[0. 1. 0.]
[0. 0. 1.]]
```

### 2) Köşegen Matrisi (np.diag)

- Sadece köşegen elemanları dolu olan matris.
#### Örnek: 
``` python
np.diag([1,2,3])

[[1 0 0]
[0 2 0]
[0 0 3]]
```
## 2- Indexleme & Dilimleme
``` python
a[0,1]       # 0. satır, 1. sütun
a[:,0]       # ilk sütun
a[1:3, :]    # 1. ve 2. satırlar
a[::-1]      # ters çevirme
```
## 3- Array Özellikleri
### 1) dtype

- dtype, array’deki veri tipini gösterir.
- Bellek yönetimi ve hız açısından çok önemlidir.
- İstersen array oluştururken dtype belirleyebilirsin.
- ekran kayıtlarında data types in numpy var.

#### Örnek:
``` python
import numpy as np

a = np.array([1, 2, 3])          
print(a.dtype)   # int64 (veya int32, sistemine göre)

b = np.array([1.0, 2.0, 3.0])    
print(b.dtype)   # float64

c = np.array([True, False, True])
print(c.dtype)   # bool

d = np.array([1, 2, 3], dtype=np.float32)  
print(d, d.dtype)  
# [1. 2. 3.] float32
```
### 2) copy()
- Array’in tam bağımsız bir kopyasını oluşturur.
- Orijinal değişse bile kopya değişmez.

#### Örnek:
``` python
import numpy as np

a = np.array([1, 2, 3])
b = a.copy()
a[0] = 99

print(a)  # [99  2  3]
print(b)  # [1  2  3]   (etkilenmez)
```
### 3) view()
- Array’in aynı veriye bakan başka bir görünümünü oluşturur.
- Veri ortak, ama şekli farklı olabilir.
- Orijinal değişirse view da değişir.

#### Örnek:
``` python
a = np.array([1, 2, 3])
b = a.view()
a[0] = 99

print(a)  # [99  2  3]
print(b)  # [99  2  3]   (etkilendi)
```
### 4) shape
- Array’in boyutlarını (satır, sütun) verir.
- Tuple (demet) döner.

#### Örnek:
``` python
a = np.array([[1,2,3],
              [4,5,6]])

print(a.shape)   # (2, 3) → 2 satır, 3 sütun
```
## 4- Yeniden Şekillendirme fonksiyonları
- NumPy’de yeniden şekillendirme (reshaping), bir dizinin boyutlarını değiştirmeyi sağlar. Böylece veriyi farklı düzenlerde kullanabiliriz. İşte en çok kullanılan yeniden şekillendirme metodları:

### 1) reshape()

- Dizinin şeklini değiştirmek için kullanılır.
- Eleman sayısı aynı kalmalıdır.

#### Örnek:
``` python
import numpy as np

a = np.arange(6)        # [0, 1, 2, 3, 4, 5]
b = a.reshape(2, 3)     # 2 satır, 3 sütun
print(b)

çıktı:

# [[0 1 2]
#  [3 4 5]]
```
- eğer reshapenin ilk parametresine -1 verilirse, satır sayısını otomatik ayarlar. **reshape(-1,3)**
### 2) ravel()

- Diziyi tek boyutlu hale getirir.
- Orijinal verinin görünümü (view) döner, yani kopya oluşturmaz (mümkünse).

#### Örnek:
``` python
a = np.array([
    [1, 2],
    [3, 4]
 ])
b = a.ravel()
print(b)

çıktı:

# [1 2 3 4]
```

### 3) flatten()

- Diziyi tek boyutlu hale getirir.
- ravel()’dan farkı: Her zaman kopya döndürür.

#### Örnek:
``` python
a = np.array([
    [1, 2],
    [3, 4]
 ])
b = a.flatten()
print(b)

çıktı:

# [1 2 3 4]
```

### 4) resize()

- Diziyi verilen şekle kalıcı olarak yeniden boyutlandırır.
- Eleman yetmezse tekrar eder.

#### Örnek:
``` python
a = np.array([1, 2, 3, 4])
a.resize(2, 3)
print(a)

çıktı:

# [[1 2 3]
#  [4 1 2]]
```
## 5- Sıralamayı değiştirme fonksiyonları

### 1) np.flip()

- Diziyi belirtilen eksen boyunca ters çevirir.
- Eksen belirtilmezse tüm eksenlerde ters çevirir.

#### Örnek:
``` python
import numpy as np

a = np.arange(6).reshape(2, 3)
print(a)
print(np.flip(a, axis=0))   # satırları ters çevirir
print(np.flip(a, axis=1))   # sütunları ters çevirir

çıktı:

# [[0 1 2]
#  [3 4 5]]

# [[3 4 5]
#  [0 1 2]]

# [[2 1 0]
#  [5 4 3]]
```

### 2) np.flipud() (Up/Down)

- Diziyi dikey (yukarı-aşağı) ters çevirir.
- axis=0 için özelleşmiş kısayol.

#### Örnek:
``` python
a = np.arange(9).reshape(3, 3)
print(np.flipud(a))

çıktı:

# [[6 7 8]
#  [3 4 5]
#  [0 1 2]]
 ```

### 3) np.fliplr() (Left/Right)

- Diziyi yatay (sol-sağ) ters çevirir.
- axis=1 için özelleşmiş kısayol.

#### Örnek:
``` python
a = np.arange(9).reshape(3, 3)
print(np.fliplr(a))

çıktı:

# [[2 1 0]
#  [5 4 3]
#  [8 7 6]]
```

## 6- İterasyon (Döngü) Fonksiyonları

### 1) np.ndenumerate

- NumPy’de np.ndenumerate fonksiyonu, çok boyutlu bir diziyi (multi-dimensional array) iterasyon yaparken indeksleri ve değerleri birlikte döndürmeye yarar. Yani, hem elemanın konumunu hem de değerini aynı anda almak için kullanılır.

#### Örnek:
``` python
import numpy as np

arr = np.array([[10, 20], [30, 40]])

for index, value in np.ndenumerate(arr):
    print(f"Index: {index}, Value: {value}")

çıktı: 

# Index: (0, 0), Value: 10
# Index: (0, 1), Value: 20
# Index: (1, 0), Value: 30
# Index: (1, 1), Value: 40

```
### 2) np.nditer()

- Çok boyutlu diziler üzerinde eleman eleman gezinmek için kullanılır.
- Normal for döngüsü satır–satır gezerken, nditer tüm elemanları düz bir şekilde döndürür.
- Daha kontrollü iteration imkânı sağlar (okuma, yazma, sıralı erişim vb.).

#### Temel Kullanım: 
#### Örnek:
``` python
import numpy as np

a = np.arange(6).reshape(2, 3)

for x in np.nditer(a):
    print(x, end=" ")

çıktı:

0 1 2 3 4 5
```
#### Flag parametresi
- Amaç: np.nditer’in davranışını değiştirmek ve ek özellikler eklemek.
- flags bir liste şeklinde verilir ve içinde farklı opsiyonlar bulunabilir.
##### En yaygın kullanılan bayraklar
| Bayrak         | Açıklama                                                      |
|----------------|---------------------------------------------------------------|
| `c_index`      | Elemanların **C-sıralı (satır satır)** indeksini sağlar.     |
| `f_index`      | Elemanların **Fortran-sıralı (sütun sütun)** indeksini sağlar. |
| `multi_index`  | Elemanların **çok boyutlu indekslerini** verir (tuple olarak). |
| `readonly`     | Diziyi **salt okunur** olarak iterasyona alır.              |
| `writeonly`    | Sadece yazma için iterator oluşturur.                       |
| `readwrite`    | Hem okuma hem yazma için (varsayılan).                       |
| `external_loop`| Elemanları bloklar halinde verir, büyük dizilerde hız sağlar.|
| `buffered`     | Bellek tamponu kullanır, veri tiplerini dönüştürmek için gerekir. |

#### Örnek:  flags=["multi_index"] kullanımı:
``` python
import numpy as np

arr = np.array([[5, 6], [7, 8]])

for x in np.nditer(arr, flags=['multi_index']):
    print(f"Index: {x.multi_index}, Value: {x}")

çıktı:

# Index: (0, 0), Value: 5
# Index: (0, 1), Value: 6
# Index: (1, 0), Value: 7
# Index: (1, 1), Value: 8
```

## 7- Dizi Birleştirme

### 1) np.concatenate()

- İki veya daha fazla array'i belirli bir eksende birleştirir.
- Varsayılan eksen axis=0 (satır bazında).

#### Örnek:
``` python
import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])

result=np.concatenate((a,b))
print(result)

çıktı:

# [1,2,3,4,5,6]
```

### 2) np.vstack() (vertical stack)

- Dizileri dikey olarak (satır ekleyerek) birleştirir.
- Daha çok 1D dizileri 2D ye dönüştürmek için kullanılır.

#### Örnek:
``` python
import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])

result=np.vstack((a,b))
print(result)

çıktı:

# [[1 2 3]
#  [4 5 6]]
```

### 3) np.hstack (horizontal stack)

- Dizileri yatay olarak (sütun ekleyerek) birleştirir.
- 1D dizilerde concatenate ile aynı sonucu verir, 2D dizilerde sütun bazında ekler.

#### Örnek:
``` python
import numpy as np

a = np.array([
            [1,2],
            [3,4]
        ])
b = np.array([
            [5,6],
            [7,8]
        ])

result=np.hstack((a,b))
print(result)

çıktı:

# [[1 2 5 6]
#  [3 4 7 8]]
```

### 4) np.stack()

- Dizileri yeni bir eksen oluşturarak birleştirir.
- Örneğin, 1D dizilerden 2D matris oluşturmak için kullanılır.

#### Örnek:
``` python
a = np.array([1,2,3])
b = np.array([4,5,6])

result=np.stack((a,b),axis=0)
print(result)

çıktı:

# [[1 2 3]
#  [4 5 6]]
```

## 8- Dizi Ayırma

### 1) np.array_split()

- Bir diziyi belirli bir sayıda alt parçaya böler.
- Normal split’ten farkı: eleman sayısı eşit bölünemese bile parçaları oluşturur (bazıları küçük olabilir).

#### Örnek:
``` python
import numpy as np

arr=np.arange(10) # [0,1,2,...,9]

result=np.array_split(arr,3)
for i in result:
    print(i)

çıktı:

# [0 1 2 3]
# [4 5 6]
# [7 8 9]
```

### 2) np.hsplit (Horizontal Split)

- Diziyi yatay olarak (sütunlara göre) böler.
- 2D (veya daha yüksek) dizilerde kullanılır.

#### Örnek:
``` python
import numpy as np

arr=np.arange(16).reshape(4,4)
print(arr)

result=np.hsplit(arr,2)
for i in result:
    print(i)

çıktı:

# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]

# [[ 0  1]
#  [ 4  5]
#  [ 8  9]
#  [12 13]]

# [[ 2  3]
#  [ 6  7]
#  [10 11]
#  [14 15]]
```

### 3) np.vsplit (Vertical Split)

- Diziyi dikey olarak (satırlara göre) böler.

#### Örnek:
``` python
import numpy as np

arr=np.arange(16).reshape(4,4)

result=np.vsplit(arr,2)
for i in result:
    print(i)

çıktı:

# [[ 0  1  2  3]
#  [ 4  5  6  7]]

# [[ 8  9 10 11]
#  [12 13 14 15]]
```

### np.dsplit (Depth Split)

- Diziyi derinlik (3. eksen) boyunca böler.
- Yani 3D dizilerde kullanılır.

#### Örnek:
``` python
import numpy as np

arr = np.arange(16).reshape(2, 2, 4)
print(arr)

result = np.dsplit(arr, 2)
for i in result:
    print(i)

çıktı:

# [[[ 0  1  2  3]
#   [ 4  5  6  7]]

#  [[ 8  9 10 11]
#   [12 13 14 15]]]

# --- Sonuç parçaları ---

# [[[0 1]
#   [4 5]]

#  [[8 9]
#   [12 13]]]

# [[[ 2  3]
#   [ 6  7]]

#  [[10 11]
#   [14 15]]]
```

## 9- Arama (Searching) Fonksiyonları

### 1) np.where(condition)

- Şarta uyan indexleri döndürür.

#### Örnek:
``` python
arr=np.array([1,2,3,4,5,6,7,8])

idx=np.where(arr>4)

print(idx)          # (array([4, 5, 6, 7]),)
print(arr[idx])       # [5 6 7 8]
```

### 2) np.argmax,np.argmin

- Dizideki en büyük/en küçük değerin indeksini döndürür.

#### Örnek:
``` python
a = np.array([3, 7, 2, 9, 5])

print(np.argmax(a))  # 3 (9'un indeksi)
print(np.argmin(a))  # 2 (2'nin indeksi)
```

### 3) np.nonzero()

- 0 olmayan elemanların indekslerini döndürür.

#### Örnek:
``` python
a = np.array([0, 4, 0, 2, 7])

print(np.nonzero(a))  # (array([1, 3, 4]),)
```

### 4) np.in1d / np.isin

- Bir dizinin elemanlarının başka bir dizide olup olmadığını kontrol eder.


#### Örnek:
``` python
a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 4, 6])

print(np.isin(a, b))   # [False  True False  True False]
print(a[np.isin(a, b)]) # [2 4]
```
## 10- Sıralama (Sorting) Fonksiyonları

### 1) np.sort()

- Diziyi küçükten büyüğe sıralar. (kopya döndürür).

#### Örnek:
``` python
a = np.array([5, 2, 9, 1, 7])

print(np.sort(a))  # [1 2 5 7 9]
```

### 2) ndarray.sort()

- Diziyi yerinde (in-place) sıralar. (kopya döndürmez).

#### Örnek:
``` python
a = np.array([5, 2, 9, 1, 7])
a.sort()
print(a)  # [1 2 5 7 9]
```

### 3) argsort()

- Sıralı diziyi değil, sıralama indekslerini döndürür.

#### Örnek:
``` python
a = np.array([5, 2, 9, 1, 7])

print(np.argsort(a))  # [3 1 0 4 2] (küçükten büyüğe sıralama sırası)
```

### 4) Çok Boyutlu Sıralama

- Satır veya sütun bazında sıralanabilir.

#### Örnek:
``` python
arr=np.array([
        [3,7,1],
        [9,2,6]
        ])

print(np.sort(arr, axis=1)) # satır bazlı sıralama

çıktı: 

# [[1 3 7]
#  [2 6 9]]
```

## 11- Array Filtreleme

### 1) Boolean mask ile filtreleme

- Bir koşulu sağlayan elemanları filtrelemek için kullanılır.

#### Örnek:
``` python
arr=np.array([23,26,67,73,48,12])

filtereds = arr[arr%2==0]

print(filtereds) # çıktı: [26 48 12]
```
- **not** = np.where ile de filtreleme yapılabilir. (başlık 9 da).

## 12- Random fonksiyonları

### 1) Rastgele sayı üretici oluşturma
``` python
import numpy as np

rng=np.random.default_rng() #rastgele sayı üretici.
```

### 2) 0-1 rastgele sayı
``` python
x = rng.random(5) # 5 adet sayı üretir.
print(x)
```

### 3) Belirli aralıkta tam sayı
``` python
y = rng.integers(low=1, hight=10, size=5) # 1-9 arası 5 sayı
print(y) 
```

### 4) Diziden rastgele seçim
``` python
arr=np.array([21,54,67,45,39])

z = rng.choice(arr, size=2, replace=False) # tekrarsız 2 seçim
print(z)
```

### 5) Seed ile tekrarlanabilirlik
``` python
rng=np.random.default_rng(21) # Aynı seed, aynı çıktılar verir
print(rng.random(5))
```

### 6) Normal dağılıma göre sayı
``` python
f = rng.normal(loc=0, scale=1, size=4) # Ortalama=0, Std=1, 4 sayı
print(f)
```

### 7) np.random.choice

- **p parametresi:** Her elemanın seçilme olasılığını belirler. Toplamları 1 olmalı.
- **size parametresi:** Kaç eleman seçileceğini belirler.
- **replace parametresi:** Aynı eleman tekrar seçilebilir mi onu kontrol eder (True/False).
``` python
import numpy as np

meyveler = ["elma", "armut", "muz", "çilek"]

result = np.random.choice(meyveler, size=5, p=[0.5, 0.2, 0.2, 0.1])
```
- Burada elma %50, armut %20, muz %20, çilek %10 olasılıkla seçilir.
- size=5 olduğu için 5 eleman seçilir.

### 8) np.random.permutation(x) / np.random.shuffle(x)

- **np.random.permutation =** Yeni bir permütasyon dizisi döndürür, orijinal dizi değişmez.
- **np.random.shuffle =** Orijinal diziyi yerinde karıştırır, geri dönüş değeri yok.
``` python
import numpy as np

a = [1,2,3,4]
print(np.random.permutation(a))  # yeni karışık dizi
np.random.shuffle(a)             # a dizisi yerinde karıştı
print(a)
```
## 13- Küme işlemleri

### 1) np.unique

- NumPy'deki numpy.unique fonksiyonu, bir dizideki benzersiz (tekrarsız) elemanları bulmak için kullanılır.
Varsayılan olarak, sıralı bir şekilde eşsiz değerleri döndürür.

**Temel Kullanımı**
``` python
numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)
```
- ***ar*** → giriş dizisi

- ***return_index*** → True olursa, benzersiz elemanların ilk göründüğü indeksleri döndürür

- ***return_inverse*** → True olursa, orijinal diziyi benzersiz değerlere eşleyen indeksleri döndürür

- ***return_counts*** → True olursa, her benzersiz elemanın kaç kez tekrarlandığını döndürür

**return_counts ile kullanımı**
``` python
import numpy as np

arr = [2,4,4,6,5,2]

values,count = np.unique(arr,return_counts=True)
for v,c in zip(values,count):
    print(f"{v} sayısı {c} adet.")

çıktı:

2 sayısı 2 adet.
4 sayısı 2 adet.
5 sayısı 1 adet.
6 sayısı 1 adet.
```
### 2) np.union1d

- np.union1d(ar1, ar2) iki dizideki elemanların birleşimini alır ve sıralı, tekrarsız bir dizi döndürür. Yani hem ar1 hem ar2 dizisindeki tüm benzersiz elemanları toplar.

**Özellikler:**
- Çıktı sıralıdır.
- Tekrarlayan elemanlar bir kez görünür.
- Genellikle set işlemlerinde kullanılır (kümeler gibi).

``` python
import numpy as np

a = np.array([1, 2, 3, 5])
b = np.array([3, 4, 5, 6])

union = np.union1d(a, b)
print(union)

çıktı:
# [1 2 3 4 5 6]
```

### 3) np.intersect1d
- np.intersect1d(ar1, ar2) iki dizinin kesişimini (ortak elemanlarını) döndürür. Çıktı sıralı ve tekrarsızdır.

**Özellikler:**
- Yalnızca her iki dizide de bulunan benzersiz elemanları alır.
- Çıktı küçükten büyüğe sıralıdır.
- Set işlemleri için idealdir.

``` python
import numpy as np

a = np.array([1, 2, 3, 5])
b = np.array([3, 4, 5, 6])

intersection = np.intersect1d(a, b)
print(intersection)

çıktı:
# [3 5]
```

### 4) np.setdiff1d

- np.setdiff1d(ar1, ar2) iki dizinin farkını alır. Yani ar1 dizisinde olup ar2 dizisinde olmayan elemanları döndürür. Çıktı sıralı ve tekrarsızdır.

**Özellikler:**
- ar1 – ar2 işlemi gibi düşünülebilir.
- Tekrarlayan elemanlar yalnızca bir kez görünür.
- Çıktı küçükten büyüğe sıralanır.

``` python
import numpy as np

a = np.array([1, 2, 3, 5])
b = np.array([3, 4, 5, 6])

diff = np.setdiff1d(a, b)
print(diff)

çıktı:
# [1 2]
```

### 5) np.setxor1d

- np.setxor1d(ar1, ar2) iki dizinin simetrik farkını alır. Yani yalnızca bir dizide olup diğerinde olmayan elemanları döndürür. Çıktı sıralı ve tekrarsızdır.

**Özellikler:**
- Ortak elemanlar çıkarılır.
- Tekrarlayan elemanlar yalnızca bir kez görünür.
- Çıktı küçükten büyüğe sıralanır.

``` python
import numpy as np

a = np.array([1, 2, 3, 5])
b = np.array([3, 4, 5, 6])

xor = np.setxor1d(a, b)
print(xor)

çıktı
# [1 2 4 6]
```
## 14- Dağılımlar / Grafikler

### 1) Normal Dağılım (Gaussian Dağılım)

- İstatistikte en çok kullanılan dağılımlardan biridir.
- Çan eğrisi şeklindedir.
- Ortalama (μ) ve standart sapma (σ) ile tanımlanır.

    - **Ortalama**, dağılımın merkezini belirler.
    - **Standart sapma**, yayılımı (ne kadar geniş/dar olduğunu) belirler.

``` python
import numpy as np
import matplotlib.pyplot as plt
import seaborn

# Ortalama = 0, Standart sapma = 1, 1000 örnek
data = np.random.normal(loc=0 ,scale=1, size=1000)

sb.displot(data,kind="kde")

plt.show()
```
- np.random.normal(0, 1, 1000) → Ortalama 0, standart sapma 1 olan 1000 rastgele sayı üretir.

### 2) Binom dağılımı

- Belirli sayıda bağımsız deneme (n) yapılır.
- Her denemede başarı olasılığı (p) aynıdır.
- Rastgele değişken, başarı sayısını verir.

- **Örnek:** 10 kere yazı-tura atarsan (n=10), yazı gelme olasılığı p=0.5 olsun. O zaman binom dağılımı sana “kaç defa yazı gelir?” sorusunun dağılımını gösterir.

``` python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Parametreler: n=10 deneme, p=0.5 başarı olasılığı, 1000 tekrar
data = np.random.binomial(n=10, p=0.5, size=1000)

sb.displot(data,kind="hist")

plt.show()
```

### 3) Poisson dağılımı

- Belirli bir zaman aralığında veya alanda nadir olayların sayısını modellemek için kullanılır.
- Tek parametresi vardır: λ (lambda) → beklenen olay sayısı (ortalama).
- Örnek: Bir saatte ortalama 3 müşteri geliyorsa, “tam olarak 5 müşteri gelme” olasılığı Poisson dağılımı ile hesaplanır.

``` python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Parametre: λ = 3, 1000 örnek
data = np.random.poisson(lam=3, size=1000)

sb.displot(data,kind="hist")

plt.show()
``` 

### 4) Uniform dağılım

- Amacı: Belirtilen aralıkta eşit olasılıkla (uniform dağılım) rastgele sayı üretir.

``` python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# low: Alt sınır, high: Üst sınır
data = np.random.uniform(low=0.0, high=10.0, size=1000)

sb.displot(data,kind="hist")

plt.show()
```

### 5) Lojistik Dağılım (Logistic Distribution)

- Lojistik dağılım, normal dağılıma çok benzeyen çan şeklinde bir dağılımdır.
- Ancak kuyrukları normal dağılıma göre daha ağırdır (uç değerler daha olasıdır).
- İstatistikte genelde lojistik regresyon, nüfus büyümesi ve olasılık modellemeleri için kullanılır.
    
    **Parametreleri:**

    - loc → dağılımın merkezi (ortalama benzeri)

    - scale → dağılımın yayılımını belirler (standart sapma benzeri)

``` python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Lojistik dağılımdan 1000 örnek üretelim
data = np.random.logistic(loc=0, scale=1, size=1000)

sb.displot(data,kde=True)

plt.show()
```
- ***178 ve 179. derslerde de dağılımlar var. Zamanı gelince bak.***

## 15- Evrensel Fonksiyonlar

### 1) Aritmetik ve Matematiksel Fonksiyonlar

| Fonksiyon | Açıklama | Örnek Kullanım |
|-----------|----------|----------------|
| `np.add(x, y)` | Eleman bazında toplama | `np.add([1, 2], [3, 4]) → [4, 6]` |
| `np.subtract(x, y)` | Eleman bazında çıkarma | `np.subtract([5, 7], [2, 3]) → [3, 4]` |
| `np.multiply(x, y)` | Eleman bazında çarpma | `np.multiply([2, 3], [4, 5]) → [8, 15]` |
| `np.prod()`   | Dizideki elemanların çarpımını alır           | `np.prod([1,2,3,4])`            | `24`       |
|               | Belirli eksende çarpım yapabilir               | `np.prod([[1,2],[3,4]], axis=0)`| `[3 8]`    |
|               |                                               | `np.prod([[1,2],[3,4]], axis=1)`| `[2 12]`   |
| `np.divide(x, y)` | Eleman bazında bölme (float) | `np.divide([10, 20], [2, 5]) → [5., 4.]` |
| `np.floor_divide(x, y)` | Tamsayıya yuvarlanmış bölme | `np.floor_divide([7, 10], [2, 3]) → [3, 3]` |
| `np.mod(x, y)` | Mod alma (kalan) | `np.mod([7, 10], [2, 3]) → [1, 1]` |
| `np.remainder(x, y)` | Kalan değeri döndürür (işaret farkı olabilir) | `np.remainder([7, -7], [2, 2]) → [1, -1]` |
| `np.power(x, y)` | Eleman bazında üs alma | `np.power([2, 3], [3, 2]) → [8, 9]` |
| `np.negative(x)` | Eleman bazında işaret değiştirme | `np.negative([1, -2, 3]) → [-1, 2, -3]` |
| `np.abs(x)` / `np.absolute(x)` | Mutlak değer alma | `np.abs([-1, -2, 3]) → [1, 2, 3]` |
| `np.sqrt(x)` | Karekök alma | `np.sqrt([1, 4, 9]) → [1., 2., 3.]` |
| `np.square(x)` | Karesini alma | `np.square([2, 3]) → [4, 9]` |
| `np.exp(x)` | e tabanında üstel alma | `np.exp([0, 1]) → [1., 2.718...]` |
| `np.log(x)` | Doğal logaritma (ln) | `np.log([1, np.e, np.e**2]) → [0., 1., 2.]` |
| `np.log10(x)` | 10 tabanında logaritma | `np.log10([1, 10, 100]) → [0., 1., 2.]` |
| `np.log2(x)` | 2 tabanında logaritma | `np.log2([1, 2, 4]) → [0., 1., 2.]` |
| `np.log1p(x)` | `log(1 + x)` daha hassas hesaplama yapar, küçük `x` değerlerinde faydalı | `np.log1p([1e-10]) ≈ 1e-10` |
| `np.expm1(x)` | `exp(x) - 1` daha hassas hesaplama yapar, küçük `x` için faydalı | `np.expm1([1e-5]) ≈ 1e-5` |
| `np.logaddexp(x, y)` | `log(exp(x) + exp(y))`’i sayısal olarak kararlı şekilde hesaplar | `np.logaddexp(1, 2) → ~2.313` |
| `np.logaddexp2(x, y)` | `log2(2**x + 2**y)`’i sayısal olarak kararlı şekilde hesaplar | `np.logaddexp2(3, 4) → ~4.585` |
| `np.sin(x)` | Sinüs | `np.sin([0, np.pi/2]) → [0., 1.]` |
| `np.cos(x)` | Kosinüs | `np.cos([0, np.pi]) → [1., -1.]` |
| `np.tan(x)` | Tanjant | `np.tan([0, np.pi/4]) → [0., 1.]` |
| `np.arcsin(x)` | Ters sinüs | `np.arcsin([0, 1]) → [0., 1.5708...]` |
| `np.arccos(x)` | Ters kosinüs | `np.arccos([1, 0]) → [0., 1.5708...]` |
| `np.arctan(x)` | Ters tanjant | `np.arctan([0, 1]) → [0., 0.7853...]` |
| `np.deg2rad(x)` | Derece → Radyan | `np.deg2rad([0, 180]) → [0., 3.1415...]` |
| `np.rad2deg(x)` | Radyan → Derece | `np.rad2deg([0, np.pi]) → [0., 180.]` |
| `np.maximum(x, y)` | Eleman bazında maksimum | `np.maximum([2, 5], [3, 4]) → [3, 5]` |
| `np.minimum(x, y)` | Eleman bazında minimum | `np.minimum([2, 5], [3, 4]) → [2, 4]` |
| `np.clip(x, min, max)` | Değerleri belirtilen aralığa sıkıştırır | `np.clip([1, 5, 10], 2, 8) → [2, 5, 8]` |

### 2)  Yuvarlama Fonksiyonları

| Fonksiyon | Açıklama | Örnek Kullanım |
|-----------|----------|----------------|
| `np.round(x, decimals=0)` / `np.around(x, decimals=0)` | Sayıları belirtilen basamağa yuvarlar | `np.round([1.25, 1.35], 1) → [1.2, 1.4]` |
| `np.rint(x)` | En yakın tamsayıya yuvarlar (float döndürür) | `np.rint([1.2, 1.8, 2.5]) → [1., 2., 2.]` |
| `np.fix(x)` | Sıfıra doğru yuvarlar | `np.fix([1.7, -1.7]) → [1., -1.]` |
| `np.floor(x)` | Aşağıya yuvarlar (en küçük tam sayı) | `np.floor([1.7, -1.7]) → [1., -2.]` |
| `np.ceil(x)` | Yukarıya yuvarlar (en büyük tam sayı) | `np.ceil([1.2, -1.2]) → [2., -1.]` |
| `np.trunc(x)` | Ondalığı atarak tam kısmı döndürür | `np.trunc([1.7, -1.7]) → [1., -1.]` |

### 3) Durum Kontrol (boolean / koşul) Fonksiyonları

| Fonksiyon        | Açıklama                                      | Örnek                        | Çıktı           |
|-----------------|-----------------------------------------------|------------------------------|----------------|
| `np.all()`       | Tüm elemanların koşulu sağladığını kontrol eder | `np.all([True, True])`       | `True`         |
| `np.any()`       | En az bir elemanın koşulu sağladığını kontrol eder | `np.any([True, False])`    | `True`         |
| `np.isfinite()`  | Elemanın sonlu olup olmadığını kontrol eder   | `np.isfinite([1, np.inf, np.nan])` | `[ True False False]` |
| `np.isinf()`     | Elemanın sonsuz olup olmadığını kontrol eder  | `np.isinf([1, np.inf, -np.inf])`  | `[False  True  True]` |
| `np.isnan()`     | Elemanın NaN olup olmadığını kontrol eder     | `np.isnan([1, np.nan])`     | `[False  True]` |
| `np.isposinf()`  | Elemanın pozitif sonsuz olup olmadığını kontrol eder | `np.isposinf([1, np.inf, -np.inf])` | `[False True False]` |
| `np.isneginf()`  | Elemanın negatif sonsuz olup olmadığını kontrol eder | `np.isneginf([1, np.inf, -np.inf])` | `[False False True]` |
| `np.isreal()`    | Elemanın reel sayı olup olmadığını kontrol eder | `np.isreal([1+0j, 2])`     | `[ True  True]` |
| `np.iscomplex()` | Elemanın kompleks sayı olup olmadığını kontrol eder | `np.iscomplex([1+2j, 3])`  | `[ True False]` |
| `np.isclose()`   | İki değer veya dizinin yaklaşık eşit olup olmadığını kontrol eder | `np.isclose([1.0,2.0],[1.0,2.01])` | `[ True False]` |
| `np.logical_and()` | İki koşulun ve işlemi sonucunu verir       | `np.logical_and([True, False],[True, True])` | `[ True False]` |
| `np.logical_or()` | İki koşulun veya işlemi sonucunu verir      | `np.logical_or([True, False],[False, False])` | `[ True False]` |
| `np.logical_not()` | Koşulun tersini verir                       | `np.logical_not([True, False])` | `[False  True]` |
| `np.logical_xor()` | Koşulların sadece birinin doğru olması durumunu kontrol eder | `np.logical_xor([True, False],[False, False])` | `[ True False]` |

### 4) Karşılaştırma Fonksiyonları

| Fonksiyon           | Açıklama                     | Örnek                         | Çıktı                 |
|--------------------|-----------------------------|-------------------------------|----------------------|
| `np.equal()`        | Elemanlar eşit mi?          | `np.equal([1,2,3],[1,2,4])`  | `[ True  True False]` |
| `np.not_equal()`    | Elemanlar eşit değil mi?    | `np.not_equal([1,2,3],[1,2,4])` | `[False False  True]` |
| `np.greater()`      | Elemanlar büyük mü?         | `np.greater([1,2,3],[0,2,4])` | `[ True False False]` |
| `np.greater_equal()`| Elemanlar büyük veya eşit mi?| `np.greater_equal([1,2,3],[1,2,4])` | `[ True  True False]` |
| `np.less()`         | Elemanlar küçük mü?         | `np.less([1,2,3],[2,2,1])`   | `[ True False False]` |
| `np.less_equal()`   | Elemanlar küçük veya eşit mi?| `np.less_equal([1,2,3],[1,3,2])` | `[ True  True False]` |

### 5) Mantıksal İşlem Fonksiyonları (Logical Operations)

| Fonksiyon           | Açıklama                              | Örnek Kullanım                          | Çıktı             |
|--------------------|--------------------------------------|-----------------------------------------|-----------------|
| `np.logical_and()`  | Elemanların ve işlemi                  | `np.logical_and([True,False],[True,True])` | `[ True False]` |
| `np.logical_or()`   | Elemanların veya işlemi                | `np.logical_or([True,False],[False,False])` | `[ True False]` |
| `np.logical_not()`  | Elemanların tersini al                 | `np.logical_not([True,False])`          | `[False  True]` |
| `np.logical_xor()`  | Elemanların yalnızca biri doğru mu?    | `np.logical_xor([True,False],[False,False])` | `[ True False]` |

### 6) Bit Düzeyinde İşlem Fonksiyonları (Bitwise Operations)

| Fonksiyon           | Açıklama                              | Örnek Kullanım                          | Çıktı             |
|--------------------|--------------------------------------|-----------------------------------------|-----------------|
| `np.bitwise_and()`  | Bit düzeyinde ve (&)                  | `np.bitwise_and([0b110,0b101],[0b101,0b011])` | `[4 1]` |
| `np.bitwise_or()`   | Bit düzeyinde veya (|)                 | `np.bitwise_or([0b110,0b101],[0b101,0b011])` | `[7 7]` |
| `np.bitwise_xor()`  | Bit düzeyinde xor (^)                  | `np.bitwise_xor([0b110,0b101],[0b101,0b011])` | `[3 6]` |
| `np.invert()`       | Bitleri tersine çevir (~)              | `np.invert([0b110,0b101])`               | `[-7 -6]` |
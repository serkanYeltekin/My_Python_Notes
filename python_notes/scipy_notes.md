# SciPy (Scientific Python)

## 1- Sabitler
**Kullanım**
``` python
from scipy import constants

# örneğin minute değerini almak istiyoruz
constant = constants.minute

print(constant) # çıktı: 60.0
```
## 2- Optimize

### a) Minimize

-minimize, çok boyutlu (ve tek boyutlu) bir fonksiyonun minimumunu bulmak için kullanılan genel amaçlı bir optimizasyon fonksiyonudur.

```python
from scipy.optimize import minimize

def myFunction(x):
    return x**2+x+2

x0=0 # başlangıç değeri
result = minimize(myFunction,x0,method="BFGS")

print(result)
```
**çıktı :**
```python
 message: Optimization terminated successfully.  
 success: True             # Optimizasyon başarılı (çözüm bulundu).
 status: 0                 # Çıkış durumu kodu (0 → başarı).
      fun: 1.75            # Minimum noktada fonksiyonun değeri.
        x: [-5.000e-01]    # Bulunan minimum nokta (x = -0.5).
      nit: 2               # İterasyon sayısı (algoritma 2 adımda bitti).
      jac: [ 0.000e+00]    # Gradient (türev) değeri, yani eğim ≈ 0.
 hess_inv: [[ 5.000e-01]]  # Yaklaşık ters Hessian matrisi (eğriliği gösterir).
     nfev: 8               # Fonksiyon değerlendirme (kaç kere f(x) hesaplandı).
     njev: 4               # Gradient değerlendirme (kaç kere türev hesaplandı).
```

### b) Root
- root, çok değişkenli (ve tek değişkenli) denklemlerin köklerini bulmak için kullanılan genel amaçlı bir fonksiyondur.
```python
from scipy.optimize import root
import numpy as np

def myFunction(x):
    return x + np.cos(x)

x0=0 # başlangıç değeri
result = root(myFunction,x0)

print(result.x)
```
**çıktı :**
```python
[-0.73908513]  # f(x) = 0 koşulunu sağlayan x değeri.
```
## 3- Seyrek verilerle çalışma
🔹 Seyrek Matrislerin Avantajı

- Normal (dense) matrisler tüm elemanları saklar → çok bellek tüketir.
- Sparse matrisler sadece sıfır olmayan elemanları saklar → bellek ve hız avantajı sağlar.

🔹 SciPy’de Başlıca Sparse Matris Formatları

scipy.sparse birçok format sunar, en yaygınları:

**CSR (Compressed Sparse Row)**

- Satır bazlı sıkıştırma
- Matris-çarpımı ve satır işlemleri için verimli
- csr_matrix

**CSC (Compressed Sparse Column)**

- Sütun bazlı sıkıştırma
- Matris-çarpımı ve sütun işlemleri için verimli
- csc_matrix

```python
import numpy as np
from scipy.sparse import csr-matrix

# Normal matris
dense = np.array([
    [0, 0, 3],
    [4, 0, 0],
    [0, 0, 5]
])

# Seyrek CSR matrise dönüştür
sparse = csr_matrix(dense)

print("Seyrek matris:")
print(sparse)
print("Normal matrise çevir:")
print(sparse.toarray())
```
**çıktı :**
```python
Seyrek matris:
  (0, 2)	3
  (1, 0)	4
  (2, 2)	5

Normal matrise çevir:
[[0 0 3]
 [4 0 0]
 [0 0 5]]
```

## 4- Connected Components” (Bağlı Bileşenler)

**1️⃣ Tanım**

**Bir graf 𝐺=(𝑉,𝐸) düşünün:**
- V = düğümler (nodes)
- E = kenarlar (edges)

**Connected Component (Bağlı Bileşen):**
- Grafın bir alt kümesidir.
- Bu alt kümedeki her düğüm diğer düğümlere bir yol ile ulaşabilir.
- Farklı bağlı bileşenler arasında yol yoktur.

Örneğin:
```python
1 — 2       4 — 5
|
3
```
- 1-2-3 bir bağlı bileşen
- 4-5 başka bir bağlı bileşen

**2️⃣ SciPy’de Kullanımı**
```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

arr = np.array([
        [0,1,2],
        [1,0,0],
        [2,0,0]
    ])
result = csr_matrix(arr)
print(connected_components(result))
```
**çıktı :** 
```python
(1, array([0, 0, 0]))
```
- 1 bağlı bileşen var
- Tüm düğümler bu bileşene ait (labels = [0,0,0]) ✅

## 5- Dijkstra
- Amaç: Başlangıç düğümünden diğer tüm düğümlere olan en kısa yolu bulmak.

- Kullanım: Yol bulma, harita, ağ yönlendirme, trafik optimizasyonu, vb.

- Sadece pozitif ağırlıklı kenarlar için geçerlidir.

- Algoritma, greedy (açgözlü) bir yaklaşımla çalışır:

    - Başlangıç düğümünden en yakın düğümü seçer

    - Bu düğüm üzerinden diğer düğümlere olan mesafeleri günceller

    - Tüm düğümler işlenene kadar devam eder

**dijkstra Parametreleri**

- **csgraph** → grafın adjacency matrix’i (sparse veya dense)
- **indices** → hangi düğümden başlanacağı (burada 0)
- **return_predecessors**=True → en kısa yolların önceki düğümlerini de döndür

**Örnek:**
```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

arr = np.array([
        [0,1,2],
        [1,0,0],
        [2,0,0]
    ])
result = csr_matrix(arr)
print(dijkstra(result,return_predecessors=True,indices=0))
```
**çıktı :** 
```python
(array([0., 1., 2.]), array([-9999,     0,     0]))
```
**Adım Adım Hesaplama**

- 0 → 0 : mesafe 0 (kendisi)
- 0 → 1 : mesafe 1 (0 → 1)
- 0 → 2 : mesafe 2 (0 → 2)

**Önceki düğümler (predecessors):**

- 0 → 0 : başlangıç → -9999
- 0 → 1 : önceki düğüm = 0
- 0 → 2 : önceki düğüm = 0
> Ders 195 Python Programlama SciPy'da Graph'lar. (17:27. dakikadan itibaren.)

## 6 - İnterp1d

- **interp1d**, SciPy kütüphanesindeki **(scipy.interpolate)** bir fonksiyondur. Görevi, verilen veriler arasında ara değerler (interpolasyon) üretmektir.
- Yani elimizde belli noktalarda hesaplanmış/verilmiş veriler varsa, bu noktaların arasındaki değerleri tahmin etmemizi sağlar.

```python
from scipy.interpolate import interp1d
import numpy as np

x = np.arange(10)
y = 2*x+1

ip_func = interp1d(x,y)
result = ip_func(np.arange(2.1,3.0,0.1))

print(result)
```
**çıktı :** 
```python
[5.2 5.4 5.6 5.8 6.  6.2 6.4 6.6 6.8]
```
- x = 2.1 için y = 2*2.1 + 1 = 5.2
- x = 2.9 için y = 2*2.9 + 1 = 6.8
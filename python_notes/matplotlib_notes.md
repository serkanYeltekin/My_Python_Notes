# Matplotlib
## Matplotlib Temel Grafik Ayarları ve Parametreleri

### 1. Başlık ve Eksen Etiketleri

Grafiğe başlık ve eksen isimleri ekler.

```python
plt.title("Başlık", fontsize=14, color="red", loc="center")
plt.xlabel("X Ekseni", fontsize=12, color="blue")
plt.ylabel("Y Ekseni", fontsize=12, color="green")
```

**Parametreler:**

* `fontsize` → Yazı boyutu
* `color` → Yazı rengi
* `loc` → Başlığın konumu (`"left"`, `"center"`, `"right"`)

---

### 2. Izgara (Grid)

Grafikteki çizgilerle verileri daha kolay okumayı sağlar.

```python
plt.grid(True, linestyle="--", linewidth=0.7, color="gray", alpha=0.7)
```

**Parametreler:**

* `axis` → `"x"`, `"y"`, `"both"` (hangi eksende gösterileceği)
* `linestyle` → `"--"`, `"-."`, `":"`
* `linewidth` → Çizgi kalınlığı
* `color` → Renk
* `alpha` → Saydamlık (0–1 arası)

---

### 3. Eksen Limitleri

Grafikte hangi aralıkların gösterileceğini belirler.

```python
plt.xlim(0, 5)
plt.ylim(2, 8)
```

**Parametreler:**

* `min`, `max` → Gösterilecek eksen aralığı

---

### 4. İşaretler (Ticks)

Eksen üzerindeki sayıları veya noktaları özelleştirir.

```python
plt.xticks([0,1,2,3,4,5], rotation=45, fontsize=10)
plt.yticks([2,4,6,8], fontsize=12)
```

**Parametreler:**

* `rotation` → Yazıları döndürme açısı
* `fontsize` → Yazı boyutu
* `labels` → Özel etiketler ekleme

---

### 5. Açıklama (Legend)

Birden fazla veri serisi çizdiğinde hangi serinin ne olduğunu gösterir.

```python
plt.plot(x, y, label="Seri 1")
plt.legend(loc="upper left", fontsize=10, frameon=True, title="Açıklamalar")
```

**Parametreler:**

* `loc` → Konum (`"upper left"`, `"lower right"`, `"best"`)
* `fontsize` → Yazı boyutu
* `frameon` → Legend kutusunu aç/kapa
* `title` → Legend başlığı

---

### 6. Çizgi ve Nokta Stili

Çizgi rengi, tipi ve noktaların görünümü ayarlanır.

```python
plt.plot(x, y, color="blue", linestyle="--", linewidth=2, marker="o", markersize=6)
```

**Parametreler:**

* `color` → Çizgi rengi (`"r"`, `"blue"`, `"#FF5733"`)
* `linestyle` → Çizgi tipi (`"-"`, `"--"`, `":"`)
* `linewidth` → Çizgi kalınlığı
* `marker` → Nokta tipi (`"o"`, `"s"`, `"x"`)
* `markersize` → Nokta boyutu

---

### 7. Figür Boyutu

Grafiğin boyutunu ayarlamak için kullanılır.

```python
plt.figure(figsize=(8,5))
```

**Parametreler:**

* `figsize` → `(genişlik, yükseklik)`

---

İstersen ben bunu alıp **tek bir örnek grafikte tüm parametreleri kullanan minimal kod** haline getirebilirim; böylece doğrudan çalıştırıp görebilirsin. Bunu yapayım mı?

## Çubuk Grafiği
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A","B","C","D"])
y = np.array([3,7,1,12])

# Çubuklar dikey olur
plt.bar(x,y,widht=0.4)

# Çubuklar yatay olur
plt.barh(x,y,height=0.4)

plt.show()
```
- **widht** dikey çucuklarda genişliği ayarlar.
- **height** yatay çubuklarda genişliği ayarlar.

## Histogram Grafiği
```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()
mydata=rng.normal(loc=170,scale=10,size=250)

plt.hist(mydata)
plt.show()
```
- Çubuk grafiğine benzer çubuklar birleşiktir.
- **loc =** 0rtalama boyu belirler.
- **scale =** Standart sapmadır.
- **size =** Veri sayısıdır.

## Dağılım Grafiği (scatter)
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,4,6,11,4,7,9,3,6,4,8])
y = np.array([90,60,70,120,65,55,100,50,90,100,100,110,75])

plt.scatter(x,y)
plt.show()
```
- Nokta dağılım grafiğidir.

## Pasta Grafiği (pie)
```python
import matplotlib.pyplot as plt
import numpy as np

y = np.array([50,25,20,5])
my_label_names = ["Apple","Banana","Strawberries","Cherries"]
my_explode = [0,0,0,0.3]

plt.pie(y,labels=my_label_names,explode=my_explode)
plt.show()
```
- **y =** Dilimlerin yüzdelik değerleri.
- **explode =** Pastadan dilimleri uzaklaştırmaya yarar.
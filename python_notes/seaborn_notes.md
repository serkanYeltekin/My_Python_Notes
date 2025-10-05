# Seaborn
## 1- displot
- Seaborn’daki displot fonksiyonu, dağılım (distribution) görselleştirmeleri için kullanılır. Veri setindeki bir sayısal değişkenin dağılımını histogram, kernel density estimate (KDE) veya ikisinin birleşimi şeklinde çizebilir.

**Özellikleri:**
- Varsayılan olarak histogram çizer.
- **kind="kde"** ile yoğunluk grafiği (density plot) çizebilir.
- **kind="ecdf"** ile birikimli dağılım fonksiyonu (ECDF) çizebilir.
- **col, row** gibi parametrelerle facet mantığı kullanarak farklı kategorilere göre dağılımı ayırabilir.

Örnek:
``` python
import seaborn as sb
import matplotlib.pyplot as plt

arr = [55, 60, 65, 70, 70, 72, 75, 80, 82, 85, 85, 90, 92, 95, 100]

# KDE (yoğunluk grafiği)
sb.displot(arr,kind="kde")

plt.show()
```
## 2- lineplot

-Seaborn’daki sb.lineplot, özellikle zaman serileri veya sayısal veriler arasındaki ilişkiyi göstermek için kullanılan **çizgi grafiğidir**.

Örnek:
``` python
import seaborn as sb
import matplotlib.pyplot as plt

# veriler
gunler = [1, 2, 3, 4, 5, 6, 7]
notlar = [55, 65, 60, 70, 75, 80, 85]

sb.lineplot(x=gunler, y=notlar)

plt.show()
```

## 3- scatterplot

- Seaborn’daki sb.scatterplot, iki sayısal değişken arasındaki ilişkiyi göstermek için kullanılan **serpiştirme (nokta) grafiği’dir.**

Örnek:
``` python
import seaborn as sb
import matplotlib.pyplot as plt

# veriler
boy = [160, 165, 170, 175, 180, 185]
kilo = [55, 60, 65, 70, 75, 80]

sb.scatterplot(x=boy, y=kilo)

plt.show()

```

## 4- barplot

- Seaborn’daki sb.barplot, kategorik verilerin ortalama veya toplam değerlerini görselleştirmek için kullanılır.

Örnek:
``` python
import seaborn as sb
import matplotlib.pyplot as plt

# Kategoriler ve değerler
meyveler = ["Elma", "Muz", "Portakal", "Çilek"]
satis = [50, 70, 30, 90]

sb.barplot(x=meyveler, y=satis)

plt.show()
```
## 5- heatmap

- Seaborn’daki sb.heatmap, genellikle tablo veya matris verilerindeki değerleri **renk yoğunluğu ile** görselleştirmek için kullanılır.

Örnek:
``` python
import seaborn as sb
import matplotlib.pyplot as plt

data=np.array([[1,2],[6,7]])

sb.heatmap(data)

plt.show()
```
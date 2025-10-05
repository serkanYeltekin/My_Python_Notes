# ML Notes

## 1- Popülasyon Standart Sapma ve Örneklem Standart Sapma

### 1) Popülasyon Standart Sapma ($\sigma$)
---
- **Tanım:** Bir **bütün kitle (popülasyon)** içerisindeki tüm verilerin ortalamadan ne kadar saptığını ölçer.  
- Elimizde **tüm veriler** vardır (örneğin bir şehirde yaşayan herkesin boyu).  
- **Formül:**

$$
\sigma = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}}
$$

- Açıklama:
  - $x_i$ → her bir veri  
  - $\mu$ → popülasyon ortalaması  
  - $N$ → toplam veri sayısı  

> Burada **bölüm N**’dir, çünkü tüm popülasyonu biliyoruz.
---
```python
import numpy as np

speed = [48,49,50,51,52]

print(f"Popülasyon Standart Sapma: {np.std(speed,ddof=0)}") # ddof = 0 iken Popülasyon Standart Sapma bulunur. 
```
### 2) Örneklem Standart Sapma ($s$)
---
- **Tanım:** Popülasyondan alınmış bir **örneklem** (küçük grup) verilerinin ortalamadan sapmasını ölçer.  
- Gerçek popülasyonu bilmediğimiz için sadece örneklemden tahmin yaparız.  
- **Formül:**

$$
s = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}}
$$

- Açıklama:
  - $x_i$ → her bir örneklem verisi  
  - $\bar{x}$ → örneklem ortalaması  
  - $n$ → örneklemdeki veri sayısı  

> Burada **bölüm n-1**’dir. Buna **Bessel düzeltmesi** denir. Amaç, popülasyonun standart sapmasını daha doğru tahmin edebilmektir.
---
```python
import numpy as np

speed = [48,49,50,51,52]

print(f"Örneklem Standart Sapma: {np.std(speed,ddof=1)}") # ddof = 1 iken Örneklem Standart Sapma bulunur.
```
### 3) Özetle Fark
---
| Özellik | Popülasyon Std. Sapma ($\sigma$) | Örneklem Std. Sapma ($s$) |
|---------|---------------------------|-------------------------|
| Veri kümesi | Tüm popülasyon | Örneklem |
| Bölüm | $N$ | $n-1$ |
| Kullanım | Gerçek dağılımı ölçmek | Popülasyonu tahmin etmek |
| Hata düzeltme | Yok | Var (Bessel düzeltmesi) |

## 2- Varyans

**Varyans**, bir veri kümesindeki değerlerin **ortalama etrafında ne kadar dağıldığını** ölçen istatistiksel bir kavramdır.  

- Standart sapmanın karesidir.
- Eğer bütün değerler ortalamaya çok yakınsa, varyans **küçük** çıkar.  
- Eğer değerler ortalamadan çok uzaksa, varyans **büyük** çıkar.  
- **Yorum yapmak** için standart sapma, **hesaplama ve teori** için varyans kullanılır.
---

### *Matematiksel Tanım :*
---
Bir veri kümesinde değerler:  

$$
x_1, x_2, ..., x_n
$$

Ortalama (aritmetik ortalama):  

$$
\mu = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

Varyans:  

$$
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2
$$

Standart sapma:  

$$
\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}
$$
---
```python
import numpy as np

speed = [48,49,50,51,52]

print(f"Popülasyon Varyansı: {np.var(speed,ddof=0)}")
print(f"Örneklem Varyansı: {np.var(speed,ddof=1)}")
```
## 3- Percentiles
📌 Percentile Nedir?

Percentile, verilerin sıralı dağılımı içinde bir değerin konumunu belirtir.

Örneğin:

- 25. percentile (Q1): Verilerin %25’i bu değerin altında yer alır.
- 50. percentile (median): Verilerin %50’si bu değerin altında, %50’si üstünde.
- 90. percentile: Verilerin %90’ı bu değerin altında.

ML’de genellikle:

- Feature scaling / normalization (özellik ölçekleme),
- Outlier detection (aykırı değer tespiti),
- Model değerlendirme (örn. latency %95 percentile)
gibi yerlerde kullanılır.

```python
import numpy as np

notes = [1,2,5,7,8,10,12,15,18,20]

print(f"25. percentile: ", np.percentile(notes,25)) 
print(f"50. percentile: ", np.percentile(notes,50))
print(f"75. percentile: ", np.percentile(notes,75))
print(f"90. percentile: ", np.percentile(notes,90))
```
**çıktı :**
```python
25. percentile: 5.75
50. percentile: 9.0
75. percentile: 14.25
90. percentile: 18.2
```
- **Öreğin ilk çıktı sayıların %25'inin 5.75 veya ondan daha az bir değere sahip olduğunu gösteriyor.**

### ML'de Kullanım Örneği : 
- Outlier detection (aykırı değerleri temizleme):
```python
import numpy as np

# Örnek veri (içinde aykırı değer var)
data = np.array([10, 12, 14, 15, 16, 18, 20, 100])

# 5. ve 95. yüzdelikleri al
low,high = np.percentile(data, [5,95])

# Bu aralığın dışında kalanları "outlier" say
filtered_data = data[(data>=low) & (data<= high)]

print("Orijinal veri:", data)
print("Filtrelenmiş veri:", filtered_data)
```
**çıktı :**
```python
Orijinal veri: [ 10  12  14  15  16  18  20 100]
Filtrelenmiş veri: [10 12 14 15 16 18 20]
```
> Burada 100, aykırı değer olarak çıkarıldı.

## 4- Ölçeklendirme (Scaling)
Verilerin belirli bir aralığa dönüştürülmesi veya standardize edilmesi işlemidir.Çünkü farklı ölçeklerdeki veriler algoritmaların performansını olumsuz etkileyebilir.

### 1. Min-Max Scaling (Normalizasyon)
- Verileri 0 ile 1 arasına (ya da belirlenen başka bir aralığa) dönüştürür.

**Formül :**
$$
X' = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
$$
- Avantaj: Özellikle nöral ağlar gibi algoritmalarda yaygın.

- Dezavantaj: Aykırı değerler (outlier) varsa çok etkilenir.
- X′, "dönüştürülmüş" ya da "ölçeklendirilmiş" yeni değerleri ifade etmek için kullanılan bir gösterimdir.

**Örnek :**
```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

notes = np.array([[50],[60],[70],[80],[90]])

scaler = MinMaxScaler()

result = scaler.fit_transform(notes)

print(result)
```
**Çıktı :** 
```python
[[0.  ]
 [0.25]
 [0.5 ]
 [0.75]
 [1.  ]]
```
### 2. Standardizasyon (Z-score Scaling)
- Verilerin ortalamasını 0, standart sapmasını 1 yapar.

**Formül :**
$$
X' = \frac{X - \mu}{\sigma}
$$
- Avantaj: Aykırı değerlerden Min-Max kadar etkilenmez.
- Kullanım: Lineer regresyon, lojistik regresyon, SVM, PCA gibi yöntemlerde çok yaygın.

**Örnek :**
```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = {
    "Height":[160,170,180,190],
    "Weight":[60,70,80,90]
}

df = pd.DataFrame(data)

scaler = StandardScaler()

result = scaler.fit_transform(df)

print(pd.DataFrame(result,columns=["Height","Weight"]))
```
**Çıktı :**
```python
     Height    Weight
0 -1.341641 -1.341641
1 -0.447214 -0.447214
2  0.447214  0.447214
3  1.341641  1.341641
```
**Örnek kullanım :**
```python
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import pandas as pd

# Standard ölçeklendirmenin nesnesini oluşturduk
scale = StandardScaler()

# Veri okuma ve x,y değişkenlerine değerler atanıyor. X girdiler, y ise istenilen değerdir
df = pd.read_csv("data (1).csv")
X = df[["Weight","Volume"]]
y = df["CO2"]

# X değerlerini ölçeklendirdik
scaledX = scale.fit_transform(X)

# Öğrenmeyi ölçeklendirilmiş değerler üzerinden yapar
regr = linear_model.LinearRegression()
regr.fit(scaledX,y)

# Tahmin edilmesi istenen veri girildi ve ölçeklendiridi
new_data = pd.DataFrame([[2300,1.3]],columns=["Weight","Volume"])
scaled = scale.transform(new_data)

# Tahmin yap
result = regr.predict(scaled)
print(result)
```
Çıktı : 
```python
[107.2087328]
```

## 5- Train-test yapısı
Makine öğrenmesinde elimizde genelde etiketli bir veri seti olur (örneğin öğrenci notları ve başarı durumu, ev fiyatları ve özellikleri vs.).
- Bu veri setini ikiye ayırırız:

  - Train (eğitim verisi) → Modelin öğrenmesi için kullandığımız kısım.

  - Test (test verisi) → Modelin daha önce görmediği verilerle sınanması için kullandığımız kısım.

**Amaç :** Model sadece ezber yapmasın, genelleme yapabilsin.

**Örnek :**
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

df = pd.read_csv("data (1).csv")
X = df[["Weight","Volume"]]
y = df["CO2"]

# Veriyi eğitim ve test olarak ayırma
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Model oluşturma ve eğitme
regr = LinearRegression()
regr.fit(X_train,y_train)

# R² skorunu hesaplama ve yazdırma
y_pred = regr.predict(X_test) 
score = r2_score(y_test,y_pred)
print("r2 score: ",score)

# Gerçek ve tahmini değerleri yan yana göstermek
new_df = pd.DataFrame({"Gerçek CO2":y_test,"Tahmini CO2":y_pred})
print(new_df)
```
**Çıktı :**
```python
r2 score:  0.4100871476339105

    Gerçek CO2  Tahmini CO2
35         120   106.690781
13          94   101.322166
26         104   104.965287
30         115   106.276007
16          99   102.136459
31         117   106.810057
21          99   104.518507
12          99    97.421216
8           98    99.831293
17         104   104.416030
9           99   100.587141
```
> Tahminler kötü çünkü veri az.

## 6- Karışıklık Matrisi (Confusion Matrix)
Karışıklık matrisi, sınıflandırma modellerinin tahminleri ile gerçek etiketler arasındaki karşılaştırmayı hücresel olarak gösteren bir tablodur. Her satır gerçek sınıfı, her sütun modelin tahmin ettiği sınıfı gösterir. Bu sayede modelin hangi sınıfları karıştırdığını, hangi sınıflarda iyi veya kötü olduğunu rahatça görebilirsiniz.

**Binom (binary) durumda hücrelerin yorumu**

Tipik düzen (scikit-learn confusion_matrix çıktısı):

     [[TN, FP],
      [FN, TP]]
- TN (True Negative): Gerçek negatif ve tahmin negatif
- FP (False Positive): Gerçek negatif ama tahmin pozitif (yanlış alarm)
- FN (False Negative): Gerçek pozitif ama tahmin negatif (kaçırma)
- TP (True Positive): Gerçek pozitif ve tahmin pozitif

**Bu hücrelerden şu metrikler hesaplanır :**

- Accuracy = (TP+TN) / toplam
- Precision = TP / (TP+FP) — pozitif tahminlerin doğruluğu
- Recall (Sensitivity) = TP / (TP+FN) — gerçek pozitiflerin yakalanma oranı
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
```python
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

actual = rng.binomial(1,0.9,size=1000) # data
predicted = rng.binomial(1,0.9,size=1000) # model tahmini örneği

Accuary = metrics.accuracy_score(actual,predicted)

Precision = metrics.precision_score(actual,predicted)

Sensitivity = metrics.recall_score(actual,predicted)

Specificity = metrics.recall_score(actual,predicted,pos_label=0)

F1_score = metrics.f1_score(actual,predicted)

print("Accuary: ",Accuary)
print("Precision: ",Precision)
print("Sensitivity: ",Sensitivity)
print("Specificity: ",Specificity)
print("F1_score: ",F1_score)
```

## 7- Grid Search
Grid Search (ızgara arama), bir modelin performansını artırmak için hiperparametrelerin en iyi kombinasyonunu bulmayı amaçlar.
- Hiperparametre: Model eğitimi sırasında kullanıcı tarafından belirlenen değerler (ör. C veya kernel SVM’de).

- Grid Search, önceden belirlenmiş tüm hiperparametre kombinasyonlarını dener ve en iyi performans veren kombinasyonu seçer. 

**Nasıl çalışır?**
1) Önce denenecek parametrelerin bir “ızgarası” tanımlanır.
2) GridSearch, tüm kombinasyonları dener
3) Her kombinasyon, genellikle cross-validation (k-katlı doğrulama) ile test edilir.
4) En iyi performans sağlayan kombinasyon seçilir.

**Örnek :**
```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

df = datasets.load_iris()
x = df["data"]
y = df["target"]

logit = LogisticRegression(max_iter=10000)
C = [0.25, 0.5, 0.75, 1, 1.25, 1.50, 1.75, 2]

scores = []

for choice in C:
    logit.set_params(C=choice)
    logit.fit(x,y)
    scores.append(logit.score(x,y))

print(scores)
```
## 8- Kategorik Veriler
Makine öğrenmesinde kategorik veriler, sayısal olmayan ama belirli kategorileri temsil eden verilerdir. Bu veriler, modelin doğrudan anlamlandıramadığı metinsel veya sınıflandırılmış bilgilerdir, bu yüzden genellikle sayısal forma dönüştürülmeleri gerekir.

**a. Label Encoding (Etiketleme Kodlama)**
- Her kategoriye bir sayı atanır.
- Kırmızı = 0 , Mavi = 1
- 📉 Dezavantaj: Model “2 > 1 > 0” gibi yanlış sıralama ilişkisi kurabilir.
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
colors = ["red","blue","green","blue"]
encoded = encoder.fit_transform(colors)
print(encoded)

çıktı : 
[2 0 1 0]
```
**b. One-Hot Encoding**
- Her kategori için ayrı bir sütun oluşturur (binary 0/1 değerleriyle).
- 📈 Avantaj: Kategoriler arasında sıralama ilişkisi kurmaz.
- 📉 Dezavantaj: Çok fazla kategori varsa, sütun sayısı artar (boyut patlaması yaşanabilir).
```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

encoder = OneHotEncoder(sparse_output=False)

colors = np.array(["gray","yellow","black","yellow"]).reshape(-1,1)
encoded = encoder.fit_transform(colors)
print(encoded)

çıktı :

[[0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]]
```
**Örnek Uygulama :** 
```python
from sklearn import linear_model
import pandas as pd

# veri çekip get_dummies ile OneHotEncoder yaptık
df = pd.read_csv("data (1).csv")
result = pd.get_dummies(df[["Car"]],dtype=int)

# Car sütununu Volume ve Weight ile birleştirdik
x = pd.concat([df[["Volume","Weight"]],result], axis=1)
y = df["CO2"]

# modeli eğittik
regr = linear_model.LinearRegression()
regr.fit(x,y)

# tahmin etmesini istediğimiz arabanın sırayla Volume, Weight ve Car değerlerini girdik
# her bir araba markası sütun olduğu için istediğimiz markaya 1 verdik diğerleri 0
predictedCO2 = regr.predict([[2300,1300,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])

print(predictedCO2)

çıktı :
# Tahmin edilen CO2 emisyonu
[122.45153299]
```

## 9- Çapraz Doğrulama (Cross-Validation)
Makine öğrenmesinde çapraz doğrulama (cross-validation), bir modelin genelleme yeteneğini yani yeni veriler üzerindeki performansını değerlendirmek için kullanılan bir yöntemdir.
Basitçe, veriyi birkaç parçaya ayırıp modelin farklı bölümler üzerinde eğitilip test edilmesini sağlar.

---
🔹 **Neden çapraz doğrulama kullanılır?**

Bir modeli sadece tek bir train-test ayrımı ile değerlendirirsek, sonuç tesadüfi bir veri bölünmesine bağlı olabilir.
Örneğin:

- Train verisinde model çok iyi performans gösterir ama test verisinde kötü olabilir.

- Veya tam tersi, şans eseri iyi sonuç alabiliriz.

Çapraz doğrulama bu rastgeleliği azaltır ve daha güvenilir bir performans tahmini sunar.

---
🔹 **En yaygın yöntem: K-Katlı Çapraz Doğrulama (K-Fold Cross Validation)**

**Adımlar:**

- 1- Veri kümesi K eşit parçaya (fold’a) bölünür.

- 2- Her adımda:

  - K-1 parça eğitim (train) için,

  - 1 parça test (validation) için kullanılır.

- 3- Bu işlem K kez tekrarlanır, her seferinde farklı bir parça test verisi olur.
- 4- Her bir denemenin doğruluk (accuracy, RMSE, F1, vb.) sonucu alınır.
- 5- Sonuçlar ortalaması, modelin genel performansını verir.
---

🔹 **Özel Türleri**
| Tür                       | Açıklama                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Stratified K-Fold**     | Sınıflar dengesizse, her fold’da sınıf oranları korunur (özellikle classification’da).                  |
| **Leave-One-Out (LOOCV)** | K = veri sayısı. Her adımda bir gözlem test için ayrılır. Çok maliyetli ama küçük verilerde kullanılır. |
| **ShuffleSplit**          | Veriyi rastgele train-test setlerine böler, K kez tekrarlar. Fold büyüklükleri sabit olmayabilir.       |
| **TimeSeriesSplit**       | Zaman serilerinde kullanılır; geçmiş verilerle eğitilir, gelecekle test edilir. Zaman sırası korunur.   |
---

**örnek :**
```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from skleran.datasets import load_iris

x,y = load_iris(return_X_y=True)

model = DecisionTreeClassifier(random_state=42)

k_folds = KFold(n_splits=5)

scores = cross_val_score(model,x,y,cv=k_folds)

print(f"Cross Validation Scores: {scores}")
print(f"Average CV scores: {scores.mean()}")
print(f"Number of CV scores used in Average: {len(scores)}")
```

---
# Makinenin öğrenme biçimleri
## Doğrusal Regresyon

- Doğrusal regresyon, bir bağımlı değişkeni (y) bir veya daha fazla bağımsız değişken (x) ile doğrusal bir ilişki kurarak tahmin etmeye çalışır.

**Matematiksel modeli:**

>   y = ax + b 
- a (intercept): Doğrunun y-eksenini kestiği nokta.

- b (slope): Doğrunun eğimi, yani x değiştiğinde y’nin nasıl değiştiğini gösterir.

Örneği bir arabanın yaşı artarsa hızı düşer mi ? Sorusunu doğrusal regresyon ile inceleyebilriz.

**Örnek :**
```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

x = [5,7,8,7,2,19,2,9,4,11,12,9,6]
y = [99,86,87,88,112,86,101,87,94,78,77,85,86]

a,b,r,p,std_err = stats.linregress(x,y)

def myFunc(x):
    return a*x+b

my_model = list(map(myFunc,x))
plt.scatter(x,y)
plt.plot(x,my_model)

plt.show()
```
- a = fonksiyon eğimi

- b = fonksiyon sabiti
- r = korelasyon katsayısıdır. İki değişken arasındaki doğrusal ilişkinin gücünü ve yönünü gösteren bir katsayıdır. Değeri -1 ile +1 arasında değişir. 0 çıkarsa aralarında hiçbir ilişki yok deriz. Eğer +1 çıkarsa aralarında mükemmel pozitif ilişki var deriz yani ikisi de birlikte artıp veya birlikte azalır anlamına gelir. -1 çıkarsa aralarında mükemmel bir negatif ilişki var deriz, mesela biri artarsa biri azalır.
- p = bu değer 0.05 in altında çıkarsa bu fonksiyon güvenilirdir yani x ile y arasında anlamlı bir ilişki var anlamına gelir. Üstünde çıkarsa ürettiği değerler yanlış çıkabilir.
- std_err = regresyonda bulunan eğimin (a) ne kadar güvenilir olduğunu gösterir.

-   | Durum                              | Ne anlama gelir                                        |
    | ---------------------------------- | ------------------------------------------------------ |
    | `std_err` ≈ 0                      | Eğim çok güvenilir — veri neredeyse mükemmel doğrusal. |
    | Küçük (`<1` veya çok küçük)        | Güvenilir model (veri ölçeğine bağlı olarak).          |
    | Büyük (`>1` veya `>> eğim değeri`) | Gürültü yüksek, eğim kararsız, model zayıf.            |

## Polinom Regresyonu

Doğrusal regresyon (Linear Regression), veriler arasındaki ilişkiyi doğru (lineer) bir denklem ile modellemeye çalışır:


$$
y = b_0 + b_1 x
$$

Ancak bazı veriler doğrusal olmayan (non-linear) ilişkilere sahiptir.
Örneğin, fiyat – yaş, hız – mesafe gibi ilişkiler bir doğruyla iyi açıklanmaz.
Bu durumda polinom regresyon kullanılır:

$$
y = b_0 + b_1 x + b_2 x^2 + b_3 x^3 + \dots + b_n x^n
$$


Yani doğrusal regresyonun üzerine x’in üstlü (kuvvetli) terimlerini ekleyerek daha esnek bir model kuruyoruz.

**Örnek :**
```python
import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

my_model = np.poly1d(np.polyfit(x,y,3))

my_line = np.linspace(1,22,100)

plt.scatter(x,y)
plt.plot(my_line,(my_model(my_line)))

plt.show()
```
**numpy.polyfit :** Verilere en iyi uyan polinomun katsayılarını bulan bir fonksiyondur.
Yani elimizde bazı noktalar (x, y) varsa, bu noktaları yakından geçen bir polinom fonksiyon bulmaya çalışır.

numpy.polyfit(x, y, deg) 
- x → Bağımsız değişken (liste veya numpy array)
- y → Bağımlı değişken (liste veya numpy array)
- deg → Polinomun derecesi (kaçıncı dereceden olacağı)
---
**numpy.poly1d :** polinomlarla çalışmayı kolaylaştıran bir sınıftır.
Elinde polinomun katsayıları varsa, np.poly1d ile bunu fonksiyon gibi kullanabilirsin.

numpy.poly1d(c)
- c → Polinom katsayıları (liste veya array).
- Katsayılar yüksek dereceden başlayarak verilmelidir.

> Yani np.polyfit polinomun katsayılarını bulur,
np.poly1d ise bu katsayıları alıp polinom nesnesi yapar.

## r2_score (determinasyon katsayısı)

- modelin tahminlerinin gerçek değerlere ne kadar uyduğunu ölçen bir metriktir.
- sana modelin veriye uyum derecesini söyler.

**Sonuç:**
- 1’e yakınsa → model çok iyi uyuyor.
- 0’a yakınsa → model çok kötü.
- Negatifse → model, verinin ortalamasını almaktan bile kötü.

**Örnek :**
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

my_model = np.poly1d(np.polyfit(x,y,3))

print(r2_score(y,my_model(x)))
```
**çıktı :**
```python
0.9432150416451026
```
- çıktı 1’e çok yakın yani modelin tahminlerini gerçek değerlere çok iyi uyuyor.

## Çoklu Regresyon (Multiple Regression)

Çoklu regresyon (multiple regression), bir bağımlı değişkeni (yani sonucu) **birden fazla bağımsız değişken kullanarak tahmin etmek** için kullanılan istatistiksel bir yöntemdir.

- Bağımlı değişken (Y): Tahmin etmek istediğimiz değişken.
Örnek: Ev fiyatı.

- Bağımsız değişkenler (X1, X2, X3…): Bağımlı değişkeni etkileyebilecek faktörler.
Örnek: Ev büyüklüğü, oda sayısı, semt, yaş gibi.

**Çoklu regresyon denklemi şöyle yazılır:**
$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \varepsilon
$$

**Açıklamalar:**

- $Y$ = Bağımlı değişken
- $X_1, X_2, \dots, X_n$ = Bağımsız değişkenler
- $\beta_0$ = Y kesiti (intercept)
- $\beta_1, \beta_2, \dots, \beta_n$ = Bağımsız değişkenlerin katsayıları
- $\varepsilon$ = Hata terimi (modelin tahmin edemediği kısmı)

**Örnek :**
```python
from sklearn import linear_model
import pandas as pd

# dosya okuma
df = pd.read_csv("data (1).csv")

# değer atama
x = df[["Volume","Weight"]]
y = df["CO2"]

# makine eğitimi
regr = linear_model.LinearRegression()
regr.fit(x,y)

# kendi değerini verip tahmin yaptırma
new_data = pd.DataFrame([[3300,1300]],columns=["Volume","Weight"])
result = regr.predict(new_data)
print(result)
```
## Karar ağacı (Decision Tree)
Sınıflandırma ve regresyon problemlerinde kullanılan denetimli bir makine öğrenmesi algoritmasıdır. Veri setini, özelliklerin değerlerine göre dallara ayırarak karar kuralları üretir. Sonuçta kökten yapraklara doğru ilerleyerek tahmin yapılır.

**Karar Ağacı Mantığı**

- Kök düğüm (root node): Verinin ilk bölündüğü yerdir.

- Dallar (branches): Kararlara göre verinin ayrıldığı yollardır.

- Yaprak düğümler (leaf nodes): Nihai karar veya sınıflandırmadır.

Bir soru gibi çalışır :

👉 “Özellik A şu değerden küçük mü?” → Evetse sol dal, hayırsa sağ dal.
Bu süreç devam eder, en sonunda sınıflandırma ya da tahmin yapılır.

**Örnek :**

- Bu kodda kullanılan datada bir komedyenin özelliklerine göre onun gösterisine gidip gitmediğimiz var. W3School'un decision tree bölümünden. 
```python
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

df = pd.read_csv("decision_tree.csv")

# datadaki string verileri sayıya dönüştürüyoruz (Makinenin anlayacağı dilden konuşuyoruz).
dN = {"UK":0,"USA":1,"TR":2}
df["Nationality"] = df["Nationality"].map(dN)
dGo = {"YES":1,"NO":0}
df["Go"] = df["Go"].map(dGo)

# x ve y değerlerini tanımlayalım
X = df[["Age","Experience","Rank","Nationality"]]
y = df["Go"]

# makinenin öğrenme tipini seçelim ve öğretelim
dtree = DecisionTreeClassifier()
dtree.fit(X,y)

# kendimiz değer verip makineden tahmin yapmasını isteyelim
new_data = pd.DataFrame([[40,10,7,1]],columns=["Age","Experience","Rank","Nationality"])
result = dtree.predict(new_data)
print(result)
```
> Bu kod çıktı olarak [0] ya da [1] üretecek yani gitmem veya giderim.

## Hiyerarşik kümeleme (Hierarchical Clustering)
Verileri kümelere ayırmak için kullanılan bir kümeleme (clustering) yöntemidir. Bu yöntemde amaç, benzer özelliklere sahip verileri ağaç benzeri bir yapı (dendrogram) şeklinde gruplayarak aralarındaki ilişkileri göstermektir.

- Hiyerarşik kümeleme, gözlemler arasındaki benzerlikleri (örneğin mesafeleri) kullanarak bir küme yapısı (hiyerarşi) oluşturur.
- Sonuçta, hangi verilerin birbirine daha yakın olduğunu gösteren bir ağaç (dendrogram) elde ederiz.

**Bu ağaç sayesinde :**
- Kaç küme olacağına sen karar verebilirsin (ağacı belli bir yükseklikten keserek).
- Kümeleme ilişkilerini görsel olarak inceleyebilirsin.

🧭 **İki türü vardır :**

---
- **a) Birleştirici (Agglomerative)** – En çok kullanılan

  - En küçük birimden (tek tek noktalardan) başlar.
  - En yakın iki noktayı birleştirir → sonra bu yeni kümeyi diğerlerine göre yeniden değerlendirir.
  - Bu işlem, tüm noktalar tek bir kümede toplanana kadar sürer.

- **b) Bölücü (Divisive)**

  - Tüm veriler tek bir küme olarak başlar.
  - En uzak noktaları birbirinden ayırarak kümeleri bölmeye başlar.
  - Sonunda her nokta kendi başına kalır.

🧮 **Benzerlik Nasıl Ölçülür?**

---
Veriler arasındaki uzaklık (distance) genellikle şu yöntemle ölçülür:

- **Öklid mesafesi (Euclidean distance)**
  - Öklid mesafesi, iki nokta arasındaki doğrusal uzaklığı ölçer.
Yani düz bir çizgiyle bir noktadan diğerine olan mesafedir.
Matematikte Pisagor teoremine dayanır.

**Formülü :**

İki nokta olsun:
A(x₁, y₁) ve B(x₂, y₂)
$$
d(A, B) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

🔗 **Kümeler Arasındaki Uzaklık (Bağlantı / Linkage)**

---
İki küme birleştirilirken aralarındaki mesafenin nasıl ölçüleceğini belirleyen bağlantı yöntemi seçilir:

- **Ward’s Method**
  - Kümeler birleşince varyans artışı en az olacak şekilde birleştirir
  - Genellikle en başarılı yöntem
  
**Formülü:**
$$
\Delta E_{AB} = \frac{n_A \cdot n_B}{n_A + n_B} \times \|\bar{x}_A - \bar{x}_B\|^2
$$

- **nA :** Küme A’daki gözlem (veri noktası) sayısı  
- **nB :** Küme B’deki gözlem (veri noktası) sayısı  
- **x̄A :** Küme A’nın ortalama (merkez) vektörü  
- **x̄B :** Küme B’nin ortalama (merkez) vektörü  
- **||x̄A − x̄B|| :** İki kümenin merkezleri arasındaki Öklid mesafesi  
- **ΔEAB :** İki küme birleştirildiğinde toplam varyanstaki artış miktarı

**Örnek :**
- **Hiyerarşik kümeleme grafiği :**
```python
from scipy.cluster.hierarchy import dendrogram,linkage
import matplotlib.pyplot as plt
import numpy as np

x = np.array([4,5,10,4,3,11,14,6,10,12])
y = np.array([21,19,24,17,16,25,24,22,21,21])

data = list(zip(x,y))
l_data = linkage(data,method="ward",metric="euclidean")
dendrogram(l_data)
plt.show()
```
- **AgglomerativeClustering kullanımı :**
```python
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np

x = np.array([4,5,10,4,3,11,14,6,10,12])
y = np.array([21,19,24,17,16,25,24,22,21,21])

data = list(zip(x,y))
l_data = AgglomerativeClustering(n_clusters=2,linkage="ward")
labels = l_data.fit_predict(data)

plt.scatter(x,y,c=labels)
plt.show()
```

## Lojistik regresyon
Lojistik regresyon, bir olayın olma olasılığını tahmin eder.
Çıktı sürekli değil, 0 veya 1 (örneğin: hasta / sağlıklı, evet / hayır) gibi sınıflar olur.

**Temel Formül :**
$$
P(y=1 \mid x) = \frac{1}{1 + e^{-(b_0 + b_1 x_1 + b_2 x_2 + \dots + b_n x_n)}}
$$
- Bu formül, elimizdeki bir girdiye (özelliklere, x) bakarak, olayın gerçekleşme olasılığını (y = 1) hesaplar.
- “Bu özelliklere sahip bir örnek, 1 sınıfına (örneğin ‘hasta’, ‘geçti’, ‘evet’) ait olma olasılığı nedir?”
sorusunun cevabını verir.

**Örnek :** 
- Formülü kullanarak kanser olma olasılıklarını hesapladık.
```python
from sklearn import linear_model
import numpy as np

X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

def logitP(logr,X):
  log_odds = logr.coef_ * X + logr.intercept_
  odds = np.exp(log_odds)
  probability = odds / (1 + odds)
  return probability

print(logitP(logr,X))
```
**Çıktı :**
```python
[[0.60749168]
 [0.19267555]
 [0.12774788]
 [0.00955056]
 [0.08037781]
 [0.0734485 ]
 [0.88362857]
 [0.77901203]
 [0.88924534]
 [0.81293431]
 [0.57718238]
 [0.96664398]]
```
- Yine aynı fomülü kullandık ama bu sefer formülü kendimiz yazmadık. **predict_proba** modülüyle yaptık.
```python
from sklearn import linear_model
import numpy as np

X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Öğrenme
logr = linear_model.LogisticRegression()
logr.fit(X,y)

# kanser olma olasılığı hesaplanacak hastanın tümor büyüklüğü
yeni_hasta = np.array([[3.78]])

# tahmin
result = logr.predict_proba(yeni_hasta)[0,1]
print(result)
```
**Çıktı :**
```python
0.6074916769842938
```

## K-Means
makine öğrenmesinde kullanılan en popüler kümeleme (clustering) yöntemlerinden biridir. Gözetimsiz (unsupervised) öğrenme kategorisine girer; yani verilerin etiketleri (örneğin sınıf bilgisi) yoktur.

- K-Means, veriyi K tane kümeye (gruba) ayıran bir algoritmadır.
- Amaç, her küme içindeki verilerin birbirine benzer, farklı kümelerdeki verilerin ise farklı olmasını sağlamaktır.

**Dirsek (Elbow) yöntemi ile K değeri bulma (n_clusters) :**
```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4,5,10,4,3,11,14,6,10,12]
y = [21,19,24,17,16,25,24,22,21,21]

data = list(zip(x,y))
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11),inertias,marker="o")
plt.title("Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()
```
> Bulunan K değeri : 2

**Bulunan K değerini uygulama :**
```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4,5,10,4,3,11,14,6,10,12]
y = [21,19,24,17,16,25,24,22,21,21]

data = list(zip(x,y))

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(x,y,c=kmeans.label_)
plt.show()
```

## K En yakın Komşu (K-nearest neighbors)
Makine öğrenmesinde K En Yakın Komşu (K-Nearest Neighbors, KNN) algoritması, denetimli (supervised) öğrenme yöntemlerinden biridir ve hem sınıflandırma (classification) hem de regresyon (regression) problemlerinde kullanılabilir.

**KNN’nin temel mantığı şudur :**

- “Bir örneğin sınıfı, ona en yakın komşularının çoğunlukta olduğu sınıfa aittir.”

Yani, yeni bir verinin etiketini tahmin etmek istediğimizde:

- **1-** Eğitim verisindeki tüm noktalarla arasındaki mesafeyi hesaplarız.

- **2-** Bu mesafelere göre en yakın K komşuyu seçeriz.

- **3-** Bu K komşunun etiketlerine bakarak tahmin yaparız :
  - **Sınıflandırma** için: Komşular arasında en çok görülen sınıf seçilir (çoğunluk oylaması).

  - **Regresyon** için: Komşuların ortalaması alınır.

🔢 **Adım Adım KNN Algoritması :**

Örneğin sınıflandırma için :

- **1-** K değeri belirlenir (örneğin K = 3).
- **2-** Test verisi ile tüm eğitim verisi arasındaki mesafe hesaplanır.(Öklid ile)
- **3-** En küçük mesafeye sahip K komşu seçilir.
- **4-** Bu komşuların etiketleri sayılır.
- **5-** En sık görülen sınıf etiketi yeni veriye atanır.

⚙️ **Önemli Parametreler ve Dikkat Edilmesi Gerekenler**

| Konu                        | Açıklama                                                                                                                                                            |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **K değeri**                | Küçük K → gürültüye duyarlı, aşırı öğrenme riski.<br>Büyük K → daha düzgün karar sınırları ama daha az esneklik. Genelde K, √n civarında seçilir (n = veri sayısı). |
| **Mesafe metriği**          | Öklid, Manhattan, Minkowski, Kosinüs benzerliği vs.                                                                                                                 |
| **Özellik ölçeklendirmesi** | KNN’de mesafeler önemli olduğu için, özellikler aynı ölçekte olmalı (**standardizasyon veya normalizasyon** yapılmalı).                                             |
| **Ağırlıklı oylama**        | Bazı varyantlarda, yakın komşulara daha fazla ağırlık verilir (örneğin mesafeyle ters orantılı olarak).                                                             |

**Örnek :**
```python
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

x = [4,5,10,4,3,11,14,8,10,12]
y = [21,19,24,17,16,25,24,22,21,21]

classes = [0,0,1,0,0,1,1,0,1,1]

data = list(zip(x,y))

model = KNeighborsClassifier(n_neighbors=1)
model.fit(data,classes)

new_x = 8
new_y = 21
new_point = [(new_x,new_y)]

prediction = model.predict(new_point)
print(prediction) # [0] mı [1] mi diye tahmin eder

#tahminin data ile birlikte görselleştirilmesi
plt.scatter(x+[new_x],y+[new_y],c=classes+[prediction[0]])
plt.show()
```
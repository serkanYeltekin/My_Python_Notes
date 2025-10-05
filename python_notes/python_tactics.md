### Eğer bir listede örneğin; notlara göre isim çekmek istiyorsan.
``` python
names = []
scores = []
for _ in range(int(input())):
    name = input()
    score = float(input())
    names.append(name)
    scores.append(score)
    
mylist = list(zip(names,scores))
second=sorted(set(scores))[1]
lastnames=[name for name,score in mylist if second==score] # bu satır
lastnames.sort()
for i in lastnames:
    print(i)
```

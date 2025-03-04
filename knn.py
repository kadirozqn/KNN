import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv")
df.head()
df.shape
df.describe()
df.drop(["Unnamed: 32", "id"],axis=1,inplace= True)
df.head()
M = df[df.diagnosis == "M"] # kotu huylu tumor
B = df[df.diagnosis == "B"] # iyi huylu tumor
M.info()
#scatter
plt.scatter(M.radius_mean, M.area_mean,color = "red", label = "kotu")
plt.scatter(B.radius_mean, B.area_mean,color = "green", label = "iyi")
plt.legend()
plt.show()
#daha anlamlı iki feature kullanalım
plt.scatter(M.radius_mean, M.texture_mean,color = "red", label = "kotu")
plt.scatter(B.radius_mean, B.texture_mean,color = "green", label = "iyi")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()
plt.scatter(M.radius_mean, M.perimeter_mean,color = "red", label = "kotu")
plt.scatter(B.radius_mean, B.perimeter_mean,color = "green", label = "iyi")
plt.xlabel("radius_mean")
plt.ylabel("perimeter_mean")
plt.legend()
plt.show()
#daha anlamlı iki feature kullanalım
plt.scatter(M.radius_mean, M.texture_mean,color = "red", label = "kotu")
plt.scatter(B.radius_mean, B.texture_mean,color = "green", label = "iyi")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()
df.diagnosis =[1 if each =="M" else 0 for each in df.diagnosis] 
print(df.info())# string olan verıler integer'a cevrildi
y = df.diagnosis.values #  labe/class
x_data = df.drop(["diagnosis"],axis = 1) #features

#normalization işlemi 

# x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values //bu satırda 'float' object has no attribute 'values' hatası alabılırsınız. Bunun için aşağıdaki yöntemi kullanabilirsiniz.

# Yöntem 1:
#  Minimum ve maksimum değerleri aynı olan sütunları bul
constant_columns = x_data.columns[x_data.min() == x_data.max()]
print("Sabit sütunlar:", constant_columns)

# Eğer sabit sütunlar varsa, onları düşür
x_data = x_data.drop(columns=constant_columns)

# Tekrar normalize et
x = (x_data - x_data.min()) / (x_data.max() - x_data.min())

print(x.describe())  # Normalizasyonun başarılı olup olmadığını doğrula

# Yöntem 2: 
#x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
#train-test split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
x_train = np.ascontiguousarray(x_train)
x_test = np.ascontiguousarray(x_test)

#knn model
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors= 3)
knn_model.fit(x_train,y_train)
prediction = knn_model.predict(x_test) 
print("predict degeri:" ,prediction)
#score
print("{}nn score: {}".format(3,knn_model.score(x_test,y_test)))

#peki en iyi k degerini nasil bulacagiz? Elbette for dongusu bıze yardımcı olacak
for each in range(1,20):
    knn2_model = KNeighborsClassifier(n_neighbors= each)
    knn2_model.fit(x_train,y_train)
    prediction = knn_model.predict(x_test)
    print("{}nn score: {}".format(each,knn2_model.score(x_test,y_test)))

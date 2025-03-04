# KNN
# README: K-Nearest Neighbors (KNN) Model for Tumor Classification

## Proje Açıklaması
Bu proje,daha önceki çalışmamda kullanmış olduğum bir veri kümesindeki tümör verilerini analiz ederek iyi huylu (**B - Benign**) ve kötü huylu (**M - Malignant**) tümörleri sınıflandırmayı amaçlamaktadır. **K-Nearest Neighbors (KNN)** algoritması kullanılarak sınıflandırma gerçekleştirilmiştir.

## Kullanılan Kütüphaneler
Projede aşağıdaki Python kütüphaneleri kullanılmıştır:
- **pandas**: Veri işleme ve manipülasyon için
- **numpy**: Matematiksel işlemler için
- **matplotlib**: Görselleştirme için
- **sklearn (scikit-learn)**: Makine öğrenmesi modelleme için

## Veri Kümesi ve Ön İşleme
- Veri **data.csv** dosyasından `pandas.read_csv()` ile yüklenmiştir.
- Gereksiz sütunlar (`Unnamed: 32`, `id`) veri kümesinden kaldırılmıştır.
- `diagnosis` sütunu **M:1, B:0** olacak şekilde **sayısal** formata çevrilmiştir.
- Özellikler (`features`) ve etiketler (`labels`) ayrılmıştır.
- Verilerin ölçeklenmesi için **Min-Max Normalizasyonu** uygulanmıştır.

## Veri Görselleştirme
- `radius_mean` ve `area_mean` özelliklerine göre **scatter plot** çizilmiştir.
- Daha anlamlı özellikler seçilerek görselleştirmeler yapılmıştır.
  - `radius_mean` vs `texture_mean`
  - `radius_mean` vs `perimeter_mean`

## Model Eğitimi ve Testi
- Veri, %70 eğitim ve %30 test olacak şekilde `train_test_split()` ile bölünmüştür.
- **K-Nearest Neighbors (KNN) Algoritması** uygulanmıştır:
  - `n_neighbors = 3` ile model eğitilmiş ve test edilmiştir.
  - Modelin başarı oranı **score()** fonksiyonu ile hesaplanmıştır.
  - Farklı **k değerleri** (1'den 19'a kadar) denenerek en iyi **k değeri** bulunmaya çalışılmıştır.

## Çalıştırma Adımları
1. Gerekli kütüphaneleri yükleyin:
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.neighbors import KNeighborsClassifier
   ```
2. **Veriyi yükleyin** ve ön işleme adımlarını uygulayın.
3. **Scatter plot** kullanarak görselleştirmeleri oluşturun.
4. **Veriyi normalize edin** ve eğitim-test kümesi olarak ayırın.
5. **KNN modelini eğitin ve test edin**.
6. **Farklı k değerleri** ile modelin performansını kıyaslayın.

## Sonuç ve Değerlendirme
- **KNN modeli, uygun k değeri seçildiğinde yüksek başarı gösterir.**
- K değeri çok küçük seçildiğinde **overfitting** (aşırı öğrenme) olabilir.
- K değeri çok büyük seçildiğinde **underfitting** (yetersiz öğrenme) olabilir.
- En uygun k değeri, veri setinin yapısına göre test edilerek bulunmalıdır.

## Notlar
- Veri kümesi eksik veya hatalı olabilir, bu yüzden her zaman veriyi ön işlemek gereklidir.
- `np.ascontiguousarray()` kullanımı, sklearn'un beklediği formatta veri sağlamak için eklenmiştir.
- Alternatif modeller (SVM, Decision Tree, Random Forest vb.) ile kıyaslanarak farklı sonuçlar elde edilebilir.

Bu proje, makine öğrenmesi konusuna giriş yapmak isteyenler için güzel bir başlangıç olacaktır. Ayrıca kaynak olarak DataI Kaggle sayfasını da
kaynak olarak kullandığımı belirtmek isterim. İlgili sayfaya https://www.kaggle.com/kanncaa1 linki ile ulaşabilirsiniz. 
İyi çalışmalar... :)

Yazar: [Kadir Özan]
Tarih: [5.03.2025]


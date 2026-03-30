# YZM304 Derin Öğrenme Projesi

## Heart Failure Clinical Records Veri Seti ile Binary Classification

---

## Introduction

Bu projede, kalp yetmezliği klinik kayıtları veri seti kullanılarak hastaların ölüm olayı (`DEATH_EVENT`) yaşayıp yaşamayacağını tahmin eden bir derin öğrenme sınıflandırma sistemi geliştirilmiştir. Problem, ikili sınıflandırma (binary classification) kapsamında ele alınmıştır.

Çalışmanın temel amacı, 13.03.2026 tarihinde laboratuvar saatlerinde uygulanan tek gizli katmanlı yapay sinir ağı modelini yeni bir veri seti üzerinde yeniden uygulamak, modelin performansını değerlendirmek ve daha sonra aynı problem üzerinde kütüphane tabanlı modeller ile karşılaştırmalı deneyler gerçekleştirmektir.

Bu kapsamda çalışma iki ana bölümden oluşmaktadır:

1. **Laboratuvar Temel Modeli (Elle Yazılmış Neural Network)**  
   NumPy kullanılarak sıfırdan yazılmış tek gizli katmanlı yapay sinir ağı modeli.

2. **Kütüphane Tabanlı Modeller (Scikit-learn MLPClassifier)**  
   Aynı problem için önce temel bir model, ardından daha gelişmiş bir mimari ile model iyileştirme çalışmaları gerçekleştirilmiştir.

Çalışma sonunda modeller accuracy, precision, recall, F1-score ve confusion matrix gibi temel ölçüm metrikleri ile değerlendirilmiş, özellikle klinik açıdan kritik olan ölüm vakalarının tahmin başarımı analiz edilmiştir.

---

## Methods

### Veri Seti

Bu çalışmada Kaggle platformundan alınan **Heart Failure Clinical Records** veri seti kullanılmıştır.

Veri seti:
- **299 gözlem**
- **12 giriş değişkeni**
- **1 hedef değişken (`DEATH_EVENT`)**

içermektedir.

Hedef değişken:
- `0` → ölüm olayı yok
- `1` → ölüm olayı var

Bu nedenle çalışma, **binary classification** problemi olarak ele alınmıştır.

---

### Veri Ön İşleme

Veri seti üzerinde aşağıdaki ön işleme adımları uygulanmıştır:

- veri setinin yüklenmesi
- sütun ve boyut incelemesi
- eksik veri kontrolü
- sınıf dağılımı analizi
- temel istatistiksel inceleme

Eksik veri analizi sonucunda veri setinde eksik gözlem bulunmadığı görülmüştür. Bu nedenle eksik veri tamamlama (imputation) işlemi uygulanmamıştır.

Sınıf dağılımı aşağıdaki gibidir:

- `DEATH_EVENT = 0` → 203 örnek
- `DEATH_EVENT = 1` → 96 örnek

Bu durum veri setinin tam dengeli olmadığını göstermektedir. Bu nedenle model değerlendirmesinde yalnızca accuracy değil, precision, recall ve F1-score gibi metrikler de dikkate alınmıştır.

---

### Veri Bölme ve Ölçekleme

Veri seti model değerlendirme sürecinde aşağıdaki şekilde bölünmüştür:

- **Eğitim (Train): %70**
- **Doğrulama (Validation): %15**
- **Test: %15**

Sınıf dağılımının korunması amacıyla `stratified sampling` yöntemi kullanılmıştır.

Ayrıca giriş değişkenlerinin farklı ölçeklerde olması nedeniyle `StandardScaler` ile standardizasyon uygulanmıştır.

Ölçekleme sürecinde:

- `fit` işlemi yalnızca eğitim verisi üzerinde yapılmış,
- doğrulama ve test verilerine yalnızca `transform` uygulanmıştır.

Bu yöntem, veri sızıntısını (data leakage) önlemek için tercih edilmiştir.

---

### Model 1: Laboratuvar Temel Modeli (Elle Yazılmış Neural Network)

Bu model, laboratuvar saatlerinde geliştirilen temel yapay sinir ağı yapısının yeni veri setine uyarlanmış halidir.

Model tamamen **NumPy** kullanılarak sıfırdan oluşturulmuştur.

Kullanılan temel bileşenler:

- `initialize_parameters`
- `forward_propagation`
- `sigmoid`
- `compute_cost`
- `backpropagation`
- `update_parameters`
- `nn_model`
- `predict`

#### Mimari
- Giriş katmanı: **12 özellik**
- Gizli katman: **8 nöron**
- Çıkış katmanı: **1 nöron**
- Gizli katman aktivasyonu: **tanh**
- Çıkış katmanı aktivasyonu: **sigmoid**

#### Eğitim Ayarları
- Optimizasyon mantığı: **Gradient Descent / SGD benzeri güncelleme**
- Öğrenme oranı (`learning_rate`): **0.01**
- İterasyon sayısı (`num_iterations`): **1000**
- Kayıp fonksiyonu: **Binary Cross Entropy Loss**

Bu model, laboratuvarda uygulanan temel mantığın veri seti üzerinde yeniden test edilmesi amacıyla kullanılmıştır.

---

### Model 2: Scikit-learn Temel Modeli

Bu model, aynı problem üzerinde **Scikit-learn `MLPClassifier`** kullanılarak kurulmuştur.

#### Mimari
- 1 gizli katman
- 8 nöron

#### Hiperparametreler
- `hidden_layer_sizes=(8,)`
- `activation='relu'`
- `solver='sgd'`
- `learning_rate_init=0.01`
- `max_iter=500`
- `random_state=42`

Bu model, laboratuvar temel modelinin daha optimize ve kütüphane tabanlı bir versiyonu olarak değerlendirilmiştir.

---

### Model 3: Geliştirilmiş Model

Bu model, temel modelin performansını artırmak amacıyla oluşturulmuştur.

#### Mimari
- 2 gizli katman
- İlk gizli katman: 16 nöron
- İkinci gizli katman: 8 nöron

#### Hiperparametreler
- `hidden_layer_sizes=(16, 8)`
- `activation='relu'`
- `solver='sgd'`
- `learning_rate_init=0.01`
- `max_iter=500`
- `alpha=0.001`
- `random_state=42`

Bu modelde:
- daha fazla model kapasitesi
- ek katman
- **L2 regularization**

kullanılarak overfitting etkisinin azaltılması ve daha iyi genelleme başarımı elde edilmesi hedeflenmiştir.

---

### Değerlendirme Metrikleri

Modeller aşağıdaki ölçüm metrikleri ile değerlendirilmiştir:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Ayrıca eğitim, doğrulama ve test sonuçları birlikte değerlendirilerek overfitting / underfitting analizi yapılmıştır.

---

## Results

### 1) Laboratuvar Temel Modeli Sonuçları

#### Accuracy
- **Train Accuracy:** 0.694
- **Test Accuracy:** 0.678

#### Classification Report (Test)

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| 0 | 0.68 | 1.00 | 0.81 |
| 1 | 0.00 | 0.00 | 0.00 |

#### Confusion Matrix

```text
[[61  0]
 [29  0]]
```

Bu sonuçlar, laboratuvar temel modelinin çoğunluk sınıfa yönelme eğiliminde olduğunu ve pozitif sınıfı (`DEATH_EVENT = 1`) yakalamakta başarısız olduğunu göstermektedir.

---

### 2) Scikit-learn Temel Model Sonuçları

#### Accuracy
- **Train Accuracy:** 0.923
- **Validation Accuracy:** 0.778
- **Test Accuracy:** 0.711

#### Classification Report (Test)

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| 0 | 0.76 | 0.84 | 0.80 |
| 1 | 0.55 | 0.43 | 0.48 |

#### Confusion Matrix

```text
[[26  5]
 [ 8  6]]
```

Bu model, laboratuvar temel modeline göre daha yüksek genel doğruluk ve daha dengeli sınıf performansı göstermiştir.

---

### 3) Geliştirilmiş Model Sonuçları

#### Accuracy
- **Train Accuracy:** 0.938
- **Validation Accuracy:** 0.800
- **Test Accuracy:** 0.711

#### Classification Report (Test)

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| 0 | 0.82 | 0.74 | 0.78 |
| 1 | 0.53 | 0.64 | 0.58 |

#### Confusion Matrix

```text
[[23  8]
 [ 5  9]]
```

Bu model, genel doğruluk açısından temel model ile benzer sonuç vermesine rağmen, pozitif sınıf olan ölüm olaylarını tahmin etmede daha yüksek recall ve F1-score üretmiştir.

---

### Model Karşılaştırma Tablosu

| Model | Train Accuracy | Validation Accuracy | Test Accuracy |
|------|---------------|--------------------|--------------|
| Laboratuvar Temel Modeli | 0.694 | - | 0.678 |
| Scikit-learn Temel Model | 0.923 | 0.778 | 0.711 |
| Geliştirilmiş Model | 0.938 | 0.800 | 0.711 |

---

## Discussion

Bu çalışmada, kalp yetmezliği klinik kayıtları veri seti üzerinde üç farklı yapay sinir ağı yaklaşımı değerlendirilmiştir.

İlk olarak laboratuvar ortamında geliştirilen temel model, yeni veri seti üzerinde yeniden uygulanmıştır. Bu model veri seti üzerinde çalıştırılmış olsa da, sınıf 1 tahmininde başarısız kalmış ve çoğunluk sınıfa yönelmiştir. Bu durum, basit yapının veri setinin karmaşıklığını yeterince temsil edemediğini göstermektedir.

Daha sonra aynı problem, Scikit-learn tabanlı bir yapay sinir ağı modeli ile tekrar ele alınmıştır. Bu model, hem genel doğruluk hem de sınıf bazlı performans açısından laboratuvar modeline göre belirgin iyileşme göstermiştir.

Son aşamada, model kapasitesini artırmak amacıyla daha derin bir mimari ve regularization içeren geliştirilmiş model oluşturulmuştur. Her ne kadar test doğruluğu temel model ile aynı kalmış olsa da, özellikle klinik açıdan daha kritik olan ölüm vakalarının tahmininde daha yüksek recall ve F1-score elde edilmiştir.

Bu sonuçlar, model değerlendirmesinde yalnızca accuracy değerine bakmanın yeterli olmadığını göstermektedir. Özellikle sağlık verileri gibi kritik alanlarda, pozitif sınıfın doğru yakalanması daha büyük önem taşımaktadır.

Bu nedenle nihai model seçiminde yalnızca test doğruluğu değil, aynı zamanda sınıf 1 için recall ve F1-score değerleri de dikkate alınmıştır. Bu bağlamda **geliştirilmiş model nihai model olarak seçilmiştir**.

---

## Project Files

Proje aşağıdaki dosyalardan oluşmaktadır:

```text
YZM304_Proje/
├── heart_failure_clinical_records_dataset.csv
├── lab_model.ipynb
├── proje.ipynb
└── README.md
```

### Dosya Açıklamaları

- **`lab_model.ipynb`**  
  Laboratuvarda geliştirilen temel yapay sinir ağı modelinin yeni veri setine uyarlanmış manuel (NumPy tabanlı) sürümüdür.

- **`proje.ipynb`**  
  Veri analizi, ön işleme, Scikit-learn tabanlı temel model, geliştirilmiş model ve sonuç karşılaştırmalarını içermektedir.

- **`heart_failure_clinical_records_dataset.csv`**  
  Kullanılan veri setidir.

- **`README.md`**  
  Çalışmanın IMRAD formatındaki proje raporudur.

---

## How to Run

### 1) Gerekli kütüphaneleri yükleyin

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 2) Proje klasörüne veri setini ekleyin

Aşağıdaki dosyanın proje klasöründe bulunduğundan emin olun:

```text
heart_failure_clinical_records_dataset.csv
```

### 3) Notebook dosyalarını açın

- `lab_model.ipynb`
- `proje.ipynb`

dosyalarını VS Code veya Jupyter Notebook ortamında açın.

### 4) Hücreleri sırayla çalıştırın

Notebook hücreleri yukarıdan aşağıya sırayla çalıştırılarak:

- veri analizi
- veri ön işleme
- model eğitimi
- performans değerlendirmesi
- model karşılaştırması

tekrar üretilebilir.

---

## Conclusion

Bu projede, laboratuvar ortamında geliştirilen temel yapay sinir ağı modeli ile başlanmış, ardından aynı problem daha gelişmiş kütüphane tabanlı modeller ile tekrar ele alınmıştır.

Sonuç olarak:
- laboratuvar temel modeli en düşük performansı göstermiş,
- Scikit-learn temel model belirgin iyileşme sağlamış,
- geliştirilmiş model ise klinik açıdan daha kritik olan pozitif sınıfı daha başarılı şekilde tahmin etmiştir.

Bu nedenle proje kapsamında **nihai tercih edilen model geliştirilmiş model olmuştur**.
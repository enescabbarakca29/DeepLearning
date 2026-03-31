# Kalp Yetmezliği Klinik Kayıtları ile NumPy, Scikit-learn ve PyTorch Tabanlı Sinir Ağı Modellerinin Karşılaştırılması

## Introduction

Bu proje, `heart_failure_clinical_records_dataset.csv` veri seti kullanılarak kalp yetmezliği hastalarında ölüm olayının (`DEATH_EVENT`) ikili sınıflandırma problemi olarak tahmin edilmesini amaçlamaktadır. Kalp yetmezliği, klinik karar destek sistemleri açısından kritik bir problemdir; özellikle hastane takip sürecinde riskli hastaların daha erken belirlenmesi, tedavi önceliklendirmesi ve izlem stratejileri açısından önem taşır.

Çalışma kapsamında önce veri seti üzerinde keşifsel veri analizi ve ön işleme yapılmıştır. Daha sonra laboratuvarda gösterilen tek gizli katmanlı NumPy tabanlı sinir ağı mantığı korunarak temel model sıfırdan yeniden uygulanmıştır. Bu kısımda özellikle ağırlık başlatma, ileri yayılım, binary cross entropy maliyet hesabı, geri yayılım, SGD ile parametre güncelleme ve tahmin adımları laboratuvar akışına sadık kalacak biçimde yeniden düzenlenmiştir. Bunun ardından daha derin iki NumPy modeli geliştirilmiş, aynı temel mimari Scikit-learn ve PyTorch ile yeniden kurulmuş ve sonuçlar ortak bir veri bölmesi üzerinde karşılaştırılmıştır.

## Methods

### Veri Seti

- Kaynak dosya: `data/heart_failure_clinical_records_dataset.csv`
- Gözlem sayısı: 299
- Değişken sayısı: 13
- Hedef değişken: `DEATH_EVENT`
- Sınıf dağılımı: 203 adet `0`, 96 adet `1`

Veri setinde eksik gözlem bulunmamaktadır ve yinelenen satır sayısı `0` olarak hesaplanmıştır. Korelasyon analizi, hedef değişken ile en belirgin ilişkinin `time` (`-0.527`), `serum_creatinine` (`0.294`), `ejection_fraction` (`-0.269`) ve `age` (`0.254`) değişkenlerinde olduğunu göstermiştir. IQR tabanlı incelemede özellikle `creatinine_phosphokinase`, `platelets` ve `serum_creatinine` sütunlarında aykırı gözlemler dikkat çekmiştir; ancak klinik anlam taşıyabilecek bu değerler silinmemiş, standartlaştırma ile ölçek etkisi azaltılmıştır.

### Ön İşleme

- Sabit rastgelelik tohumu: `42`
- Bölme oranı: `%60` eğitim, `%20` doğrulama, `%20` test
- Bölme yöntemi: `stratified train/validation/test split`
- Standardizasyon: `StandardScaler`
- Kritik kural: `scaler` yalnızca eğitim verisi üzerinde fit edilmiştir
- Aynı split ve aynı scaler bütün modellerde yeniden kullanılmıştır
- Temel NumPy, Scikit-learn ve PyTorch baseline modellerinde aynı random seed ve aynı başlangıç ağırlıkları kullanılmıştır

Hazırlık adımları `src/data_utils.py` içinde merkezi olarak tanımlanmıştır. Bu dosya aynı zamanda `models/scaler.joblib` ve `models/split_indices.json` artefaktlarını üretir.

### Modeller

#### 1. Laboratuvar Temel Modeli (NumPy)

Laboratuvarda gösterilen mantık referans alınarak aşağıdaki bileşenler sıfırdan yazılmıştır:

- parameter initialization
- forward propagation
- binary cross entropy cost
- backward propagation
- SGD ile parametre güncelleme
- training loop
- prediction

Mimari:

- `12 -> 8 -> 1`
- Gizli katman aktivasyonu: `tanh`
- Çıkış aktivasyonu: `sigmoid`
- Öğrenme oranı: `0.05`
- Epoch sayısı: `400`
- Başlatma stratejisi: laboratuvar notebook’una paralel biçimde küçük Gauss dağılımı (`std=0.01`) ve sıfır bias

#### 2. Geliştirilmiş NumPy Modelleri

- Model A: `12 -> 16 -> 8 -> 1`, öğrenme oranı `0.01`, `600` epoch
- Model B: `12 -> 32 -> 16 -> 1`, öğrenme oranı `0.005`, `300` epoch, `L2=0.001`

Bu modeller aynı NumPy çekirdeği üzerine kurulmuştur; yalnızca katman derinliği, kapasite ve düzenlileştirme değiştirilmiştir.

#### 3. Scikit-learn Modeli

- Sınıf: `MLPClassifier`
- Mimari: `hidden_layer_sizes=(8,)`
- Aktivasyon: `tanh`
- Çözücü: `sgd`
- `random_state=42`
- `learning_rate_init=0.05`
- `max_iter=400`
- Başlatma stratejisi: NumPy temel modelde üretilen ortak başlangıç ağırlıkları `MLPClassifier` içine doğrudan aktarılmıştır

#### 4. PyTorch Modeli

- Mimari: `12 -> 8 -> 1`
- Aktivasyon: `tanh` + `sigmoid`
- Kayıp fonksiyonu: `BCELoss`
- Optimizasyon: `SGD`
- Öğrenme oranı: `0.05`
- Epoch sayısı: `400`
- Veri akışı: `Dataset` + `DataLoader`
- Başlatma stratejisi: NumPy temel model ile aynı başlangıç ağırlıkları ve bias değerleri PyTorch katmanlarına aktarılmıştır

### Değerlendirme Metrikleri

Bütün modeller aynı test kümesinde aşağıdaki metriklerle değerlendirilmiştir:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

Ek olarak eğitim ve doğrulama kaybı ile doğruluk eğrileri izlenmiştir.

## Results

### Temel Model Sonucu

Laboratuvar temelli NumPy modelinin test sonuçları:

- Accuracy: `0.7833`
- Precision: `0.7500`
- Recall: `0.4737`
- Specificity: `0.9268`
- F1-score: `0.5806`
- Confusion matrix: `[[38, 3], [10, 9]]`

Bu sonuç, düşük olmayan genel doğruluğa rağmen pozitif sınıfı yakalamada sınırlı bir recall değerine işaret etmektedir.

### Model Karşılaştırma Tablosu

| Model | Mimari | Epoch | Train Acc. | Val Acc. | Test Acc. | Precision | Recall | Specificity | F1-score | Confusion Matrix |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NumPy Model A | `12 -> 16 -> 8 -> 1` | 600 | 0.8380 | 0.8667 | 0.8333 | 0.8462 | 0.5789 | 0.9512 | 0.6875 | `[[39, 2], [8, 11]]` |
| Scikit-learn Baseline | `12 -> 8 -> 1` | 400 | 0.8547 | 0.8333 | 0.7833 | 0.7500 | 0.4737 | 0.9268 | 0.5806 | `[[38, 3], [10, 9]]` |
| NumPy Model B | `12 -> 32 -> 16 -> 1 (L2)` | 300 | 0.8436 | 0.8500 | 0.7500 | 0.6667 | 0.4211 | 0.9024 | 0.5161 | `[[37, 4], [11, 8]]` |
| NumPy Baseline | `12 -> 8 -> 1` | 400 | 0.8547 | 0.8333 | 0.7833 | 0.7500 | 0.4737 | 0.9268 | 0.5806 | `[[38, 3], [10, 9]]` |
| PyTorch Baseline | `12 -> 8 -> 1` | 400 | 0.8492 | 0.8333 | 0.7833 | 0.7143 | 0.5263 | 0.9024 | 0.6061 | `[[37, 4], [9, 10]]` |

### En İyi Model

Bu proje kapsamında en iyi genel model olarak `NumPy Model A` seçilmiştir. Bunun temel gerekçeleri:

- En yüksek doğrulama doğruluğunu (`0.8667`) vermesi
- Test kümesinde en yüksek accuracy (`0.8333`) ve F1-score (`0.6875`) değerine ulaşması
- Temel NumPy modele göre recall ve F1-score değerlerini anlamlı biçimde artırması
- Confusion matrix üzerinde yanlış negatif sayısını göreli olarak azaltması

## Discussion

Sonuçlar, laboratuvar temel modelinin veri setine uyarlanabilir ve savunulabilir bir başlangıç sunduğunu göstermektedir; ancak tek gizli katmanlı yapı pozitif sınıfı yakalamada sınırlı kalmıştır. Temel modelde eğitim doğruluğu `0.8547`, doğrulama doğruluğu ise `0.8333` düzeyinde kalmış; bu durum ağır bir overfitting üretmemekle birlikte pozitif sınıfın yeterince ayrıştırılamadığını göstermiştir. Model A’da gizli katman sayısının ve nöron kapasitesinin artırılması, sınıflar arası doğrusal olmayan örüntülerin daha iyi temsil edilmesini sağlamış ve özellikle F1-score üzerinde iyileşme üretmiştir.

Model B’ye L2 regularization eklenmesine rağmen bu yapı en iyi test sonucunu vermemiştir. Bunun nedeni, veri setinin küçük olması ve daha geniş mimarinin düzenlileştirme altında bile karar sınırını beklenen kadar iyileştirememesi olabilir. Scikit-learn ve PyTorch modelleri temel mimariyi başarılı şekilde yeniden üretmiş, fakat NumPy Model A’nın ulaştığı recall ve F1 dengesine çıkamamıştır. Özellikle ortak başlangıç ağırlıkları, aynı veri bölmesi ve aynı SGD yaklaşımı altında Scikit-learn sonucunun temel NumPy modelle neredeyse çakışması, referans mimarinin farklı kütüphanelerde tutarlı biçimde tekrarlandığını göstermektedir.

Overfitting açısından bakıldığında, ilk denemelerde daha derin NumPy modellerinde eğitim doğruluğunun çok yükselip doğrulama performansının geride kalabildiği görülmüştür. Bu nedenle hiperparametreler doğrulama performansına göre yeniden ayarlanmıştır. Nihai Model A seçimi, yüksek eğitim başarısına kaçmadan doğrulama ve test başarısını birlikte iyileştirdiği için yapılmıştır. Underfitting ise özellikle temel modelde daha belirgin şekilde gözlenmiştir; bunun göstergesi olarak doğrulama doğruluğunun görece sınırlı kalması ve pozitif sınıf recall değerinin düşük olması verilebilir.

İleri çalışma olarak şu geliştirmeler yapılabilir:

- Daha sistematik hiperparametre araması
- Sınıf dengesizliğine duyarlı kayıp veya örnekleme stratejileri
- ROC-AUC ve PR-AUC gibi ek metriklerin eklenmesi
- Daha kontrollü erken durdurma mekanizması

## Proje Yapısı

```text
Project-1/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
│   └── heart_failure_clinical_records_dataset.csv
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_lab_model_numpy.ipynb
│   ├── 03_improved_models_numpy.ipynb
│   ├── 04_sklearn_model.ipynb
│   └── 05_pytorch_model.ipynb
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── numpy_models.py
│   ├── sklearn_model.py
│   └── pytorch_model.py
├── outputs/
│   ├── figures/
│   └── reports/
└── models/
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Notebook Çalıştırma

1. Proje kök dizinine geçin.
2. `jupyter notebook` veya `jupyter lab` başlatın.
3. Notebook’ları sırasıyla çalıştırın:
   - `01_eda_preprocessing.ipynb`
   - `02_lab_model_numpy.ipynb`
   - `03_improved_models_numpy.ipynb`
   - `04_sklearn_model.ipynb`
   - `05_pytorch_model.ipynb`

Notebook’lar kendi başlarına çalışacak şekilde düzenlenmiştir ve `src/` modüllerini doğrudan kullanır.

## Üretilen Çıktılar

- Şekiller: `outputs/figures/`
- Karşılaştırma tablosu: `outputs/reports/model_comparison.csv`
- Ayrıntılı sonuç özeti: `outputs/reports/detailed_results.md`
- Ham rapor verisi: `outputs/reports/summary.json`
- Scaler ve model artefaktları: `models/`

## main.py Çalıştırma

```bash
python main.py
```

Bu komut veriyi yükler, ortak ön işleme adımlarını uygular, laboratuvar temelli NumPy temel modelini eğitir ve test sonuçlarını terminale yazdırır.

# YZM304 Derin Öğrenme Project-2

Bu proje, Ankara Üniversitesi YZM304 Derin Öğrenme dersi için Intel Image Classification veri seti kullanılarak hazırlanmıştır. Proje `Project-2/data/` altındaki mevcut veri yapısı ile çalışır; dışarıdan veri indirme işlemi yapmaz.

## Introduction

Bu çalışmanın problemi, doğal sahne görüntülerini altı sınıftan birine atayan bir görüntü sınıflandırma sistemi geliştirmektir. Kullanılan sınıflar `buildings`, `forest`, `glacier`, `mountain`, `sea` ve `street` kategorileridir. Bu görev, piksel uzayındaki yerel örüntüleri öğrenebilen Evrişimsel Sinir Ağları (CNN) için doğal bir kullanım alanıdır. Projede hem sıfırdan yazılmış iki CNN mimarisi, hem de literatürde yaygın kullanılan ResNet18 tabanlı hazır bir mimari incelenmiştir. Ek olarak hibrit yaklaşımda bir CNN ile çıkarılan öznitelikler üzerinde klasik makine öğrenmesi modeli eğitilmiştir. Çalışmanın amacı, farklı mimari tercihlerin sınıflandırma başarımına etkisini sistematik olarak karşılaştırmaktır.

## Methods

### Veri seti yapısı

Proje veri seti yolunu relative path ile kullanır:

```text
Project-2/
├── data/
│   ├── seg_train/
│   ├── seg_test/
│   └── seg_pred/
```

Kod, Intel veri setinde görülebilen iç içe klasör varyasyonlarını da otomatik çözer. `seg_train` eğitim verisi, `seg_test` bağımsız test verisi, `seg_pred` ise etiketlenmemiş tahmin örnekleri için kullanılır.

### Ön işleme

- Eğitim verisine `Resize`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `Normalize` uygulanır.
- Doğrulama ve test verisine yalnızca `Resize`, `ToTensor`, `Normalize` uygulanır.
- Validation set, `seg_train` içinden stratified split ile ayrılır.
- Veri sızıntısını önlemek için test kümesi eğitime katılmaz.

### Train / validation / test yapısı

- Eğitim kümesi: `seg_train` içinden ayrılan ana bölüm
- Doğrulama kümesi: `seg_train` içinden `VAL_SIZE` oranında ayrılan bölüm
- Test kümesi: `seg_test`

Tüm bu ayarlar `src/config.py` içinden değiştirilebilir.

### Model 1 mimarisi

`model1_lenet_like.py` içinde RGB girişe uygun, LeNet-5 mantığını takip eden bir CNN elle yazılmıştır. Modelde açık şekilde konvolüsyon, ReLU, max-pooling, flatten ve tam bağlantılı katmanlar tanımlanmıştır.

### Model 2 mimarisi

`model2_improved_cnn.py` içinde ilk modele göre Batch Normalization ve Dropout eklenmiş daha güçlü bir CNN yazılmıştır. Bu yapı, eğitim kararlılığını artırmak ve aşırı öğrenmeyi azaltmak için tercih edilmiştir.

### Model 3 mimarisi

`model3_transfer.py` içinde `torchvision.models.resnet18` kullanılmıştır. Son fully connected katman 6 sınıf için yeniden tanımlanır. Varsayılan olarak `USE_PRETRAINED=False` seçilmiştir; bunun nedeni projenin çevrimdışı ve tekrar üretilebilir olmasıdır. İstenirse `config.py` üzerinden açılabilir.

### Hibrit yaklaşım

Hibrit modelde ResNet18 tabanlı CNN’in feature extraction kısmı kullanılır. Eğitim, doğrulama ve test görüntülerinden çıkarılan öznitelikler vektöre dönüştürülür. Zorunlu `.npy` çıktıları:

- `outputs/features/X_train_features.npy`
- `outputs/features/y_train.npy`
- `outputs/features/X_test_features.npy`
- `outputs/features/y_test.npy`

Ek olarak shape bilgileri `outputs/features/feature_shapes.json` ve `outputs/features/feature_shapes.txt` içine yazılır. Bu özniteliklerle scikit-learn `LogisticRegression` modeli eğitilir ve test edilir.

### Loss function, optimizer ve hiperparametreler

- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning rate: `1e-3`
- Epoch sayısı: `5`
- Batch size: `64`

Bu seçimler CPU üzerinde de tamamlanabilecek, fakat yine de karşılaştırmalı deney üretmeye uygun dengeli bir başlangıç noktası olduğu için seçilmiştir. Daha uzun eğitim için aynı dosyadan artırılabilir.

### Donanım ve eğitim stratejisi

Kod GPU varsa `cuda`, yoksa `cpu` kullanır. Bu sistemde otomatik fallback vardır. Eğitim sırasında en iyi doğrulama başarımına ulaşan ağırlıklar `.pth` olarak kaydedilir. Ayrıca erken durdurma mantığı da bulunmaktadır.

## Results

Sonuçlar çalıştırma sonrasında aşağıdaki klasörlerde üretilir:

- `outputs/metrics/`: model bazlı metrik JSON/CSV dosyaları ve karşılaştırma tablosu
- `outputs/figures/`: sınıf dağılımı, örnek görüntüler, loss ve accuracy grafikleri
- `outputs/confusion_matrices/`: görsel ve sayısal confusion matrix çıktıları
- `outputs/reports/`: classification report, veri özeti ve tahmin raporları
- `outputs/features/`: hibrit model için çıkarılmış öznitelik dizileri

Karşılaştırma tablosu şu sütunları içerir:

- Model
- Train Accuracy
- Validation Accuracy
- Test Accuracy
- Precision
- Recall
- F1-score
- Training Time

Bu projede `python main.py` çalıştırıldıktan sonra elde edilen gerçek sonuçlar aşağıdaki gibidir:

| Model | Train Accuracy | Validation Accuracy | Test Accuracy | Precision | Recall | F1-score | Training Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model 1 - LeNet Like | 0.8179 | 0.8098 | 0.8107 | 0.8166 | 0.8107 | 0.8106 | 00:12:26 |
| Model 2 - Improved CNN | 0.7228 | 0.7738 | 0.7870 | 0.7960 | 0.7870 | 0.7869 | 00:11:39 |
| Model 3 - ResNet18 | 0.8105 | 0.7927 | 0.7887 | 0.7906 | 0.7887 | 0.7888 | 00:09:02 |
| Hibrit Model - CNN Feature + Logistic Regression | 0.8919 | 0.8347 | 0.8237 | 0.8235 | 0.8237 | 0.8235 | 00:01:12 |

En iyi test başarımı hibrit modelde elde edilmiştir. Buna karşılık tam CNN modelleri arasında en iyi test sonucu `Model 1 - LeNet Like` modeline aittir.

Hibrit öznitelik dosyalarının gerçek shape bilgileri:

- `X_train_features.npy`: `(11227, 512)`
- `y_train.npy`: `(11227,)`
- `X_test_features.npy`: `(3000, 512)`
- `y_test.npy`: `(3000,)`
- Ek doğrulama öznitelik bilgileri: `X_val=(2807, 512)`, `y_val=(2807,)`

Gerçek çıktı dosyaları:

- Loss grafikleri: `outputs/figures/model1_lenet_like_loss.png`, `model2_improved_cnn_loss.png`, `model3_transfer_loss.png`
- Accuracy grafikleri: `outputs/figures/model1_lenet_like_accuracy.png`, `model2_improved_cnn_accuracy.png`, `model3_transfer_accuracy.png`
- Confusion matrix dosyaları: `outputs/confusion_matrices/` altındaki `.png` ve `.csv` çıktıları
- Karşılaştırma tablosu: `outputs/metrics/comparison_metrics.csv`
- Sınıflandırma raporları: `outputs/reports/`

`main.py` tamamlandığında en iyi model ayrıca `outputs/metrics/comparison_metrics.json` içinde belirtilir. Hibrit özellik dosyalarının shape bilgileri ekrana basılır ve dosyaya kaydedilir.

## Discussion

Gerçek deney sonuçlarına göre en yüksek test doğruluğu `0.8237` ile hibrit modelde görülmüştür. Bu sonuç, ResNet18 tabanlı öznitelik çıkarıcının sahne sınıfları için ayırt edici özellikler ürettiğini ve bu özellikler üzerinde lojistik regresyonun etkili karar sınırları öğrendiğini göstermektedir. Ayrıca hibrit modelin eğitim süresi yalnızca `00:01:12` olduğu için doğruluk-zaman dengesi açısından da en verimli çözüm olmuştur.

Tam CNN modelleri arasında en iyi sonuç `Model 1 - LeNet Like` ile elde edilmiştir (`test accuracy = 0.8107`). Bu durum, mevcut ayarlar ve kısa eğitim süresi altında daha sade mimarinin veri setine daha hızlı uyum sağlayabildiğini göstermektedir. `Model 2 - Improved CNN` teorik olarak BatchNorm ve Dropout ile daha güçlü olsa da, bu modelin `5` epoch içinde tam potansiyeline ulaşamadığı görülmektedir. Dolayısıyla düzenlileştirme katmanları, daha uzun eğitim veya daha iyi hiperparametre ayarı ile daha fazla fayda sağlayabilir.

`Model 3 - ResNet18` modelinin test doğruluğu `0.7887` olarak ölçülmüştür. Bu projede `pretrained=False` seçildiği için model ağırlıkları sıfırdan öğrenilmiştir. Çevrimdışı tekrar üretilebilirlik açısından bu tercih doğru olsa da, pretrained ağırlıklar kullanıldığında ResNet18’in daha yüksek performans üretmesi muhtemeldir. Buna rağmen bu deney, transfer mimarisinin hazır omurga yapısını görevimize uyarlama sürecini açık biçimde göstermektedir.

BatchNorm ve Dropout etkisi sonuçlara bakıldığında karışık bir tablo sunmaktadır. `Model 2` aşırı öğrenmeyi sınırlamış olsa da kısa eğitim süresinde temel modele üstünlük kuramamıştır. Veri setinin sahne sınıfları arasında bazı görsel benzerlikler bulunduğu için, özellikle doğal manzaralar arasında karar sınırlarının daha karmaşık hale gelmesi mümkündür. Gelecekte daha yüksek epoch, learning rate scheduler, pretrained backbone, daha büyük görüntü boyutu ve ek augmentasyonlar ile performans artırılabilir.

## References

1. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
2. torchvision Documentation: https://pytorch.org/vision/stable/index.html
3. scikit-learn Documentation: https://scikit-learn.org/stable/
4. Intel Image Classification Dataset (Kaggle): https://www.kaggle.com/datasets/puneet6060/intel-image-classification
5. He, K., Zhang, X., Ren, S., Sun, J. Deep Residual Learning for Image Recognition.

## Proje klasör yapısı

```text
Project-2/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
├── notebooks/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── utils.py
│   ├── train.py
│   ├── evaluate.py
│   ├── feature_extraction.py
│   ├── classical_ml.py
│   ├── plots.py
│   └── models/
└── outputs/
```

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Nasıl çalıştırılır

Ana akış:

```bash
python main.py
```

Tüm proje ayarları `src/config.py` içinden düzenlenebilir. Özellikle `IMAGE_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE` ve `USE_PRETRAINED` tek merkezden yönetilir.

## Notebook dosyaları

- `01_data_analysis.ipynb`: veri analizi ve sınıf dağılımı
- `02_model1_lenet_like.ipynb`: Model 1 eğitimi
- `03_model2_improved_cnn.ipynb`: Model 2 eğitimi
- `04_model3_pretrained.ipynb`: ResNet18 tabanlı model
- `05_hybrid_feature_extraction_ml.ipynb`: hibrit öznitelik çıkarımı ve klasik ML
- `06_comparison_and_visualizations.ipynb`: sonuçların karşılaştırılması

Notebook’lar `src/` altındaki modülleri kullanacak şekilde tasarlanmıştır; böylece kod tekrarına düşmeden tekrar üretilebilir bir yapı korunur.

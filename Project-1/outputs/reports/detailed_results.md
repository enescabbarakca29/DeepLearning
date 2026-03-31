# Ayr?nt?l? Model Sonu?lar?

En iyi model: **numpy_model_a**

## numpy_baseline
- Mimari: `12 -> 8 -> 1`
- Epoch: `400`
- Accuracy: `0.7833`
- Precision: `0.7500`
- Recall: `0.4737`
- Specificity: `0.9268`
- F1-score: `0.5806`
- Confusion matrix: `[[38, 3], [10, 9]]`
- Uyum yorumu: Belirgin bir overfitting görülmüyor; modelin genellemesi kabul edilebilir düzeyde.
- Not: Laboratuvar mant???na dayal? temel NumPy model. Ortak ba?lang?? a??rl?klar? k???k Gauss da??l?m? ve s?f?r bias ile olu?turuldu.

## numpy_model_a
- Mimari: `12 -> 16 -> 8 -> 1`
- Epoch: `600`
- Accuracy: `0.8333`
- Precision: `0.8462`
- Recall: `0.5789`
- Specificity: `0.9512`
- F1-score: `0.6875`
- Confusion matrix: `[[39, 2], [8, 11]]`
- Uyum yorumu: Belirgin bir overfitting görülmüyor; modelin genellemesi kabul edilebilir düzeyde.
- Not: Daha derin mimari ile kapasite art?r?ld?; derin a? i?in daha kararl? Xavier ba?latma kullan?ld?.

## numpy_model_b
- Mimari: `12 -> 32 -> 16 -> 1 (L2)`
- Epoch: `300`
- Accuracy: `0.7500`
- Precision: `0.6667`
- Recall: `0.4211`
- Specificity: `0.9024`
- F1-score: `0.5161`
- Confusion matrix: `[[37, 4], [11, 8]]`
- Uyum yorumu: Belirgin bir overfitting görülmüyor; modelin genellemesi kabul edilebilir düzeyde.
- Not: Daha geni? mimari ve L2 regularization kullan?ld?.

## sklearn_baseline
- Mimari: `12 -> 8 -> 1`
- Epoch: `400`
- Accuracy: `0.7833`
- Precision: `0.7500`
- Recall: `0.4737`
- Specificity: `0.9268`
- F1-score: `0.5806`
- Confusion matrix: `[[38, 3], [10, 9]]`
- Uyum yorumu: Belirgin bir overfitting görülmüyor; modelin genellemesi kabul edilebilir düzeyde.
- Not: Scikit-learn MLPClassifier ile ayn? 8 n?ronlu tek gizli katman, tanh aktivasyonu, SGD ??z?c?s? ve ortak ba?lang?? a??rl?klar? kullan?ld?.

## pytorch_baseline
- Mimari: `12 -> 8 -> 1`
- Epoch: `400`
- Accuracy: `0.7833`
- Precision: `0.7143`
- Recall: `0.5263`
- Specificity: `0.9024`
- F1-score: `0.6061`
- Confusion matrix: `[[37, 4], [9, 10]]`
- Uyum yorumu: Belirgin bir overfitting görülmüyor; modelin genellemesi kabul edilebilir düzeyde.
- Not: PyTorch uygulamas? temel modelle ayn? katman d?zeni, tanh gizli aktivasyonu, sigmoid ??k??, SGD ve ortak ba?lang?? a??rl?klar? ile e?itildi.

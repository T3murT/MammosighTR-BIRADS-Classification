# MammosighTR Veri Seti Üzerinde Derin Öğrenme Tabanlı BI-RADS Mamografi Sınıflandırması

Bu açık kaynaklı yazılım deposu, T.C. Sağlık Bakanlığı Ulusal Kanser Tarama Programı (KETEM) kapsamında derlenen **MammosighTR (`MG_EGITIM_1`)** ulusal mamografi veri kümesi (Koç ve ark., 2025) üzerinde üç sınıflı klinik triyaj odaklı **BI-RADS sınıflandırması (`BI-RADS 0`, `BI-RADS 1-2`, `BI-RADS 4-5`)** gerçekleştirmek amacıyla geliştirilmiş veri ön işleme, derin öğrenme model eğitimi, hasta bazlı çoklu projeksiyon (multi-view) bağlamsal füzyon, klinik Pareto kalibrasyonu ve Grad-CAM açıklanabilirlik (XAI) betiklerini barındırmaktadır.

## 1. Depo Yapısı ve İçerik

| Dosya Adı | İçerik ve Metodolojik Kapsam |
| :--- | :--- |
| `mammosight_dataset_prep.ipynb` | **Akışlı DICOM Ön İşleme ve Veri Hazırlama:** Çok parçalı DICOM arşivlerinin yerel diske bütünüyle açılmadan `stream-unzip` ile akışlı okunması, hasta kimliği (`case_id`) bazında sızıntısız tabakalı (%70 eğitim, %15 doğrulama, %15 test) veri ayrımı, fotometrik polarite düzeltmesi (`MONOCHROME1` -> `MONOCHROME2`), bağlı bileşen analizi ile kurşun yön etiketi temizliği, en-boy oranını koruyan 512x512 letterbox ölçekleme ve 3 kanallı çoklu kontrast kodlama (Gamma + CLAHE Fine + CLAHE Detail/Unsharp). |
| `mammosight_yolov8.ipynb` | **YOLOv8m-cls Eğitimi ve Değerlendirmesi:** C2f tabanlı YOLOv8m-cls mimarisinin 2 aşamalı progresif çözünürlük (Aşama 1: 512x512, Aşama 2: 576x576 FixRes) stratejisiyle eğitimi, 6-görünümlü çok ölçekli test zamanı veri artırma (TTA), hasta bazlı 4-yön (`RCC`, `LCC`, `RMLO`, `LMLO`) bağlamsal olasılık füzyonu ve Pareto-optimal klinik eşik kalibrasyonu. |
| `mammosight_yolo11.ipynb` | **YOLO11m-cls Eğitimi ve Değerlendirmesi:** C3k2 ve C2PSA (Kısmi Öz-Dikkat) bloklarına sahip YOLO11m-cls mimarisinin 2 aşamalı (512x512 -> 576x576 FixRes) eğitimi, 6-görünümlü TTA çıkarımı, hasta bazlı 4-yön bağlamsal füzyon ve klinik karar eşiği optimizasyonu. |
| `mammosight_efficientnet.ipynb` | **EfficientNet-B0 (GeM + Kalite MLP) Eğitimi ve Değerlendirmesi:** Genelleştirilmiş Ortalama Havuzlama (`GeM`) ve 12 boyutlu görüntü kalitesi/konumlandırma öznitelik vektörünü işleyen yardımcı MLP kolu ile zenginleştirilmiş EfficientNet-B0 mimarisi; `FocalCrossEntropyLoss`, `ModelEMA` ve 5 kollu çoklu düşürme (Multi-Sample Dropout) düzenlileştirmesi. |
| `mammosight_sota_ensemble.ipynb` | **Hibrit Topluluk Öğrenmesi (Ensemble) ve Grad-CAM Analizi:** `ConvNeXt-Small (GeM + Quality-MLP)` ve `YOLO11m-cls` (ve isteğe bağlı üçüncü üye olarak `EfficientNet-B0`) modellerinin olasılık düzeyinde dışbükey ağırlıklı birleştirilmesi, hasta bazlı 4-yön çoklu görünüm füzyonu ve sınıf bazlı Grad-CAM görsel açıklanabilirlik haritalarının (`mammosight_gradcam.png`) üretimi. |

## 2. Veri Seti ve Klinik Sınıflandırma Şeması

Deneysel çalışmalar, MammosighTR koleksiyonunun `MG_EGITIM_1` alt kümesinde yer alan 18.879 vakalık ulusal tarama verisi üzerinde yürütülmüştür. Klinik sevk zincirine uygun olarak olgular üç hedef sınıfta modellenmiştir:
- **BI-RADS 0:** Ek görüntüleme veya karşılaştırma gerektiren eksik değerlendirmeler.
- **BI-RADS 1-2:** Rutin tarama takibinde kalan negatif veya iyi huylu (benign) bulgular.
- **BI-RADS 4-5:** Histopatolojik örnekleme (biyopsi) endikasyonu taşıyan malignite kuşkulu lezyonlar.

Veri sızıntısını (data leakage) kesin biçimde önlemek amacıyla dört standart yönü eksiksiz 3.980 hastaya ait 15.920 mamogramlık çalışma kohortu görüntü satırları yerine **hasta kimliği (`case_id`, `random_state=42`)** bazında tabakalı olarak %70 eğitim ($N_{\text{ham}}=11.144$ projeksiyon / $2.786$ hasta; azınlık aşırı örneklemesi sonrası $N_{\text{dengeli}}=14.688$), %15 doğrulama ($N=2.388$ projeksiyon / $597$ hasta) ve %15 bağımsız test ($N=2.388$ projeksiyon / $597$ hasta) kümelerine ayrılmıştır.

## 3. Deneysel Sonuçlar ve Ablasyon Analizi (Bağımsız Test Kümesi, $N=2.388$ Projeksiyon / $597$ Hasta)

Bağımsız test kümesi ($N=2.388$ mamogram: 440 BI-RADS 0, 1.052 BI-RADS 1-2, 896 BI-RADS 4-5) üzerinde hem 4-yön bağlamsal füzyon öncesi tekil görünüm baz modeller hem de 4-yön çoklu görünüm füzyonu sonrası modeller ve Hibrit Topluluk (Ensemble) başarımı aşağıda özetlenmiştir:

| Performans Metriği | İlk Baz v1 (Tekil, $N=6.444$) | Tekil Baz v2 (YOLO11m, $N=2.388$) | YOLO11m-cls (4-Yön) | YOLOv8m-cls (4-Yön) | EfficientNet-B0 (GeM+Q-MLP, 4-Yön) | Hibrit Topluluk (Ensemble, 4-Yön) | Net Kazanç (Aynı Test / İlk Baz v1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parametre (M) / GFLOPs** | 10,4 / 39,6 | 10,4 / 39,6 | 10,4 / 39,6 | 17,0 / 41,9 | **5,4 / 2,05** | 60,6 (Topluluk) | - |
| **3-Sınıflı Genel Doğruluk (%)** | 59,40 | 61,18 | 69,26 | 71,15 | 69,43 | **73,70** | **+12,52 / +14,30 puan** |
| **BI-RADS 4-5 Duyarlılık / Recall (%)** | 57,37 | 60,38 | 82,03 | 79,58 | 76,67 | **86,50** | **+26,12 / +29,13 puan** |
| **BI-RADS 4-5 ROC-AUC** | 0,8090 | 0,8042 | 0,8840 | 0,9020 | 0,8800 | **0,9200** | **+0,1158 / +0,1110** |
| **BI-RADS 1-2 ROC-AUC** | 0,7790 | 0,7690 | 0,8390 | 0,8470 | 0,8460 | **0,8773** | **+0,1083 / +0,0983** |
| **BI-RADS 0 ROC-AUC** | 0,7180 | 0,6805 | 0,7410 | 0,7520 | 0,6700 | **0,8093** | **+0,1288 / +0,0913** |
| **Makro ROC-AUC (Ortalama)** | 0,7690 | 0,7512 | 0,8213 | 0,8340 | 0,7987 | **0,8689** | **+0,1177 / +0,0999** |
| **2x2 Tanısal Alt-Matris Doğruluğu (1-2 vs 4-5) (%)** | 77,19 | 75,83 | 84,41 | 86,75 | 85,69 | **87,40** | **+11,57 / +10,21 puan** |
| **BI-RADS 4-5 Dengeli Doğruluk (%)** | 71,80 | 71,85 | 81,26 | 83,05 | 81,40 | **84,57** | **+12,72 / +12,77 puan** |
| **BI-RADS 4-5 Özgüllük / Specificity (%)** | 86,22 | 83,31 | 80,50 | **86,53** | 86,13 | 82,64 | -0,67 / -3,58 puan |
| **BI-RADS 4-5 Kesinlik / Precision (%)** | 72,19 | 68,48 | 71,64 | **78,01** | 76,85 | 74,95 | +6,47 / +2,76 puan |
| **BI-RADS 4-5 F1-Skoru** | 0,6400 | 0,6418 | 0,7648 | 0,7878 | 0,7676 | **0,8031** | **+0,1613 / +0,1631** |
| **BI-RADS 1-2 F1-Skoru** | 0,6500 | 0,6827 | 0,7413 | 0,7580 | 0,7584 | **0,7734** | **+0,0907 / +0,1234** |
| **BI-RADS 0 F1-Skoru** | 0,4200 | 0,3053 | 0,3191 | 0,3733 | 0,2031 | **0,4615** | **+0,1562 / +0,0415** |
| **Makro F1-Skoru (3-Sınıf)** | 0,5700 | 0,5432 | 0,6084 | 0,6397 | 0,5764 | **0,6793** | **+0,1361 / +0,1093** |

*Not: Hibrit Topluluk modelinin aynı $N=2.388$ test kümesindeki tekil görünüm baz modele (`Tekil Baz v2`) kıyasla sağladığı genel doğruluk (+12,52 puan; McNemar $\chi^2=168,4$; $p<0,0001$), BI-RADS 4-5 duyarlılık (+26,12 puan; McNemar $\chi^2=214,6$; $p<0,0001$) ve BI-RADS 4-5 ROC-AUC (+0,1158; DeLong $Z=11,84$; $p<0,0001$) artışları hem projeksiyon ($N=2.388$) hem de hasta kümesi ($N_{\text{hasta}}=597$) düzeyinde istatistiksel olarak yüksek anlamlılığa sahiptir.*

## 4. Donanım Ortamı ve Çalıştırma Sırası

Betikler Kaggle (`2x NVIDIA Tesla T4 GPU`, 32 GB VRAM) veya yerel CUDA destekli çalışma ortamlarıyla uyumludur. Uzun süreli oturumlarda kesintisiz çalışmayı güvence altına almak amacıyla tüm defterlerde durum izleme (`prep_checkpoint.json`, `stage_state.json`) ve çıkarım önbellekleme (`*_tta_cache.npz`) mekanizmaları tanımlanmıştır.

Önerilen çalıştırma sırası:
1. `mammosight_dataset_prep.ipynb` çalıştırılarak ön işlenmiş `yolo_dataset.zip` arşivi oluşturulur.
2. `mammosight_yolov8.ipynb`, `mammosight_yolo11.ipynb` ve `mammosight_efficientnet.ipynb` betikleriyle bireysel mimariler eğitilir ve test metrikleri kaydedilir.
3. `mammosight_sota_ensemble.ipynb` çalıştırılarak `ConvNeXt-Small` modeli eğitilir, mimarilerin olasılık tahminleri hasta bazlı çoklu projeksiyon füzyonu altında birleştirilir ve Grad-CAM analiz çıktıları üretilir.

## 5. Lisans ve Atıf

Bu yazılım deposu akademik araştırma ve yeniden üretilebilirlik amacıyla açık kaynaklı olarak paylaşılmıştır. Çalışmayı kullanmanız durumunda ilgili makaleye ve MammosighTR veri setine (Koç ve ark., 2025, *Radiology: Artificial Intelligence*) atıf yapılması rica olunur.

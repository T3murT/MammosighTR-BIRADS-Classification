# MammosighTR Veri Setinde Çoklu Görünüm Bağlamsal Füzyonu ve Hibrit Topluluk Öğrenmesi ile Mamografi BI-RADS Sınıflandırması

Bu açık kaynaklı yazılım deposu, T.C. Sağlık Bakanlığı Ulusal Kanser Tarama Programı (KETEM) kapsamında derlenen **MammosighTR (`MG_EGITIM_1`)** ulusal mamografi veri kümesi (Koç ve ark., 2025) üzerinde üç sınıflı klinik triyaj odaklı **BI-RADS sınıflandırması (`BI-RADS 0`, `BI-RADS 1-2`, `BI-RADS 4-5`)** gerçekleştirmek amacıyla geliştirilmiş veri ön işleme, derin öğrenme model eğitimi, hasta bazlı çoklu projeksiyon (multi-view) bağlamsal füzyon, ağırlıklı klinik logit kalibrasyonu ve Grad-CAM açıklanabilirlik (XAI) betiklerini barındırmaktadır.

## 1. Depo Yapısı ve İçerik

| Dosya Adı | İçerik ve Metodolojik Kapsam |
| :--- | :--- |
| `mammosight_dataset_prep.ipynb` | **Akışlı DICOM Ön İşleme ve Veri Hazırlama:** Çok parçalı DICOM arşivlerinin yerel diske bütünüyle açılmadan `stream-unzip` ile akışlı okunması, hasta kimliği (`case_id`) bazında sızıntısız tabakalı (%70 eğitim, %15 doğrulama, %15 test) veri ayrımı, fotometrik polarite düzeltmesi (`MONOCHROME1` -> `MONOCHROME2`), bağlı bileşen analizi ile kurşun yön etiketi temizliği, en-boy oranını koruyan 512x512 letterbox ölçekleme ve 3 kanallı çoklu kontrast kodlama (Gamma + CLAHE Fine + CLAHE Detail/Unsharp). |
| `mammosight_yolov8.ipynb` | **YOLOv8m-cls Eğitimi ve Değerlendirmesi:** C2f tabanlı YOLOv8m-cls mimarisinin 2 aşamalı progresif çözünürlük (Aşama 1: 512x512 [50 ep], Aşama 2: 576x576 FixRes [12 ep]) stratejisiyle eğitimi, çok ölçekli test zamanı veri artırma (TTA), hasta bazlı 4-yön (`RCC`, `LCC`, `RMLO`, `LMLO`) bağlamsal olasılık füzyonu ve ağırlıklı klinik logit kalibrasyonu. |
| `mammosight_yolo11.ipynb` | **YOLO11m-cls Eğitimi ve Değerlendirmesi:** C3k2 ve C2PSA (Kısmi Öz-Dikkat) bloklarına sahip YOLO11m-cls mimarisinin 2 aşamalı (512x512 -> 576x576 FixRes) eğitimi, TTA çıkarımı, hasta bazlı 4-yön bağlamsal füzyon ve klinik logit kalibrasyonu. |
| `mammosight_efficientnet.ipynb` | **EfficientNet-B0 (GeM + Kalite MLP) Eğitimi ve Değerlendirmesi:** Genelleştirilmiş Ortalama Havuzlama (`GeM`) ve 12 boyutlu görüntü kalitesi/konumlandırma öznitelik vektörünü işleyen yardımcı MLP kolu ile zenginleştirilmiş EfficientNet-B0 mimarisi; `FocalCrossEntropyLoss`, `ModelEMA` ve `CoarseDropout` düzenlileştirmesi. |
| `mammosight_sota_ensemble.ipynb` | **Hibrit Topluluk Öğrenmesi (Ensemble) ve Grad-CAM Analizi:** `ConvNeXt-Small (GeM + Quality-MLP)` ve `YOLO11m-cls` (ve isteğe bağlı üçüncü üye olarak `EfficientNet-B0`) modellerinin olasılık düzeyinde dışbükey ağırlıklı birleştirilmesi, hasta bazlı 4-yön çoklu görünüm füzyonu ve sınıf bazlı Grad-CAM görsel açıklanabilirlik haritalarının (`mammosight_gradcam.png`) üretimi. |

## 2. Veri Seti ve Klinik Sınıflandırma Şeması

Deneysel çalışmalar, MammosighTR koleksiyonunun `MG_EGITIM_1` alt kümesinde yer alan dört standart bilateral projeksiyonu (`RCC, LCC, RMLO, LMLO`) eksiksiz 3.980 hastaya ait 15.920 mamogramlık çalışma kohortu üzerinde yürütülmüştür. Radyolog referans BI-RADS etiketleri klinik sevk zincirine uygun olarak üç hedef sınıfta modellenmiştir:
- **BI-RADS 0:** Ek görüntüleme veya önceki tetkiklerle karşılaştırma gerektiren eksik değerlendirmeler.
- **BI-RADS 1-2:** Rutin tarama takibinde kalan negatif veya iyi huylu (benign) bulgular.
- **BI-RADS 4-5:** Biyopsi / histopatolojik doğrulama sevki gerektiren kuşkulu bulgular.

Veri sızıntısını (data leakage) kesin biçimde önlemek amacıyla 3.980 hastaya ait 15.920 mamogram görüntü satırları yerine **hasta kimliği (`case_id`, `random_state=42`)** bazında tabakalı olarak %70 eğitim ($N_{\text{ham}}=11.144$ projeksiyon / $2.786$ hasta; sınıf dengeleme sonrası $N_{\text{dengeli}}=14.688$), %15 doğrulama ($N=2.388$ projeksiyon / $597$ hasta) ve %15 bağımsız test ($N=2.388$ projeksiyon / $597$ hasta) kümelerine ayrılmıştır.

## 3. Deneysel Sonuçlar ve Ablasyon Analizi (Aynı Bağımsız Test Kümesi, $N_{\text{hasta}}=597$ / $N=2.388$ Projeksiyon)

Bağımsız test kümesi ($597$ hasta / $2.388$ mamogram: $110$ hasta [440 kesit] BI-RADS 0, $263$ hasta [1.052 kesit] BI-RADS 1-2, $224$ hasta [896 kesit] BI-RADS 4-5) üzerinde 4-yön bağlamsal füzyon öncesi tekil görünüm baz model, 4-yön çoklu görünüm füzyonu sonrası bireysel modeller ve Hibrit Topluluk (Ensemble) başarımı aşağıda özetlenmiştir:

| Performans Metriği | Tekil Baz (YOLO11m) | YOLO11m (4-Yön) | YOLOv8m (4-Yön) | EffNet-B0 (4-Yön) | ConvNeXt-S (4-Yön) | Hibrit Topluluk (4-Yön) | Net Kazanç (Hibrit vs Tekil) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parametre (M) / GFLOPs (512px)** | 10,4 / 25,3 | 10,4 / 25,3 | 17,0 / 26,8 | **5,4 / 2,05** | 50,2 / 44,6 | 60,6 / 69,9 | - |
| **Hasta Düzeyi Doğruluk ($N=597$) (%)** | 61,31 | 69,51 | 71,36 | 69,68 | 71,69 | **74,04** | **+12,73 puan** |
| **Projeksiyon Doğruluğu ($N=2.388$) (%)** | 61,18 | 69,26 | 71,15 | 69,43 | 71,48 | **73,70** | **+12,52 puan** |
| **BI-RADS 4-5 Duyarlılık / Recall (%)** | 60,38 | 82,03 | 79,58 | 76,67 | 80,47 | **86,50** | **+26,12 puan** |
| **Sabit %85 Özgüllükte 4-5 Duyarlılığı (%)** | 57,81 | 78,57 | 81,36 | 77,90 | 82,14 | **84,82** | **+27,01 puan** |
| **BI-RADS 4-5 ROC-AUC** | 0,804 | 0,884 | 0,902 | 0,880 | 0,906 | **0,920** | **+0,116** |
| **BI-RADS 1-2 ROC-AUC** | 0,769 | 0,839 | 0,847 | 0,846 | 0,858 | **0,877** | **+0,108** |
| **BI-RADS 0 ROC-AUC** | 0,681 | 0,741 | 0,752 | 0,670 | 0,762 | **0,809** | **+0,128** |
| **ROC-AUC (Makro Ortalama)** | 0,751 | 0,821 | 0,834 | 0,799 | 0,842 | **0,869** | **+0,118** |
| **BI-RADS 4-5 Dengeli Doğruluk (%)** | 71,85 | 81,26 | 83,05 | 81,40 | 83,37 | **84,57** | **+12,72 puan** |
| **BI-RADS 4-5 Özgüllük / Specificity (%)** | 83,31 | 80,50 | **86,53** | 86,13 | 86,26 | 82,64 | -0,67 puan |
| **BI-RADS 4-5 Kesinlik / Precision (%)** | 68,48 | 71,64 | **78,01** | 76,85 | 77,89 | 74,95 | +6,47 puan |
| **BI-RADS 4-5 F1-Skoru** | 0,6418 | 0,7648 | 0,7878 | 0,7676 | 0,7916 | **0,8031** | **+0,1613** |
| **BI-RADS 1-2 F1-Skoru** | 0,6827 | 0,7413 | 0,7580 | 0,7584 | 0,7612 | **0,7734** | **+0,0907** |
| **BI-RADS 0 F1-Skoru** | 0,3053 | 0,3191 | 0,3733 | 0,2031 | 0,3918 | **0,4615** | **+0,1562** |
| **Makro F1-Skoru (3-Sınıf)** | 0,5432 | 0,6084 | 0,6397 | 0,5764 | 0,6482 | **0,6793** | **+0,1361** |
| **2x2 Koşullu Tanısal Doğruluk (%)†** | 75,83 | 84,41 | 86,75 | 85,69 | 86,91 | **87,40** | **+11,57 puan** |

*Not: Hibrit Topluluk modelinin aynı bağımsız test kümesindeki ($N_{\text{hasta}}=597$, $N=2.388$ projeksiyon) tekil görünüm baz modele kıyasla sağladığı hasta düzeyi doğruluk (+12,73 puan; hasta düzeyinde McNemar $b=89, c=13, \chi^2=55,15, p<0,0001$), BI-RADS 4-5 duyarlılık (+26,12 puan; hasta düzeyinde McNemar $b=68, c=9, \chi^2=43,70, p<0,0001$) ve BI-RADS 4-5 ROC-AUC (+0,116; küme-uyumlu DeLong $Z_{\text{küme}}=5,92, p<0,0001$; küme-düzeltmeli %95 GA: 0,894–0,946) artışları istatistiksel olarak yüksek anlamlılığa sahiptir. †2x2 Koşullu Tanısal Doğruluk gerçek etiketi ve tahmini BI-RADS 1-2 veya 4-5 olan olgular üzerinden hesaplandığından paydası modele göre değişmektedir (`1.804`–`1.866`).*

## 4. Donanım Ortamı ve Çalıştırma Sırası

Betikler Kaggle (`2x NVIDIA Tesla T4 GPU`, 32 GB VRAM) veya yerel CUDA destekli çalışma ortamlarıyla uyumludur. Uzun süreli oturumlarda kesintisiz çalışmayı güvence altına almak amacıyla tüm defterlerde durum izleme (`prep_checkpoint.json`, `stage_state.json`) ve çıkarım önbellekleme (`*_tta_cache.npz`) mekanizmaları tanımlanmıştır.

Önerilen çalıştırma sırası:
1. `mammosight_dataset_prep.ipynb` çalıştırılarak ön işlenmiş `yolo_dataset.zip` arşivi oluşturulur.
2. `mammosight_yolov8.ipynb`, `mammosight_yolo11.ipynb` ve `mammosight_efficientnet.ipynb` betikleriyle bireysel mimariler eğitilir ve test metrikleri kaydedilir.
3. `mammosight_sota_ensemble.ipynb` çalıştırılarak `ConvNeXt-Small` modeli eğitilir, mimarilerin olasılık tahminleri hasta bazlı çoklu projeksiyon füzyonu altında birleştirilir ve Grad-CAM analiz çıktıları üretilir.

## 5. Lisans ve Atıf

Bu yazılım deposu akademik araştırma ve yeniden üretilebilirlik amacıyla açık kaynaklı olarak paylaşılmıştır. Çalışmayı kullanmanız durumunda ilgili makaleye ve MammosighTR veri setine (Koç ve ark., 2025, *Radiology: Artificial Intelligence*) atıf yapılması rica olunur.

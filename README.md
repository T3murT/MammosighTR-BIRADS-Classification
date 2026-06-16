# MammosighTR-BIRADS-Classification

Bu depo, meme kanseri tarama sınıflandırması için MammosighTR veri seti üzerinde YOLOv8m-cls, YOLO11m-cls ve EfficientNet-B0 modellerinin uygulanmasını ve karşılaştırmalı analizini içerir.

## Veri Seti ve Ön İşleme
* **Veri Seti:** MammosighTR (T.C. Sağlık Bakanlığı Açık Veri Portalı'ndan 18.879 vaka).
* **Sınıflar:** Klinik gruplara göre üç kategoriye ayrılmıştır: BI-RADS 0, BI-RADS 1-2 ve BI-RADS 4-5.
* **Ön İşleme:** CLAHE kontrast artırma, akıllı meme bölgesi kırpma (siyah arka plan gürültüsünü temizleme) ve 512x512 piksel boyutlandırma.

## Depo Yapısı
* `mammosight_dataset_prep.ipynb`: Kaggle veri seti hazırlama notebook'u.
* `mammosight_yolov8.ipynb`: YOLOv8m-cls eğitim ve değerlendirme notebook'u.
* `mammosight_yolo11.ipynb`: YOLO11m-cls eğitim ve değerlendirme notebook'u.
* `mammosight_efficientnet.ipynb`: EfficientNet-B0 eğitim ve değerlendirme notebook'u.

## Performans Özeti
YOLO11m-cls, YOLOv8m-cls modeline kıyasla %39 daha az parametre kullanmasına rağmen, BI-RADS 4-5 (yüksek risk) sınıfı için %57.0 Duyarlılık (Recall) oranı elde ederek en yüksek klinik faydayı sağlamıştır.

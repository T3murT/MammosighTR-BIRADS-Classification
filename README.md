# MammosighTR — BI-RADS Sınıflandırma ve Model Değerlendirme Paketi

Bu depo, T.C. Sağlık Bakanlığı KETEM arşivinden derlenen **MammosighTR (`MG_EGITIM_1`)** mamografi veri seti üzerinde **3-Sınıflı Klinik Sevk Odaklı BI-RADS Sınıflandırması (`BI-RADS 0`, `BI-RADS 1-2`, `BI-RADS 4-5`)** için geliştirilmiş veri hazırlama, derin öğrenme eğitim ve değerlendirme kodlarını içerir.

---

## Notebook Dosyaları ve Görevleri

| Sıra | Notebook Dosyası | Açıklama |
| :--- | :--- | :--- |
| **1** | `mammosight_dataset_prep.ipynb` | **Veri Hazırlama:** Teknofest parçalı zip akışından (`stream-unzip`) görüntüleri okur; hasta bazlı (`Olgu Numarası`) %70/%15/%15 ayrımı yapar; pektoral kas silme, meme dokusu kırpma, CLAHE ve 3-kanallı kontrast kodlama uygular. Parça ve kontrol noktası (`prep_checkpoint.json`) yönetimi sağlar. |
| **2** | `mammosight_yolov8.ipynb` | **YOLOv8m-cls Eğitimi:** 2 Aşamalı (512x512 -> 576x576 FixRes) eğitim, epoch bazlı kontrol noktası kaydı, TTA önbellekleme, hasta bazlı 4-yön (`RCC, LCC, RMLO, LMLO`) multi-view bağlamsal füzyon ve klinik Pareto kalibrasyonu. |
| **3** | `mammosight_yolo11.ipynb` | **YOLO11m-cls Eğitimi:** C3k2 + C2PSA dikkat bloklu YOLO11m-cls mimarisi için 2 aşamalı (512 -> 576) eğitim, TTA önbellekleme, hasta bazlı multi-view bağlamsal füzyon ve klinik kalibrasyon. |
| **4** | `mammosight_efficientnet.ipynb` | **EfficientNet-B0 Eğitimi:** GeM + MaxPool hibrit havuzlama, 12-boyutlu görüntü kalitesi ve pozisyonlama öznitelik MLP yan kolu, FocalCrossEntropyLoss, ModelEMA ve multi-view bağlamsal füzyon. |
| **5** | `mammosight_sota_ensemble.ipynb` | **Topluluk Modeli ve Grad-CAM:** ConvNeXt-Small (GeM + Kalite MLP), YOLO11m-cls ve EfficientNet-B0 mimarilerinin topluluk öğrenmesi, multi-view bağlamsal füzyon ve Grad-CAM açıklanabilirlik haritaları (`mammosight_gradcam.png`). |

---

## Çalışma ve Kontrol Noktası (Checkpoint) Yönetimi

Notebook'lar uzun süreli oturumlarda ilerlemeyi korumak üzere periyodik olarak kontrol noktası dosyaları (`*.pt`, `*.pth`, `*.json`, `*.npz`) üretir:
- **Veri Hazırlama:** `prep_checkpoint.json` ve tampon dosyaları üzerinden tamamlanan arşiv parçaları doğrulanır ve sıradaki parçaya geçilir.
- **Model Eğitimi:** `last.pt` / `last_checkpoint.pth` ve `stage_state.json` dosyaları üzerinden tamamlanan eğitim aşamaları algılanır ve bir sonraki aşama yürütülür.

# 🐢 Deniz Kaplumbağası Biyometrik Tanımlama Sistemi

Bu proje, **SeaTurtleID2022** veri seti kullanılarak geliştirilen, derin öğrenme tabanlı bir bireysel tanımlama sistemidir. **MobileNetV2** mimarisi üzerinde uygulanan **Fine-Tuning** stratejisi ile kaplumbağaların karakteristik desenleri üzerinden kimlik tespiti yapılmaktadır.

---

## 📌 Proje Özeti
Deniz kaplumbağalarının korunması ve takibi için geliştirilen bu sistem, geleneksel markalama yöntemlerine alternatif olarak biyometrik (desen tabanlı) bir yaklaşım sunar.
# 🐢 Deniz Kaplumbağası Biyometrik Tanımlama Sistemi

Bu proje, **SeaTurtleID2022** veri seti kullanılarak geliştirilen, derin öğrenme tabanlı bir bireysel tanımlama sistemidir. **MobileNetV2** mimarisi üzerinde uygulanan **Fine-Tuning** stratejisi ile kaplumbağaların karakteristik desenleri üzerinden kimlik tespiti yapılmaktadır.

---

## 📌 Proje Özeti
Deniz kaplumbağalarının korunması ve takibi için geliştirilen bu sistem, geleneksel markalama yöntemlerine alternatif olarak biyometrik (desen tabanlı) bir yaklaşım sunar.

*   **Model:** v5 Fine-Tuned MobileNetV2
*   **Başarı Oranı:** %68.38 (Validation Accuracy)
*   **Mimari:** Modüler SOLID prensipleri ve Multi-Agent iş akışı
*   **Kapsam:** 19 farklı kaplumbağa bireyinin yüksek doğrulukla ayırt edilmesi

---

## 🛠️ Gereksinimler ve Bağımlılıklar
Projenin çalışması için sisteminizde **Python 3.10+** yüklü olmalıdır. Gerekli kütüphaneleri yüklemek için terminale aşağıdaki komutu yazın:

```bash
pip install tensorflow numpy opencv-python pillow streamlit
```

### Ana Bileşenler:
*   **TensorFlow:** Modelin yüklenmesi ve çıkarım (inference) işlemleri.
*   **OpenCV:** Görüntü ön işleme süreçleri.
*   **Streamlit:** Web tabanlı kullanıcı arayüzü.
*   **Pillow:** Görüntü dosya formatı yönetimi.

---

## 🚀 Çalıştırma Talimatları

Proje iki farklı modda çalıştırılabilir:

### 1. Grafik Arayüz Modu (Önerilen)
Kullanıcı dostu web arayüzünü başlatmak için proje dizininde terminale şu komutu yazın:

```bash
streamlit run app.py
```

> **Not:** Bu komut sonrası tarayıcınızda otomatik olarak bir panel açılacaktır. Buradan fotoğraf yükleyerek anlık analiz yapabilirsiniz.

### 2. Terminal Modu (Geliştirici)
Doğrudan kod üzerinden hızlı test yapmak ve teknik logları takip etmek için:

```bash
python main.py
```

---

## 📁 Proje Yapısı
*   `main.py`: Çekirdek tahmin mantığı ve `TurtleRecognizer` sınıfı.
*   `app.py`: Streamlit tabanlı modern kullanıcı arayüzü.
*   `config.py`: Model yolları ve  görüntü boyutları.
*   `proje_data/`: Analiz edilecek kaplumbağa fotoğraflarını (veri setini) içeren klasör.
*   `models/`: Model ağırlıklarını (`.h5`) ve sınıf isimlerini barındıran klasör.

---

**Geliştirici:** Abdullah Toprak Kılıç  
**Bölüm:** Pamukkale Üniversitesi - Bilgisayar Mühendisliği  
**Tarih:** Mayıs 2026

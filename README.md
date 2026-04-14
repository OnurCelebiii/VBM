# 🛰️ GNSS Konum Hatası İstatistiksel Analizi

**VBM604 — İstatistiksel Veri Analizi**
Hacettepe Üniversitesi · Bilgisayar Mühendisliği

---

## 📌 Proje Özeti

Bu proje, farklı GNSS (Global Navigation Satellite System) konstellasyonlarının ve ortam koşullarının **konum hatası** üzerindeki istatistiksel etkisini analiz etmektedir.

Temel araştırma soruları:
- GPS, Galileo, GLONASS ve BeiDou sistemlerinin konum hataları arasında istatistiksel olarak anlamlı fark var mı?
- Kentsel ortam (Urban Canyon) açık gökyüzü (Open Sky) ile kıyaslandığında ne kadar daha fazla hata üretiyor?
- PDOP, CN0 ve uydu sayısı konum hatasını ne ölçüde açıklıyor?

---

## 🗂️ Proje Yapısı

```
gnss_analysis/
│
├── data/
│   └── gnss_measurements.csv      # 3000 ölçüm, 8 değişken
│
├── figures/
│   ├── fig1_overview.png           # Dağılım, Q-Q, kutu, violin grafikleri
│   ├── fig2_corr_regression.png    # Korelasyon matrisi + regresyon grafikleri
│   └── fig3_constellation_detail.png  # Konstellasyon karşılaştırması
│
├── output/
│   └── statistical_report.txt     # Tüm istatistiksel çıktılar
│
├── generate_data.py               # Veri üretim scripti
├── analysis.py                    # Tam istatistiksel analiz
└── README.md
```

---

## 📡 Veri Seti

### Kaynak
Veri, **Google Smartphone Decimeter Challenge (ION GNSS+ 2021–2023)** yayınlarında bildirilen istatistiksel karakteristiklere ve **ITU-R P.1546** sinyal yayılım modeline dayalı gerçekçi GNSS ölçüm simülasyonudur.

> Gerçek veri indirmek için: [Kaggle — Google Android GNSS Dataset](https://www.kaggle.com/datasets/google/android-smartphones-high-accuracy-datasets)

### Değişkenler

| Değişken | Açıklama | Birim |
|---|---|---|
| `constellation` | Uydu sistemi (GPS/Galileo/GLONASS/BeiDou) | — |
| `environment` | Ortam tipi (Open Sky / Urban Canyon) | — |
| `elevation_deg` | Uydu yükselme açısı | derece |
| `cn0_dbhz` | Taşıyıcı-gürültü oranı | dB-Hz |
| `num_satellites` | Görünen uydu sayısı | adet |
| `pdop` | Position Dilution of Precision | — |
| **`position_error_m`** | **Konum hatası (bağımlı değişken)** | **metre** |
| `multipath_index` | Çok yollu yayılım katsayısı | 0–1 |

### Özet İstatistikler

| Özellik | Değer |
|---|---|
| Toplam ölçüm | 3.000 |
| Ortalama konum hatası | **2.91 m** |
| Std sapma | 1.96 m |
| Medyan | 2.60 m |
| Min / Maks | 0.10 m / 8.77 m |
| Çarpıklık | +0.36 (sağa çarpık) |

---

## 📊 Analiz Yöntemleri ve Bulgular

### 1. Betimleyici İstatistikler
Konum hatasının sağa çarpık dağıldığı gözlemlenmiştir (çarpıklık = +0.36). Kentsel ortamda ortalama hata **4.77 m** iken açık gökyüzünde **1.40 m**'dir.

### 2. Normallik Testleri

| Test | İstatistik | p-değeri | Sonuç |
|---|---|---|---|
| Shapiro-Wilk | W = 0.9378 | 1.33 × 10⁻¹³ | Normal değil ❌ |
| D'Agostino K² | 867.66 | 3.90 × 10⁻¹⁸⁹ | Normal değil ❌ |

**Yorum:** Dağılım normal değildir. Log-normal dağılım daha uygun olmakla birlikte parametrik testler büyük örneklem (N=3000) sayesinde Merkezi Limit Teoremi gereği güvenle uygulanabilir.

### 3. Hipotez Testi — Ortam Etkisi

```
H₀: μ(Open Sky) = μ(Urban Canyon)
H₁: μ(Open Sky) ≠ μ(Urban Canyon)
```

| Test | Sonuç |
|---|---|
| Levene (varyans homojenliği) | p < 0.001 → Varyanslar eşit değil |
| Welch t-testi | t = -87.41, **p ≈ 0** |
| Cohen's d (etki büyüklüğü) | **d = 3.25 (çok büyük)** |

**✅ H₀ reddedildi.** Kentsel ortam konum hatasını ortalama **3.37 metre** artırmaktadır.

### 4. Tek Yönlü ANOVA — Konstellasyon Etkisi

```
H₀: μ(GPS) = μ(Galileo) = μ(GLONASS) = μ(BeiDou)
H₁: En az bir grubun ortalaması farklıdır
```

| | F-istatistiği | p-değeri |
|---|---|---|
| ANOVA | **32.41** | 1.30 × 10⁻²⁰ |

**✅ H₀ reddedildi.** Post-hoc Tukey HSD sonuçları:

| Karşılaştırma | Fark | Anlamlı? |
|---|---|---|
| Galileo vs GLONASS | -0.99 m | ✅ Evet |
| Galileo vs GPS | -0.37 m | ✅ Evet |
| GPS vs GLONASS | -0.62 m | ✅ Evet |
| BeiDou vs GPS | -0.21 m | ❌ Hayır |

**Sıralama (en iyi → en kötü):** Galileo < GPS < BeiDou < GLONASS

### 5. Korelasyon Analizi

| Değişken | Pearson r | Anlamlılık |
|---|---|---|
| PDOP | **+0.833** | *** |
| Uydu sayısı | **-0.791** | *** |
| CN0 | **-0.564** | *** |
| Çok yollu yayılım | +0.376 | *** |
| Yükselme açısı | -0.045 | * |

**Yorum:** PDOP konum hatasıyla en güçlü pozitif, uydu sayısı ise en güçlü negatif korelasyona sahiptir.

### 6. Çoklu Doğrusal Regresyon

```
position_error ~ pdop + cn0 + elevation + num_satellites
               + multipath + constellation + environment
```

| Metrik | Değer |
|---|---|
| R² | **0.947** |
| Adj. R² | **0.947** |
| F-istatistiği | 5973.39 (p ≈ 0) |
| AIC | 3752.53 |

**En önemli bağımsız değişkenler (|t| sırasına göre):**
1. `environment_Urban Canyon` → β = +1.985 (t = 66.97)
2. `pdop` → β = +1.209 (t = 58.78)
3. `cn0_dbhz` → β = -0.059 (t = -17.56)
4. `constellation_Galileo` → β = -0.497 (t = -15.99)

---

## 🔑 Ana Bulgular

1. **Kentsel ortam en kritik faktör**: Urban Canyon, Open Sky'a kıyasla konum hatasını ortalama **+3.37 m artırmaktadır** (Cohen's d = 3.25, çok büyük etki).

2. **Galileo en hassas sistem**: Galileo, GLONASS'a göre ortalama **0.99 m daha az hata** üretmekte; GPS'e göre ise **0.37 m daha iyi** performans göstermektedir.

3. **PDOP belirleyici**: PDOP ile konum hatası arasında r = +0.83 korelasyon vardır. PDOP < 2 tutulduğunda hata dramatik biçimde azalmaktadır.

4. **Model açıklayıcılığı yüksek**: Regresyon modeli konum hatasının **%94.7'sini açıklamaktadır** (R² = 0.947).

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükle
pip install pandas numpy scipy matplotlib seaborn statsmodels

# Veriyi üret
python generate_data.py

# Tam analizi çalıştır
python analysis.py
```

---

## 📦 Bağımlılıklar

```
pandas >= 1.5
numpy >= 1.23
scipy >= 1.9
matplotlib >= 3.6
seaborn >= 0.12
statsmodels >= 0.13
```

---

## 📚 Referanslar

1. Google Android Team. *Smartphone Decimeter Challenge*. ION GNSS+ 2021–2023.
2. Kaplan, E. & Hegarty, C. *Understanding GPS/GNSS: Principles and Applications*, 3rd ed. Artech House, 2017.
3. ITU-R P.1546-6. *Method for point-to-area predictions for terrestrial services in the frequency range 30 MHz to 4 000 MHz*, 2019.
4. Barbeau, S. *awesome-gnss: Community list of open-source GNSS software and resources*. GitHub, 2024.

---

## ✍️ Katkı

Bu proje VBM604 İstatistiksel Veri Analizi dersi kapsamında hazırlanmıştır.
Veri üretimi, istatistiksel analiz ve görselleştirme tamamen Python ile gerçekleştirilmiştir.

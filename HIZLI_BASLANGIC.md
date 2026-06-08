# Quick Start Guide - FinBERT Projesi

## 🚀 Hızlı Başlangıç (5 Dakika)

### 1. Kurulum

```bash
# Terminal'i aç ve proje klasörüne git
cd "Financial News NLP and Deep Learning for Stock Prediction"

# Kütüphaneleri yükle
pip install -r requirements.txt

# FinBERT'i kur (ilk seferde ~5 dakika sürer)
python setup_finbert.py
```

### 2. Demo Çalıştır

```bash
# Streamlit uygulamasını başlat
streamlit run app/streamlit_app.py
```

Tarayıcında `http://localhost:8501` açılacak.

### 3. Veri Topla (Dashboard'dan)

1. Sol menüden bir hisse senedi seç (örn: AAPL)
2. "Fetch Data" butonuna bas
3. Bekle (30-60 saniye)
4. Grafikleri gör!

---

## 📋 Tam Pipeline (Bitirme Projesi İçin)

### Adım 1: Veri Toplama (10 dakika)

```bash
# Hisse senedi fiyatlarını çek
python src/data_collection/stock_scraper.py

# Haberleri çek
python src/data_collection/news_scraper.py
```

**Çıktılar:**
- `data/raw/stock_prices.csv` (5 hisse için 2 yıllık veri)
- `data/raw/financial_news.csv` (haberler)

### Adım 2: Veri İşleme (15-30 dakika)

```bash
# FinBERT ile sentiment analizi
python src/preprocessing/nlp_processor.py

# Teknik indikatörleri hesapla
python src/preprocessing/feature_engineer.py
```

**Çıktılar:**
- `data/processed/news_with_sentiment.csv`
- `data/processed/daily_sentiment_aggregated.csv`
- `data/processed/stock_features.csv`
- `data/processed/stock_features_with_sentiment.csv`

**Not:** FinBERT CPU'da yavaş. GPU varsa 5 dakika, yoksa 30 dakika.

### Adım 3: Model Eğitimi

#### A) Yön sınıflandırması (güncel — ikinci gelişme raporu)

```bash
# Tüm ticker'lar birlikte (pooled)
py -3.10 src/training/train_classification.py --epochs 35

# Ticker başına ayrı model + otomatik sentiment modu seçimi
py -3.10 src/training/train_classification_per_ticker.py --auto-sentiment --epochs 30 --threshold-objective accuracy
```

**Çıktılar:**
- `data/models/classification/classification_metrics.csv`
- `data/models/classification/per_ticker_metrics_auto_sentiment.csv`
- `data/models/classification/baseline_direction_classifier.pth`
- `data/models/classification/combined_direction_classifier.pth`

**Özet sonuçlar (test seti):**

| Deney | Ort. Accuracy |
|-------|----------------|
| Pooled baseline | ~%51,0 |
| Pooled combined | ~%49,9 |
| Per-ticker + auto sentiment | ~%51,6 |

Ticker bazında en iyi örnek: **MSFT ~%67,7** (combined + full sentiment modu).

#### B) Regresyon (ilk deneyler / isteğe bağlı)

```bash
py -3.10 src/training/train.py --model-type both --epochs 100 --target-column target_change_pct
```

**Çıktılar:**
- `data/models/baseline_lstm_model.pth`
- `data/models/combined_model.pth`

### Adım 4: Sonuçları İncele

#### Jupyter Notebook ile:

```bash
jupyter notebook
```

Sırayla aç:
1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_nlp_analysis.ipynb`
3. `notebooks/03_model_experiments.ipynb`

#### Streamlit Dashboard ile:

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Sonuçların Ekran Görüntülerini Al

### 1. Veri Analizi
- Hisse fiyat grafikleri
- Korelasyon matrisi
- Volatilite analizi

### 2. Sentiment Analizi
- FinBERT sentiment dağılımı
- En pozitif/negatif haberler
- Sentiment-fiyat ilişkisi

### 3. Model Sonuçları
- Eğitim loss grafikleri
- Tahmin vs gerçek grafikleri
- Accuracy metrikleri
- Karşılaştırma tablosu

### 4. Dashboard
- Canlı tahmin ekranı
- İnteraktif grafikler

**İpucu:** Her notebook'ta grafikler otomatik kaydediliyor:
- `data/evaluation/` klasöründe

---

## 🎓 Sunum İçin Hazırlık

### Slaytlar için Görseller

```bash
# Tüm evaluation görsellerini topla
ls data/evaluation/

# Bunları kullan:
- baseline_predictions.png
- combined_predictions.png
- baseline_directional.png
- model_comparison.png
```

### Demo Senaryosu (Canlı Sunum)

1. **Giriş (1 dk)**
   - "FinBERT kullanarak hisse senedi tahmini"
   
2. **Streamlit Demo (3 dk)**
   - AAPL seç
   - Fetch Data
   - News & Sentiment tabına geç
   - FinBERT skorlarını göster
   - Predictions tabına geç

3. **Jupyter Demo (2 dk)**
   - Notebook 3'ü aç
   - Model karşılaştırmasını göster

### Söyleyecekler

**FinBERT kısmı:**
- "FinBERT, BERT modelinin finansal haberler için fine-tune edilmiş versiyonu"
- "4,000'den fazla finansal makale üzerinde eğitilmiş"
- "%92 doğruluk oranı"
- "Normal BERT'ten farkı: 'bearish', 'earnings beat' gibi terimleri anlaması"

**Sonuçlar:**
- "Baseline LSTM: %XX accuracy"
- "FinBERT ile: %XX accuracy"
- "FinBERT sentiment'ın katkısı: +%X artış"

---

## ⚠️ Yaygın Sorunlar ve Hızlı Çözümler

### "FinBERT downloading hatası"
```bash
python setup_finbert.py
```

### "CUDA out of memory"
```bash
# CPU kullan veya batch size küçült
# Ya da VADER'a geç (geçici)
```

### "No news data"
```bash
# Finviz engellediyse bekle veya VPN kullan
# Veya Kaggle'dan dataset indir
```

### "Model eğitimi çok yavaş"
```bash
# Epochs'u azalt:
python src/training/train.py --epochs 20
```

---

## 📝 Hızlı Kontrol Listesi

- [ ] Kütüphaneler yüklendi
- [ ] FinBERT setup tamamlandı
- [ ] Veri toplandı (fiyatlar + haberler)
- [ ] Sentiment analizi yapıldı (FinBERT)
- [ ] Feature engineering tamamlandı
- [ ] Her iki model eğitildi
- [ ] Jupyter notebooks çalıştırıldı
- [ ] Ekran görüntüleri alındı
- [ ] Streamlit demo test edildi
- [ ] README okundu

---

## 💡 Son İpuçları

1. **İlk gün**: Veriyi topla, işle, kısa eğitim yap (20 epoch)
2. **İkinci gün**: Tam eğitim yap (100 epoch), sonuçları analiz et
3. **Üçüncü gün**: Raporunu yaz, görselleri hazırla
4. **Sunum günü**: Demo'yu test et, yedek ekran görüntüleri hazırla

5. **Demo sırasında internet yoksa:**
   - Önceden çektiğin verileri göster
   - Ekran görüntülerini kullan
   - Jupyter notebook'ları slayt gibi göster

6. **FinBERT bahsederken:**
   - BERT'in ne olduğunu kısaca açıkla
   - Neden finansal model gerektiğini söyle
   - Sonuçlarda farkını göster

Başarılar! 🎉

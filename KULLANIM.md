# Kullanım Kılavuzu (Turkish Usage Guide)

## Projeyi Çalıştırma Adımları

### 1. Kurulum

```bash
# Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# FinBERT modelini indir ve test et (ÖNERİLİR)
python setup_finbert.py
```

**Not:** İlk çalıştırmada FinBERT modeli indirilecek (~440MB). Bu yaklaşık 5-10 dakika sürebilir.

**Alternatif - VADER (Daha hızlı ama daha az doğru):**
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
# nlp_processor.py içinde method="vader" kullan
```

### 2. Veri Toplama

#### Hisse Senedi Fiyatlarını Çek

```bash
python src/data_collection/stock_scraper.py
```

Bu script:
- AAPL, TSLA, GOOGL, MSFT, AMZN için son 2 yıllık veriyi çeker
- `data/raw/stock_prices.csv` dosyasına kaydeder
- Yaklaşık 2-3 dakika sürer

#### Haberleri Çek

```bash
python src/data_collection/news_scraper.py
```

Bu script:
- Finviz'den finansal haberleri toplar
- `data/raw/financial_news.csv` dosyasına kaydeder
- Rate limiting nedeniyle 5-10 dakika sürebilir

**Önemli:** Web scraping bazen engellenebilir. Bu durumda:
- Birkaç saat bekleyip tekrar deneyin
- Veya Kaggle'dan hazır veri seti kullanın

### 3. Veri İşleme

#### Sentiment Analizi (FinBERT)

```bash
python src/preprocessing/nlp_processor.py
```

Bu script:
- FinBERT modelini yükler (ilk seferde biraz zaman alır)
- Haberlere sentiment analizi yapar
- Her habere pozitif/negatif/nötr skorları atar
- Günlük bazda aggregate eder

**İlk çalıştırmada:**
- GPU varsa: 5-10 dakika
- CPU'da: 15-30 dakika

Çıktılar:
- `data/processed/news_with_sentiment.csv`
- `data/processed/daily_sentiment_aggregated.csv`

**VADER kullanmak isterseniz:**
```python
# nlp_processor.py içinde 284. satırı değiştirin:
processor = NLPProcessor(method="vader")  # Çok daha hızlı ama daha az doğru
```

#### Feature Engineering (Teknik İndikatörler)

```bash
python src/preprocessing/feature_engineer.py
```

Çıktılar:
- `data/processed/stock_features.csv`
- `data/processed/stock_features_with_sentiment.csv`

### 4. Model Eğitimi

#### Tüm Modelleri Eğit

```bash
python src/training/train.py --model-type both --epochs 100
```

#### Sadece Baseline Model

```bash
python src/training/train.py --model-type baseline --epochs 50
```

**Parametreler:**
- `--model-type`: baseline, combined, veya both
- `--epochs`: Epoch sayısı (varsayılan: 100)
- `--sequence-length`: LSTM için geri bakış süresi (varsayılan: 60)

**Eğitim Süresi:**
- GPU ile: 10-20 dakika
- CPU ile: 30-60 dakika

### 5. Sonuçları Görüntüleme

#### Jupyter Notebooks

```bash
jupyter notebook
```

Sırayla açın:
1. `01_data_exploration.ipynb` - Veri analizi
2. `02_nlp_analysis.ipynb` - Sentiment analizi
3. `03_model_experiments.ipynb` - Model sonuçları

#### Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Tarayıcınızda `http://localhost:8501` adresine gidin.

## Yaygın Sorunlar ve Çözümleri

### Sorun 1: "ModuleNotFoundError"

**Çözüm:**
```bash
pip install -r requirements.txt
```

### Sorun 2: "FinBERT model not found" veya download hatası

**Çözüm:**
```bash
# Setup scriptini tekrar çalıştır
python setup_finbert.py

# Veya manuel olarak:
python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')"
```

**Alternatif - VADER kullan:**
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
# Sonra kod içinde method="vader" yap
```

### Sorun 3: "CUDA out of memory"

**Çözüm:**
- Batch size'ı azaltın: `--batch-size 16`
- Veya CPU kullanın (yavaş olur ama çalışır)

### Sorun 4: Web Scraping Engellendi

**Çözüm:**
- Birkaç saat bekleyin
- VPN kullanın
- Veya hazır veri seti kullanın

### Sorun 5: "Not enough data"

**Çözüm:**
- Sequence length'i azaltın: `--sequence-length 30`
- Daha uzun period için veri çekin

## Kısa Yollar

### Minimum Çalışan Sistem (MVP)

Hızlı test için:

```bash
# 1. Tek hisse senedi için veri çek
# stock_scraper.py içinde sadece ['AAPL'] kullanın

# 2. Basit model eğit
python src/training/train.py --model-type baseline --epochs 20

# 3. Sonuçları gör
jupyter notebook notebooks/03_model_experiments.ipynb
```

### En Hızlı Demo

```bash
streamlit run app/streamlit_app.py
```

Dashboard'u açın ve "Fetch Data" butonuna basın.

## Proje Sunumu İçin

### Hazırlık Listesi

- [ ] Tüm veriler toplandı mı?
- [ ] Modeller eğitildi mi?
- [ ] Jupyter notebook'lar çalışıyor mu?
- [ ] Streamlit app açılıyor mu?
- [ ] Sonuç grafikleri kaydedildi mi?

### Sunum Sırası

1. **Giriş** (2 dk)
   - Problem tanımı
   - Neden önemli?

2. **Metod** (5 dk)
   - Veri toplama
   - NLP sentiment analizi
   - LSTM modeli
   - Mimari diyagramı göster

3. **Sonuçlar** (5 dk)
   - Accuracy metrikleri
   - Grafikleri göster
   - Baseline vs Combined karşılaştırması

4. **Demo** (3 dk)
   - Streamlit uygulamasını çalıştır
   - Canlı tahmin göster

5. **Sonuç** (2 dk)
   - Başarılar
   - Zorluklar
   - Gelecek çalışmalar

### Demo İçin İpuçları

1. Önceden veri çekin (canlıda yavaş olabilir)
2. Grafikleri ekran görüntüsü olarak kaydedin (yedek için)
3. İnternet bağlantısı kontrolü yapın
4. Farklı hisse senetleriyle test edin

## Akademik Rapor İçin

### Bölümler

1. **Özet (Abstract)**
   - Problem
   - Yaklaşım
   - Sonuçlar

2. **Giriş**
   - Motivasyon
   - Literatür taraması
   - Katkılar

3. **Metod**
   - Veri toplama
   - Feature engineering
   - Model mimarisi
   - Eğitim prosedürü

4. **Deneyler**
   - Veri seti istatistikleri
   - Hyperparameter seçimi
   - Karşılaştırmalar

5. **Sonuçlar**
   - Metrikler (tablo)
   - Grafikler
   - Analiz

6. **Tartışma**
   - Başarılar
   - Limitasyonlar
   - Gelecek çalışmalar

7. **Sonuç**
   - Özet
   - Katkılar

### Tablolar ve Grafikler

Eklenecek görseller:
- Mimari diyagram (flowchart)
- Veri dağılımı histogram
- Sentiment analizi pie chart
- Eğitim loss grafikleri
- Tahmin vs gerçek grafikleri
- Confusion matrix
- Karşılaştırma tablosu

## İletişim

Sorunlar için:
1. Hata mesajını kaydedin
2. Hangi adımda olduğunuzu not edin
3. Python versiyonunu kontrol edin: `python --version`

Başarılar! 🎓

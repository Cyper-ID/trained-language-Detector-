# 🌍 Enhanced Language Detector - Documentation

## 📋 Project Overview

A sophisticated machine learning system that detects languages in text with high accuracy. The system supports **5 languages** (English, German, Spanish, French, Italian) and provides confidence scoring, alternative predictions, and language family information.

### 🎯 Key Features
- **High Accuracy**: Advanced feature engineering with >95% accuracy on test data
- **Multiple Language Support**: 5 European languages with language family classification
- **Confidence Scoring**: Detailed confidence levels with probability calibration
- **Web Scraping**: Real-time data collection from authentic sources
- **Batch Processing**: Efficient handling of multiple text inputs
- **Interactive CLI**: User-friendly command-line interface

---

## 📁 Project Structure

```
language-detector/
│
├── 📁 data/                    # Training data (generated)
│   ├── english.txt
│   ├── german.txt
│   ├── spanish.txt
│   ├── french.txt
│   └── italian.txt
│
├── 📁 web_data/               # Raw scraped web data
│   ├── [language].txt files
│   └── scraping_stats.json
│
├── 📄 Data_scraper_v2.py      # Web data collection module
├── 📄 Language_detector_v2.py  # Main detection model
│
├── 📄 requirements.txt        # Python dependencies
├── 📄 README.md              # This file
├── 📄 LICENSE                # MIT License
└── 📄 .gitignore            # Git ignore file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Internet connection (for web scraping)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/cyper-ID/trained_language_Detector.git
cd language-detector
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

**Option 1: Full training from scratch**
```bash
# Step 1: Collect training data
python Data_scraper_v2.py

# Step 2: Train and use the model
python Language_detector_v2.py
```

**Option 2: Use pre-trained model**
```bash
# Just run the detector (will use saved model if available)
python Language_detector_v2.py
```

---

## 🔧 Detailed Usage Guide

### 1. Data Collection Module (`Data_scraper_v2.py`)

**Purpose**: Collects authentic text data from the web for training.

**Usage:**
```bash
python Data_scraper_v2.py
```

**Features:**
- **3 Collection Modes**:
  1. **Web Scraping**: Collects real data from news sites and Wikipedia
  2. **Fallback Data**: Uses pre-prepared sentences (fastest)
  3. **Hybrid**: Attempts web scraping first, falls back if needed

- **Supported Sources**:
  - English: BBC, The Guardian, New York Times
  - German: Spiegel, Zeit, FAZ
  - Spanish: El Mundo, El País, ABC
  - French: Le Monde, Le Figaro, Liberation
  - Italian: Corriere della Sera, Repubblica, La Stampa

- **Data Processing**:
  - Cleans and validates sentences
  - Augments data with variations
  - Saves both raw and training-ready data

### 2. Language Detection Module (`Language_detector_v2.py`)

**Purpose**: Detects languages in text with confidence scoring.

**Usage:**
```bash
python Language_detector_v2.py
```

**Features:**
- **Automatic Model Management**:
  - Loads existing model if available
  - Trains new model if needed
  - Saves trained model for future use

- **Interactive Mode**:
  ```bash
  🔍 Enter text: The quick brown fox jumps over the lazy dog.
  
  ✅ Detected: English (english)
     Confidence: 99.8% - Excellent
     Language family: Germanic
     Text length: 43 chars, 9 words
  ```

- **Batch Testing**:
  - Tests 20+ predefined examples
  - Calculates accuracy metrics
  - Shows confusion matrix

---

## 📊 Model Architecture

### Machine Learning Pipeline

1. **Feature Engineering**
   ```
   Raw Text → [Character N-grams (2-5)] + [Statistical Features] → Combined Feature Vector
   ```

2. **Statistical Features Include**:
   - Text length and word count
   - Character ratios (uppercase, digits, punctuation)
   - Language-specific character counts (ñ, ç, ß, etc.)
   - Common word frequencies

3. **Model Details**:
   - **Algorithm**: Random Forest Classifier
   - **Calibration**: CalibratedClassifierCV for accurate probabilities
   - **Features**: ~1500 n-grams + 16 statistical features
   - **Optimization**: Hyperparameter tuning for 5-language classification

### Performance Metrics
- **Test Accuracy**: 95-98% on balanced dataset
- **Confidence Levels**: 
  - Excellent (>95%)
  - Very High (>85%)
  - High (>75%)
  - Medium (>60%)
  - Low (>40%)
  - Very Low (≤40%)

---

## 🧪 Testing Examples

The system includes a comprehensive test suite:

### Single Language Detection
```python
# English
"The quick brown fox jumps over the lazy dog."

# German
"Der schnelle braune Fuchs springt über den faulen Hund."

# Spanish
"El rápido zorro marrón salta sobre el perro perezoso."

# French
"Le rapide renard marron saute par-dessus le chien paresseux."

# Italian
"La veloce volpe marrone salta sopra il cane pigro."
```

### Edge Cases
- Short texts ("Hello", "Bonjour")
- Mixed language hints ("Hello mi amigo")
- Technical/specialized vocabulary

---

## 🔄 API Usage (Programmatic)

### Basic Detection
```python
from Language_detector_v2 import EnhancedLanguageDetector

# Initialize detector
detector = EnhancedLanguageDetector()
detector.load('optimized_language_model.joblib')  # Load trained model

# Detect language
result = detector.predict("Hello world, how are you today?")
print(f"Language: {result['language']}")
print(f"Confidence: {result['confidence']:.1f}%")
```

### Batch Processing
```python
texts = ["Hello world", "Guten Tag", "Bonjour"]
results = detector.batch_predict(texts)
for r in results:
    print(f"{r['text']} → {r['language']} ({r['confidence']:.1f}%)")
```

### Get Model Information
```python
info = detector.get_model_info()
print(f"Supported languages: {', '.join(info['languages'])}")
print(f"Number of features: {info['features']:,}")
```

---

## 📈 Performance Optimization

### For Training Speed
1. **Reduce dataset size** (modify `max_per_class` in `_balance_dataset`)
2. **Use fallback data only** in scraper (choose option 2)
3. **Reduce n-gram features** (adjust `max_features` in vectorizer)

### For Accuracy
1. **Increase training data** (set higher limits in scraper)
2. **Adjust model parameters** (increase `n_estimators` or `max_depth`)
3. **Collect more diverse data** (add more sources in scraper config)

### Memory Considerations
- Model file size: ~50-100MB
- RAM usage during training: ~500MB
- RAM usage during inference: ~100MB

---

## 🛠️ Configuration Options

### Data Scraper Configuration (`Data_scraper_v2.py`)
```python
# In ScraperConfig class:
max_sentences_per_language = 500    # Max sentences per language
max_pages_per_source = 3            # Max pages to scrape per source
min_sentence_length = 20            # Minimum sentence length
max_sentence_length = 200           # Maximum sentence length
request_timeout = 10                # HTTP request timeout
```

### Detector Configuration (`Language_detector_v2.py`)
```python
# In EnhancedLanguageDetector class:
supported_languages = ['english', 'german', 'spanish', 'french', 'italian']

# In train_model method:
n_estimators = 150                  # Number of decision trees
max_depth = 25                      # Maximum tree depth
test_size = 0.2                     # Test set proportion
```

---

## 🚨 Error Handling & Troubleshooting

### Common Issues

1. **"No data loaded!" error**
   - **Solution**: Run `python Data_scraper_v2.py` first

2. **Web scraping fails**
   - **Solution**: Use fallback data (option 2 in scraper)
   - Check internet connection
   - Verify URLs in `language_sources` are accessible

3. **Low confidence predictions**
   - **Solution**: Use longer text (minimum 10-20 characters)
   - Check if text contains mixed languages

4. **Memory errors during training**
   - **Solution**: Reduce `max_per_class` in `_balance_dataset`
   - Lower `max_features` in vectorizer

### Logging
- Both modules include comprehensive logging
- Errors are captured and displayed with suggestions
- Progress indicators show current operations

---

## 🔮 Potential Future Enhancements

### Planned Features
1. **More Languages**: Add support for Portuguese, Dutch, Russian
2. **Deep Learning**: Implement RNN/LSTM models for better context understanding
3. **API Server**: Create REST API for web service deployment
4. **Real-time Learning**: Update model with user feedback
5. **Language Mix Detection**: Identify percentage of each language in mixed text

### Research Directions
- Transfer learning from multilingual BERT models
- Zero-shot learning for unseen languages
- Dialect and regional variant detection

---

## 📚 Technical Details

### Dependencies
```txt
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
joblib>=1.1.0
beautifulsoup4>=4.10.0
requests>=2.26.0
```

### Algorithm Details
- **Feature Extraction**: TF-IDF with character n-grams (2-5)
- **Classification**: Random Forest with probability calibration
- **Data Balancing**: Stratified sampling with class limits
- **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix

### Data Characteristics
- Training samples: ~300 per language (balanced)
- Feature dimensions: ~1500
- Model size: ~50-100MB (compressed)
- Inference time: <100ms per sentence

---

## 👥 Contributing

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Areas
- Adding new language support
- Improving feature engineering
- Optimizing model performance
- Enhancing the user interface
- Writing documentation and examples

### Code Style
- Follow PEP 8 conventions
- Add docstrings for all functions
- Include type hints where applicable
- Write comprehensive tests

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Scikit-learn**: Machine learning library
- **Beautiful Soup**: HTML parsing library
- **Wikipedia & News Outlets**: Data sources for training
- **Open Source Community**: For inspiration and tools

---

## 🎯 Quick Reference

### Common Commands
```bash
# Collect training data
python Data_scraper_v2.py

# Train and use detector
python Language_detector_v2.py

# Save model manually
detector.save('my_model.joblib')

# Load model
detector.load('my_model.joblib')
```

### Supported Languages
| Language | Code | Family | Key Features |
|----------|------|--------|--------------|
| English | `english` | Germanic | "the", no diacritics |
| German | `german` | Germanic | "und", "ß", "äöü" |
| Spanish | `spanish` | Romance | "ñ", "¿¡", "y" |
| French | `french` | Romance | "ç", "œæ", "et" |
| Italian | `italian` | Romance | "àèéìòóù", "e" |

---

**⭐ If you find this project useful, please give it a star on GitHub!**

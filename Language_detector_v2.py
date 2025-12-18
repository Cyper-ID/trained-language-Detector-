"""
enhanced_language_detector.py
Enhanced Language Detector optimized for 5 languages
"""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from scipy.sparse import hstack
import json
import warnings
warnings.filterwarnings('ignore')

class EnhancedLanguageDetector:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.vectorizer = None
        self.model = None
        self.language_names = {}
        self.language_families = {
            'english': 'Germanic',
            'german': 'Germanic', 
            'spanish': 'Romance',
            'french': 'Romance',
            'italian': 'Romance'
        }
        self.supported_languages = ['english', 'german', 'spanish', 'french', 'italian']
        
    def load_data_from_files(self):
        """Load data from text files"""
        print("📚 Loading language data...")
        
        all_texts = []
        all_labels = []
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            print(f"❌ Error: '{self.data_dir}' directory not found!")
            print("Please run 'python web_data_scraper.py' first.")
            return None, None
        
        # Load only the 5 supported languages
        for language in self.supported_languages:
            filename = f"{language}.txt"
            filepath = os.path.join(self.data_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        sentences = [line.strip() for line in f if line.strip()]
                    
                    if sentences:
                        all_texts.extend(sentences)
                        all_labels.extend([language] * len(sentences))
                        self.language_names[language] = language.capitalize()
                        print(f"  ✓ {language.capitalize():12} - {len(sentences):4} sentences")
                        
                except Exception as e:
                    print(f"  ✗ Error reading {filename}: {e}")
            else:
                print(f"  ✗ File not found: {filename}")
        
        if not all_texts:
            print("❌ No data loaded!")
            print("Available files in data directory:")
            for f in os.listdir(self.data_dir):
                if f.endswith('.txt'):
                    print(f"  - {f}")
            return None, None
        
        # Balance the dataset
        balanced_texts, balanced_labels = self._balance_dataset(all_texts, all_labels)
        
        print(f"\n📊 Total: {len(balanced_texts)} sentences (balanced)")
        print(f"   Languages: {', '.join(self.language_names.values())}")
        
        return balanced_texts, balanced_labels
    
    def _balance_dataset(self, texts, labels, max_per_class=300):
        """Balance dataset by limiting samples per class"""
        from collections import defaultdict
        
        lang_texts = defaultdict(list)
        for text, label in zip(texts, labels):
            lang_texts[label].append(text)
        
        balanced_texts = []
        balanced_labels = []
        
        for lang, lang_samples in lang_texts.items():
            # Randomly sample if we have more than max_per_class
            if len(lang_samples) > max_per_class:
                import random
                selected = random.sample(lang_samples, max_per_class)
            else:
                selected = lang_samples
            
            balanced_texts.extend(selected)
            balanced_labels.extend([lang] * len(selected))
            
            print(f"  {lang.capitalize():12}: {len(lang_samples):4} → {len(selected):4} samples")
        
        return balanced_texts, balanced_labels
    
    def extract_advanced_features(self, texts):
        """Extract language-specific features"""
        features = []
        for text in texts:
            if not text:
                text = ""
            
            # Basic text statistics
            char_features = {
                'length': len(text),
                'word_count': len(text.split()),
                'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
                'space_ratio': text.count(' ') / len(text) if len(text) > 0 else 0,
                'digit_ratio': sum(c.isdigit() for c in text) / len(text) if len(text) > 0 else 0,
                'uppercase_ratio': sum(c.isupper() for c in text) / len(text) if len(text) > 0 else 0,
                'punctuation_ratio': sum(c in '.,;:!?' for c in text) / len(text) if len(text) > 0 else 0,
            }
            
            # Language-specific character patterns
            special_chars = {
                'accents_latin': sum(1 for c in text if c in 'áéíóúàèìòùâêîôûäëïöü'),
                'spanish_chars': sum(1 for c in text if c in 'ñ¿¡'),
                'french_chars': sum(1 for c in text if c in 'çœæ'),
                'german_chars': sum(1 for c in text if c in 'ßäöü'),
                'italian_chars': sum(1 for c in text if c in 'àèéìòóù'),
                'apostrophe': text.count("'") + text.count("’"),
            }
            
            # Common word patterns
            common_words = {
                'the_english': text.lower().count(' the '),
                'und_german': text.lower().count(' und '),
                'y_spanish': text.lower().count(' y ') + text.lower().count(' e '),
                'et_french': text.lower().count(' et '),
                'e_italian': text.lower().count(' e ') + text.lower().count(' ed '),
            }
            
            features.append({**char_features, **special_chars, **common_words})
        
        if features:
            feature_names = list(features[0].keys())
            feature_array = np.array([[f[name] for name in feature_names] for f in features])
            return feature_array
        return np.array([])
    
    def train_model(self, texts, labels, test_size=0.2):
        """Train optimized language detection model"""
        print("\n🔧 Training optimized model for 5 languages...")
        
        # Enhanced vectorizer with language-specific features
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',  # Word-boundary aware character n-grams
            ngram_range=(2, 5),
            max_features=1500,
            lowercase=True,
            min_df=2,
            max_df=0.9,
            sublinear_tf=True  # Use sublinear TF scaling
        )
        
        # Transform text to features
        X_ngrams = self.vectorizer.fit_transform(texts)
        
        # Extract advanced features
        X_additional = self.extract_advanced_features(texts)
        
        # Combine features
        X_combined = hstack([X_ngrams, X_additional])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, labels, 
            test_size=test_size, 
            random_state=42,
            stratify=labels
        )
        
        print(f"   Training: {X_train.shape[0]:,} samples")
        print(f"   Testing:  {X_test.shape[0]:,} samples")
        print(f"   Features: {X_combined.shape[1]:,}")
        print(f"   Languages: {len(set(labels))}")
        
        # Optimized Random Forest
        base_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced_subsample',
            n_jobs=-1,
            verbose=0
        )
        
        # Calibrate for better probabilities
        self.model = CalibratedClassifierCV(base_model, cv=5, n_jobs=-1)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Training complete!")
        print(f"📈 Accuracy: {accuracy:.2%}")
        
        # Detailed report
        print("\n📋 Performance by language:")
        target_names = [self.language_names[l] for l in sorted(set(labels))]
        print(classification_report(y_test, y_pred, target_names=target_names, digits=3))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=sorted(set(labels)))
        print("\n🎯 Confusion Matrix (rows=actual, columns=predicted):")
        print("     " + " ".join(f"{self.language_names[l][:3]:>3}" for l in sorted(set(labels))))
        for i, lang in enumerate(sorted(set(labels))):
            print(f"{self.language_names[lang][:3]:3} " + " ".join(f"{cm[i,j]:3}" for j in range(len(cm[i]))))
        
        return accuracy
    
    def predict(self, text, top_n=2, return_features=False):
        """Predict language with high accuracy"""
        if self.vectorizer is None or self.model is None:
            return {"error": "Model not trained"}
        
        text = text.strip()
        if not text:
            return {"error": "Empty text provided"}
        if len(text) < 3:
            return {"error": "Text too short (minimum 3 characters)"}
        
        # Extract features
        X_ngrams = self.vectorizer.transform([text])
        X_additional = self.extract_advanced_features([text])
        X_combined = hstack([X_ngrams, X_additional])
        
        # Predict
        lang_code = self.model.predict(X_combined)[0]
        probabilities = self.model.predict_proba(X_combined)[0]
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            code = self.model.classes_[idx]
            prob = probabilities[idx] * 100
            name = self.language_names.get(code, code)
            family = self.language_families.get(code, 'Unknown')
            top_predictions.append({
                'language': name,
                'code': code,
                'confidence': prob,
                'family': family
            })
        
        # Confidence level
        max_prob = probabilities.max() * 100
        if max_prob > 95:
            confidence_level = "Excellent"
        elif max_prob > 85:
            confidence_level = "Very High"
        elif max_prob > 75:
            confidence_level = "High"
        elif max_prob > 60:
            confidence_level = "Medium"
        elif max_prob > 40:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"
        
        result = {
            'text': text[:80] + "..." if len(text) > 80 else text,
            'language': self.language_names.get(lang_code, lang_code),
            'language_code': lang_code,
            'confidence': max_prob,
            'confidence_level': confidence_level,
            'language_family': self.language_families.get(lang_code, 'Unknown'),
            'top_predictions': top_predictions,
            'text_length': len(text),
            'word_count': len(text.split()),
        }
        
        # Warnings
        if len(text) < 10:
            result['warning'] = "Text is very short. Accuracy may be reduced."
        elif len(text) < 25:
            result['warning'] = "Text is short. For better accuracy, use longer text."
        
        # Detect potential mixed language
        if len(top_predictions) > 1:
            top1_conf = top_predictions[0]['confidence']
            top2_conf = top_predictions[1]['confidence']
            if top1_conf - top2_conf < 15:  # Close predictions
                result['note'] = "Text shows characteristics of multiple languages"
        
        return result
    
    def batch_predict(self, texts):
        """Batch prediction for efficiency"""
        if not texts:
            return []
        
        X_ngrams = self.vectorizer.transform(texts)
        X_additional = self.extract_advanced_features(texts)
        X_combined = hstack([X_ngrams, X_additional])
        
        predictions = self.model.predict(X_combined)
        probabilities = self.model.predict_proba(X_combined)
        
        results = []
        for i, text in enumerate(texts):
            lang_code = predictions[i]
            max_prob = probabilities[i].max() * 100
            
            results.append({
                'text': text[:50] + "..." if len(text) > 50 else text,
                'language': self.language_names.get(lang_code, lang_code),
                'confidence': max_prob,
                'language_code': lang_code,
            })
        
        return results
    
    def save(self, filename='optimized_language_model.joblib'):
        """Save trained model"""
        if self.model and self.vectorizer:
            data = {
                'vectorizer': self.vectorizer,
                'model': self.model,
                'language_names': self.language_names,
                'language_families': self.language_families,
                'supported_languages': self.supported_languages
            }
            joblib.dump(data, filename, compress=3)
            print(f"✅ Model saved to '{filename}'")
            return True
        return False
    
    def load(self, filename='optimized_language_model.joblib'):
        """Load trained model"""
        if os.path.exists(filename):
            try:
                data = joblib.load(filename)
                self.vectorizer = data['vectorizer']
                self.model = data['model']
                self.language_names = data['language_names']
                self.language_families = data.get('language_families', {})
                self.supported_languages = data.get('supported_languages', [])
                print(f"✅ Model loaded from '{filename}'")
                print(f"   Languages: {len(self.language_names)}")
                return True
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        return False
    
    def get_model_info(self):
        """Get model information"""
        return {
            'languages': list(self.language_names.values()),
            'language_codes': list(self.language_names.keys()),
            'language_families': list(set(self.language_families.values())),
            'features': self.vectorizer.get_feature_names_out().shape[0] if self.vectorizer else 0,
            'model_type': type(self.model).__name__ if self.model else 'Not trained',
        }

def main():
    """Main program"""
    print("=" * 70)
    print("OPTIMIZED LANGUAGE DETECTOR - 5 LANGUAGES")
    print("=" * 70)
    print("Supported languages: English, German, Spanish, French, Italian")
    print("=" * 70)
    
    detector = EnhancedLanguageDetector()
    
    # Try to load existing model
    model_file = 'optimized_language_model.joblib'
    if not detector.load(model_file):
        print("\n🔄 Training new optimized model...")
        
        # Load data
        texts, labels = detector.load_data_from_files()
        if texts is None:
            print("❌ Failed to load training data.")
            print("   Please run 'python web_data_scraper.py' first to collect data.")
            return
        
        # Train model
        detector.train_model(texts, labels)
        
        # Save model
        detector.save(model_file)
    
    # Show model info
    info = detector.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"  Languages: {', '.join(info['languages'])}")
    print(f"  Features: {info['features']:,}")
    print(f"  Model type: {info['model_type']}")
    
    # Comprehensive testing
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TESTING")
    print("=" * 70)
    
    test_suite = [
        # English tests
        ("The quick brown fox jumps over the lazy dog.", "English - Classic"),
        ("Machine learning algorithms improve with more data.", "English - Technical"),
        ("The weather is beautiful today in London.", "English - Casual"),
        
        # German tests
        ("Der schnelle braune Fuchs springt über den faulen Hund.", "German - Classic"),
        ("Maschinelles Lernen verbessert sich mit mehr Daten.", "German - Technical"),
        ("Das Wetter ist heute schön in Berlin.", "German - Casual"),
        
        # Spanish tests
        ("El rápido zorro marrón salta sobre el perro perezoso.", "Spanish - Classic"),
        ("Los algoritmos de aprendizaje automático mejoran con más datos.", "Spanish - Technical"),
        ("El clima está hermoso hoy en Madrid.", "Spanish - Casual"),
        
        # French tests
        ("Le rapide renard marron saute par-dessus le chien paresseux.", "French - Classic"),
        ("Les algorithmes d'apprentissage automatique s'améliorent avec plus de données.", "French - Technical"),
        ("Le temps est magnifique aujourd'hui à Paris.", "French - Casual"),
        
        # Italian tests
        ("La veloce volpe marrone salta sopra il cane pigro.", "Italian - Classic"),
        ("Gli algoritmi di apprendimento automatico migliorano con più dati.", "Italian - Technical"),
        ("Il tempo è bellissimo oggi a Roma.", "Italian - Casual"),
        
        # Short texts
        ("Hello", "Very short English"),
        ("Guten Tag", "Short German"),
        ("Hola amigos", "Short Spanish"),
        
        # Mixed/ambiguous
        ("Hello mi amigo", "English-Spanish mix"),
        ("Bonjour my friend", "French-English mix"),
    ]
    
    correct = 0
    total = len(test_suite)
    
    for text, description in test_suite:
        result = detector.predict(text)
        if 'error' not in result:
            expected = description.split(" - ")[0].lower()
            is_correct = expected in result['language'].lower() or result['language_code'] == expected
            
            if is_correct:
                correct += 1
                mark = "✓"
            else:
                mark = "✗"
            
            print(f"{mark} {description:30} → {result['language']:12} ({result['confidence']:.1f}%)")
            
            if not is_correct and result['top_predictions']:
                print(f"    Alternatives: {', '.join([f'{p['language']} ({p['confidence']:.1f}%)' for p in result['top_predictions'][1:]])}")
    
    accuracy = (correct / total) * 100
    print(f"\n📈 Test Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Enter text to detect language (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\n🔍 Enter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', '']:
                break
            
            if len(user_input) < 2:
                print("Please enter at least 2 characters.")
                continue
            
            result = detector.predict(user_input)
            
            if 'error' in result:
                print(f"❌ {result['error']}")
            else:
                print(f"\n✅ Detected: {result['language']} ({result['language_code']})")
                print(f"   Confidence: {result['confidence']:.1f}% - {result['confidence_level']}")
                print(f"   Language family: {result['language_family']}")
                print(f"   Text length: {result['text_length']} chars, {result['word_count']} words")
                
                if 'warning' in result:
                    print(f"   ⚠️  {result['warning']}")
                if 'note' in result:
                    print(f"   📝 {result['note']}")
                
                if result['confidence'] < 80 and len(result['top_predictions']) > 1:
                    print(f"\n   Alternative predictions:")
                    for pred in result['top_predictions'][1:]:
                        print(f"     • {pred['language']} ({pred['family']}): {pred['confidence']:.1f}%")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Save model
    print("\n💾 Saving model...")
    detector.save(model_file)
    print("✅ Done!")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import re
import nltk
import json
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict, Counter
import warnings
from datetime import datetime
import string

warnings.filterwarnings('ignore')

# دانلود منابع NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class ImprovedHashtagRecommendationSystem:
    def __init__(self):
        self.stemmer = PorterStemmer()
        # کاهش stop words فارسی
        self.stop_words = {'و', 'در', 'به', 'از', 'که', 'این', 'را', 'با', 'است', 'برای', 'آن', 'یک', 'خود', 'تا', 'کرد', 'بر', 'هم', 'نیز', 'می', 'ما', 'یا', 'شد'}
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,  # خودمان stop words را مدیریت می‌کنیم
            ngram_range=(1, 2),
            min_df=1,  # کاهش محدودیت فراوانی
            max_df=0.9,
            sublinear_tf=True
        )
        self.nb_classifier = None
        self.mlb = MultiLabelBinarizer()
        self.hashtag_counts = Counter()
        self.min_hashtag_freq = 1  # کاهش محدودیت فراوانی هشتگ‌ها
        
    def preprocess_text(self, text):
        """پیش‌پردازش بهبود یافته متن برای فارسی"""
        if pd.isna(text) or text == "":
            return ""

        # حذف URL ها و mentions
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|RT\s*', '', text)
        
        # حذف هشتگ‌ها از متن اصلی
        text = re.sub(r'#\w+', '', text)
        
        # نگه‌داری حروف فارسی، انگلیسی، اعداد و فاصله
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077Fa-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()

        if len(text) < 2:  # کاهش محدودیت طول
            return ""

        # تقسیم به کلمات
        words = text.split()
        
        # فیلتر کردن کلمات
        filtered_words = []
        for word in words:
            if (len(word) > 1 and  # کاهش محدودیت طول کلمه
                word not in self.stop_words and 
                not word.isdigit()):
                # stemming فقط برای کلمات انگلیسی
                if re.match(r'^[a-zA-Z]+$', word):
                    word = self.stemmer.stem(word)
                filtered_words.append(word)

        return ' '.join(filtered_words)

    def extract_hashtags(self, text):
        """استخراج هشتگ‌ها از متن"""
        if pd.isna(text):
            return []
        
        # الگوی بهتر برای هشتگ‌های فارسی/انگلیسی
        hashtags = re.findall(r'#[\w\u0600-\u06FF]+', text)
        normalized_hashtags = []
        
        for tag in hashtags:
            tag_clean = tag.lower().strip()
            if len(tag_clean) > 3:  # افزایش حداقل طول
                normalized_hashtags.append(tag_clean)
        
        return list(set(normalized_hashtags))

    def filter_frequent_hashtags(self, hashtag_lists):
        """فیلتر کردن هشتگ‌ها با محدودیت کمتر"""
        self.hashtag_counts = Counter()
        for hashtags in hashtag_lists:
            self.hashtag_counts.update(hashtags)
        
        total_tweets = len(hashtag_lists)
        self.min_hashtag_freq = 1  # حداقل یک بار تکرار
        
        frequent_hashtags = set([
            hashtag for hashtag, count in self.hashtag_counts.items() 
            if count >= self.min_hashtag_freq
        ])
        
        print(f"تعداد کل هشتگ‌های منحصربه‌فرد: {len(self.hashtag_counts)}")
        print(f"حداقل فراوانی: {self.min_hashtag_freq}")
        print(f"هشتگ‌های انتخاب شده: {len(frequent_hashtags)}")
        
        if self.hashtag_counts:
            print("هشتگ‌های پرفراوان:")
            for hashtag, count in self.hashtag_counts.most_common(10):
                print(f"  {hashtag}: {count}")
        
        return frequent_hashtags

    def prepare_training_data(self, texts, hashtag_lists):
        """آماده‌سازی داده‌های آموزش"""
        processed_texts = []
        for text in texts:
            processed = self.preprocess_text(text)
            processed_texts.append(processed)
        
        frequent_hashtags = self.filter_frequent_hashtags(hashtag_lists)
        
        filtered_texts = []
        filtered_hashtags = []
        
        for text, hashtags in zip(processed_texts, hashtag_lists):
            if len(text.strip()) > 5:  # حداقل طول بیشتر
                valid_hashtags = [h for h in hashtags if h in frequent_hashtags]
                if len(valid_hashtags) > 0:
                    filtered_texts.append(text)
                    filtered_hashtags.append(valid_hashtags)
        
        print(f"داده‌های آموزش: {len(filtered_texts)}")
        return filtered_texts, filtered_hashtags, frequent_hashtags

    def train_naive_bayes(self, X_train, y_train):
        """آموزش مدل Naive Bayes"""
        print("آموزش Naive Bayes...")
        
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        print(f"ابعاد ماتریس: {X_train_vectorized.shape}")
        
        y_train_binary = self.mlb.fit_transform(y_train)
        print(f"تعداد هشتگ‌های نهایی: {len(self.mlb.classes_)}")
        
        # بررسی توزیع برچسب‌ها
        label_counts = np.sum(y_train_binary, axis=0)
        print("توزیع برچسب‌ها:")
        for i, (label, count) in enumerate(zip(self.mlb.classes_, label_counts)):
            if i < 10:  # نمایش 10 برچسب اول
                print(f"  {label}: {count}")
        
        if y_train_binary.shape[1] == 1:
            self.nb_classifier = MultinomialNB(alpha=0.5)  # افزایش alpha
            self.nb_classifier.fit(X_train_vectorized, y_train_binary.ravel())
        else:
            self.nb_classifier = MultiOutputClassifier(
                MultinomialNB(alpha=0.5), 
                n_jobs=-1
            )
            self.nb_classifier.fit(X_train_vectorized, y_train_binary)
        
        print("✅ آموزش Naive Bayes تکمیل شد")

    def predict_hashtags_nb(self, texts, top_k=3, threshold=0.3):  # افزایش threshold
        """پیش‌بینی هشتگ با Naive Bayes"""
        if not texts:
            return []

        processed_texts = [self.preprocess_text(text) for text in texts]
        X_test_vectorized = self.vectorizer.transform(processed_texts)
        
        predictions = []

        try:
            if hasattr(self.nb_classifier, 'estimators_'):
                for i in range(len(texts)):
                    if len(processed_texts[i].strip()) < 5:
                        predictions.append([])
                        continue
                        
                    text_vector = X_test_vectorized[i:i + 1]
                    hashtag_scores = {}

                    for j, estimator in enumerate(self.nb_classifier.estimators_):
                        if j < len(self.mlb.classes_):
                            try:
                                proba = estimator.predict_proba(text_vector)
                                if len(proba[0]) > 1:
                                    score = proba[0][1]
                                    if score >= threshold:
                                        hashtag_scores[self.mlb.classes_[j]] = score
                            except:
                                continue

                    if hashtag_scores:
                        # اعمال تنوع در انتخاب
                        sorted_hashtags = sorted(hashtag_scores.items(),
                                              key=lambda x: x[1], reverse=True)
                        
                        selected_hashtags = []
                        for hashtag, score in sorted_hashtags:
                            if len(selected_hashtags) >= top_k:
                                break
                            # جلوگیری از انتخاب هشتگ‌های مشابه
                            if not any(hashtag[1:5] in selected[1:5] for selected in selected_hashtags):
                                selected_hashtags.append(hashtag)
                        
                        predictions.append(selected_hashtags)
                    else:
                        predictions.append([])
            else:
                probabilities = self.nb_classifier.predict_proba(X_test_vectorized)
                for i, proba in enumerate(probabilities):
                    if len(processed_texts[i].strip()) < 5:
                        predictions.append([])
                        continue
                    
                    if len(proba) > 1 and proba[1] >= threshold:
                        predictions.append(list(self.mlb.classes_))
                    else:
                        predictions.append([])

        except Exception as e:
            print(f"خطا در پیش‌بینی NB: {e}")
            predictions = [[] for _ in range(len(texts))]

        return predictions


class ImprovedCARHashtagRecommender:
    def __init__(self, min_support=0.01, min_confidence=0.1, min_lift=1.0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rules = []
        self.word_hashtag_rules = {}
        self.bigram_hashtag_rules = {}
        self.word_hashtag_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.bigram_hashtag_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.word_count = defaultdict(int)
        self.bigram_count = defaultdict(int)
        self.hashtag_count = defaultdict(int)

    def generate_association_rules(self, X_train, y_train):
        """تولید قوانین انجمنی با پشتیبانی از داده‌های کم"""
        print("تولید قوانین CAR...")

        total_docs = len(X_train)
        print(f"تعداد اسناد: {total_docs}")

        # محاسبه همبستگی‌ها
        for i, (text, hashtags) in enumerate(zip(X_train, y_train)):
            words = text.split()
            word_set = set(words)
            
            # bigram ها
            bigrams = set()
            for j in range(len(words) - 1):
                bigram = f"{words[j]} {words[j+1]}"
                bigrams.add(bigram)
            
            # شمارش کلمات و bigram ها
            for word in word_set:
                if len(word) > 2:
                    self.word_count[word] += 1
                    for hashtag in hashtags:
                        self.word_hashtag_cooccurrence[word][hashtag] += 1
            
            for bigram in bigrams:
                self.bigram_count[bigram] += 1
                for hashtag in hashtags:
                    self.bigram_hashtag_cooccurrence[bigram][hashtag] += 1

            # شمارش هشتگ‌ها
            for hashtag in hashtags:
                self.hashtag_count[hashtag] += 1

        print(f"کلمات: {len(self.word_count)}, Bigrams: {len(self.bigram_count)}, هشتگ‌ها: {len(self.hashtag_count)}")

        # تولید قوانین
        rules = []
        min_word_freq = max(1, int(total_docs * self.min_support))

        # قوانین تک‌کلمه‌ای
        for word, word_freq in self.word_count.items():
            if word_freq >= min_word_freq:
                word_support = word_freq / total_docs
                
                for hashtag, cooccur_count in self.word_hashtag_cooccurrence[word].items():
                    hashtag_freq = self.hashtag_count[hashtag]
                    
                    if hashtag_freq >= 1:  # حداقل یک بار تکرار
                        confidence = cooccur_count / word_freq
                        hashtag_support = hashtag_freq / total_docs
                        lift = confidence / hashtag_support if hashtag_support > 0 else 0

                        if confidence >= self.min_confidence and lift >= self.min_lift:
                            rules.append({
                                'antecedent': word,
                                'consequent': hashtag,
                                'support': word_support,
                                'confidence': confidence,
                                'lift': lift,
                                'count': cooccur_count,
                                'score': confidence * lift * cooccur_count,
                                'type': 'unigram'
                            })

        # قوانین دوکلمه‌ای
        for bigram, bigram_freq in self.bigram_count.items():
            if bigram_freq >= min_word_freq:
                bigram_support = bigram_freq / total_docs
                
                for hashtag, cooccur_count in self.bigram_hashtag_cooccurrence[bigram].items():
                    hashtag_freq = self.hashtag_count[hashtag]
                    
                    if hashtag_freq >= 1:  # حداقل یک بار تکرار
                        confidence = cooccur_count / bigram_freq
                        hashtag_support = hashtag_freq / total_docs
                        lift = confidence / hashtag_support if hashtag_support > 0 else 0

                        if confidence >= self.min_confidence and lift >= self.min_lift:
                            rules.append({
                                'antecedent': bigram,
                                'consequent': hashtag,
                                'support': bigram_support,
                                'confidence': confidence,
                                'lift': lift,
                                'count': cooccur_count,
                                'score': confidence * lift * cooccur_count * 1.2,  # وزن بیشتر برای bigram
                                'type': 'bigram'
                            })

        self.rules = sorted(rules, key=lambda x: x['score'], reverse=True)

        # تفکیک قوانین
        self.word_hashtag_rules = defaultdict(list)
        self.bigram_hashtag_rules = defaultdict(list)
        
        for rule in self.rules:
            if rule['type'] == 'unigram':
                self.word_hashtag_rules[rule['antecedent']].append(rule)
            else:
                self.bigram_hashtag_rules[rule['antecedent']].append(rule)

        print(f"قوانین تولید شده: {len(self.rules)}")
        print(f"قوانین تک‌کلمه‌ای: {len([r for r in self.rules if r['type'] == 'unigram'])}")
        print(f"قوانین دوکلمه‌ای: {len([r for r in self.rules if r['type'] == 'bigram'])}")
        
        if len(self.rules) > 0:
            print("\nنمونه قوانین:")
            for rule in self.rules[:5]:
                print(f"  {rule['antecedent']} -> {rule['consequent']} "
                      f"(conf: {rule['confidence']:.3f}, lift: {rule['lift']:.3f}, count: {rule['count']})")

    def predict_hashtags(self, texts, top_k=3, min_score_threshold=0.01):
        """پیش‌بینی هشتگ‌ها با CAR"""
        predictions = []

        for text in texts:
            if len(text.strip()) < 2:
                predictions.append([])
                continue
                
            words = text.split()
            word_set = set(words)
            
            # bigram ها
            bigrams = set()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigrams.add(bigram)
            
            hashtag_scores = defaultdict(float)
            matched_rules = []

            # قوانین تک‌کلمه‌ای
            for word in word_set:
                if word in self.word_hashtag_rules:
                    for rule in self.word_hashtag_rules[word]:
                        score = rule['confidence'] * rule['lift'] * rule['count']
                        if score >= min_score_threshold:
                            hashtag_scores[rule['consequent']] += score
                            matched_rules.append(rule)

            # قوانین دوکلمه‌ای
            for bigram in bigrams:
                if bigram in self.bigram_hashtag_rules:
                    for rule in self.bigram_hashtag_rules[bigram]:
                        score = rule['confidence'] * rule['lift'] * rule['count'] * 1.2
                        if score >= min_score_threshold:
                            hashtag_scores[rule['consequent']] += score
                            matched_rules.append(rule)

            if hashtag_scores:
                # نرمال‌سازی امتیازها
                max_score = max(hashtag_scores.values())
                if max_score > 0:
                    for hashtag in hashtag_scores:
                        hashtag_scores[hashtag] /= max_score

                # انتخاب بهترین هشتگ‌ها
                top_hashtags = sorted(hashtag_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                predictions.append([hashtag for hashtag, score in top_hashtags])
            else:
                predictions.append([])

        return predictions


def evaluate_predictions(y_true, y_pred, method_name):
    """ارزیابی پیش‌بینی‌ها"""
    print(f"\n=== ارزیابی {method_name} ===")
    
    total_predictions = len(y_pred)
    successful_predictions = sum(1 for pred in y_pred if len(pred) > 0)
    total_hashtags = sum(len(pred) for pred in y_pred)
    
    print(f"تعداد کل پیش‌بینی‌ها: {total_predictions}")
    print(f"پیش‌بینی‌های موفق: {successful_predictions} ({successful_predictions/total_predictions*100:.1f}%)")
    print(f"تعداد کل هشتگ‌های پیشنهادی: {total_hashtags}")
    print(f"میانگین هشتگ به ازای هر توییت: {total_hashtags/total_predictions:.2f}")
    
    unique_hashtags = set()
    for pred in y_pred:
        unique_hashtags.update(pred)
    print(f"تعداد هشتگ‌های منحصربه‌فرد: {len(unique_hashtags)}")
    
    # نمایش هشتگ‌های پرتکرار
    hashtag_frequency = Counter()
    for pred in y_pred:
        hashtag_frequency.update(pred)
    
    if hashtag_frequency:
        print("هشتگ‌های پرتکرار:")
        for hashtag, count in hashtag_frequency.most_common(10):
            print(f"  {hashtag}: {count}")
    
    return {
        'successful_predictions': successful_predictions,
        'total_hashtags': total_hashtags,
        'unique_hashtags': len(unique_hashtags),
        'success_rate': successful_predictions/total_predictions
    }


def save_results(nb_predictions, car_predictions, test_texts, nb_stats, car_stats):
    """ذخیره نتایج"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_filename = f"hashtag_report_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("=== گزارش سیستم پیشنهاد هشتگ ===\n")
        f.write(f"تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. آمار کلی:\n")
        f.write(f"تعداد کل توییت‌های تست: {len(test_texts)}\n\n")
        
        f.write("Naive Bayes:\n")
        f.write(f"  - پیش‌بینی‌های موفق: {nb_stats['successful_predictions']} ({nb_stats['success_rate']*100:.1f}%)\n")
        f.write(f"  - تعداد کل هشتگ‌ها: {nb_stats['total_hashtags']}\n")
        f.write(f"  - هشتگ‌های منحصربه‌فرد: {nb_stats['unique_hashtags']}\n\n")
        
        f.write("CAR:\n")
        f.write(f"  - پیش‌بینی‌های موفق: {car_stats['successful_predictions']} ({car_stats['success_rate']*100:.1f}%)\n")
        f.write(f"  - تعداد کل هشتگ‌ها: {car_stats['total_hashtags']}\n")
        f.write(f"  - هشتگ‌های منحصربه‌فرد: {car_stats['unique_hashtags']}\n\n")
        
        f.write("2. نمونه نتایج:\n")
        f.write("-" * 50 + "\n")
        
        sample_count = 0
        for i, (text, nb_pred, car_pred) in enumerate(zip(test_texts, nb_predictions, car_predictions)):
            if (len(nb_pred) > 0 or len(car_pred) > 0) and sample_count < 20:
                sample_count += 1
                f.write(f"\nنمونه {sample_count}:\n")
                f.write(f"متن: {text[:150]}...\n")
                f.write(f"Naive Bayes: {nb_pred}\n")
                f.write(f"CAR: {car_pred}\n")
                
                nb_set = set(nb_pred)
                car_set = set(car_pred)
                common = nb_set.intersection(car_set)
                if common:
                    f.write(f"مشترک: {list(common)}\n")
                f.write("-" * 40 + "\n")
    
    print(f"✅ گزارش ذخیره شد: {report_filename}")
    return report_filename


class DataCleaner:
    def __init__(self):
        # کلمات توقف فارسی
        self.persian_stopwords = {
            'و', 'در', 'به', 'از', 'که', 'این', 'را', 'با', 'است', 'برای',
            'آن', 'یک', 'خود', 'تا', 'کرد', 'بر', 'هم', 'نیز', 'می', 'ما', 'یا',
            'شد', 'اگر', 'همه', 'نه', 'دیگر', 'آنها', 'باید', 'هر', 'او', 'ما',
            'من', 'تو', 'چه', 'بود', 'کند', 'هی', 'شما', 'چرا', 'بی', 'بعد',
            'کنید', 'کنم', 'باشد', 'شده', 'شود', 'رفت', 'داشت', 'نمی', 'ولی',
            'چون', 'برابر', 'شدند', 'هستند', 'بودند', 'بسیار', 'دارد', 'همین',
            'میکنم', 'داد', 'نداره', 'میشه', 'هست', 'بهش', 'روی', 'کردن'
        }
        # کلمات توقف انگلیسی
        self.english_stopwords = set(stopwords.words('english'))
        self.all_stopwords = self.persian_stopwords.union(self.english_stopwords)
        
        # کاراکترهای خاص برای حذف
        self.special_chars = set(string.punctuation) - {'#'}
        
    def normalize_persian_text(self, text):
        """نرمال‌سازی متن فارسی"""
        # تبدیل کاراکترهای عربی به فارسی
        replacements = {
            'ي': 'ی',
            'ك': 'ک',
            'ة': 'ه',
            '١': '1',
            '٢': '2',
            '٣': '3',
            '٤': '4',
            '٥': '5',
            '٦': '6',
            '٧': '7',
            '٨': '8',
            '٩': '9',
            '٠': '0'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text
        
    def clean_text(self, text):
        """تمیز کردن متن"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # نرمال‌سازی متن فارسی
        text = self.normalize_persian_text(text)
        
        # حذف منشن‌ها و لینک‌ها
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'RT\s+', '', text)
        
        # حذف کاراکترهای خاص به جز #
        text = ''.join(ch for ch in text if ch not in self.special_chars)
        
        # حذف اعداد
        text = re.sub(r'\d+', '', text)
        
        # حذف فاصله‌های اضافی
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_hashtags(self, text):
        """استخراج هشتگ‌های منحصر به فرد"""
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        # پیدا کردن همه هشتگ‌ها
        hashtags = re.findall(r'#[\w\u0600-\u06FF_]+', text)
        
        # نرمال‌سازی و حذف تکراری‌ها
        normalized_hashtags = []
        for tag in hashtags:
            # نرمال‌سازی هشتگ
            tag = self.normalize_persian_text(tag)
            tag = tag.lower().strip()
            
            # فیلتر کردن هشتگ‌های نامعتبر
            if len(tag) > 2 and not any(ch in self.special_chars for ch in tag[1:]):
                normalized_hashtags.append(tag)
        
        # حذف تکراری‌ها با حفظ ترتیب
        return list(dict.fromkeys(normalized_hashtags))
    
    def tokenize_text(self, text):
        """توکنایز کردن متن"""
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        # توکنایز با NLTK
        tokens = word_tokenize(text)
        
        # فیلتر کردن توکن‌ها
        filtered_tokens = []
        for token in tokens:
            if (len(token) > 2 and  # حداقل 3 کاراکتر
                token not in self.all_stopwords and  # حذف کلمات توقف
                not any(ch in self.special_chars for ch in token) and  # حذف کاراکترهای خاص
                not token.startswith('#')):  # حذف هشتگ‌ها از متن
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def clean_dataset(self, input_file, output_file=None):
        """تمیز کردن کل دیتاست"""
        print("شروع تمیز کردن دیتاست...")
        
        # خواندن داده‌ها
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = []
                for line in f:
                    try:
                        tweet = json.loads(line.strip())
                        data.append(tweet)
                    except:
                        continue
        except Exception as e:
            print(f"خطا در خواندن فایل: {e}")
            return None
        
        print(f"تعداد توییت‌های اولیه: {len(data)}")
        
        # تبدیل به دیتافریم
        cleaned_data = []
        hashtag_counts = Counter()
        
        for tweet in data:
            # استخراج متن
            text = tweet.get('text', tweet.get('full_text', ''))
            
            # تمیز کردن متن
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                continue
            
            # استخراج هشتگ‌ها
            hashtags = self.extract_hashtags(text)
            if hashtags:
                hashtag_counts.update(hashtags)
            
            # توکنایز متن
            tokens = self.tokenize_text(cleaned_text)
            if not tokens:
                continue
            
            cleaned_data.append({
                'text': cleaned_text,
                'tokens': tokens,
                'hashtags': hashtags
            })
        
        print(f"تعداد توییت‌های تمیز شده: {len(cleaned_data)}")
        
        # فیلتر کردن هشتگ‌های کم‌تکرار
        min_hashtag_freq = 3
        frequent_hashtags = {tag for tag, count in hashtag_counts.items() if count >= min_hashtag_freq}
        print(f"تعداد هشتگ‌های منحصر به فرد با حداقل {min_hashtag_freq} تکرار: {len(frequent_hashtags)}")
        
        # فیلتر کردن توییت‌ها بر اساس هشتگ‌های پرتکرار
        filtered_data = []
        for item in cleaned_data:
            valid_hashtags = [tag for tag in item['hashtags'] if tag in frequent_hashtags]
            if valid_hashtags or not item['hashtags']:  # حفظ توییت‌های بدون هشتگ هم
                item['hashtags'] = valid_hashtags
                filtered_data.append(item)
        
        print(f"تعداد توییت‌های نهایی: {len(filtered_data)}")
        
        # نمایش نمونه هشتگ‌های پرتکرار
        print("\nنمونه هشتگ‌های پرتکرار:")
        for tag, count in sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            if count >= min_hashtag_freq:
                print(f"  {tag}: {count}")
        
        # ذخیره نتایج
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            print(f"\nداده‌های تمیز شده در فایل {output_file} ذخیره شدند.")
        
        return filtered_data

def main():
    """اجرای اصلی برنامه"""
    print("شروع برنامه...")
    
    # تمیز کردن داده‌ها
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_dataset('tweets.5000.json', 'cleaned_tweets.json')
    
    if not cleaned_data:
        print("خطا در تمیز کردن داده‌ها")
        return
    
    # تبدیل به دیتافریم
    df = pd.DataFrame(cleaned_data)
    print(f"\nتعداد کل توییت‌ها: {len(df)}")
    
    # جداسازی توییت‌های دارای هشتگ و بدون هشتگ
    tweets_with_hashtags = df[df['hashtags'].apply(len) > 0]
    tweets_without_hashtags = df[df['hashtags'].apply(len) == 0]
    
    print(f"توییت‌های دارای هشتگ: {len(tweets_with_hashtags)}")
    print(f"توییت‌های بدون هشتگ: {len(tweets_without_hashtags)}")
    
    # استفاده از 80% توییت‌های دارای هشتگ برای آموزش
    train_df, val_df = train_test_split(tweets_with_hashtags, test_size=0.2, random_state=42)
    test_df = tweets_without_hashtags
    
    print(f"\nتعداد داده‌های آموزش: {len(train_df)}")
    print(f"تعداد داده‌های اعتبارسنجی: {len(val_df)}")
    print(f"تعداد داده‌های تست: {len(test_df)}")
    
    # آموزش مدل‌ها
    print("\nآموزش مدل Naive Bayes...")
    nb_model = ImprovedHashtagRecommendationSystem()
    
    # استفاده از توکن‌های آماده به جای متن خام
    train_texts = [' '.join(tokens) for tokens in train_df['tokens']]
    train_hashtags = train_df['hashtags'].tolist()
    
    # آماده‌سازی داده‌های آموزش
    processed_texts, processed_hashtags, _ = nb_model.prepare_training_data(train_texts, train_hashtags)
    
    # آموزش مدل Naive Bayes
    nb_model.train_naive_bayes(processed_texts, processed_hashtags)
    
    print("\nآموزش مدل CAR...")
    car_model = ImprovedCARHashtagRecommender(
        min_support=0.005,
        min_confidence=0.1,
        min_lift=1.0
    )
    car_model.generate_association_rules(processed_texts, processed_hashtags)
    
    # پیش‌بینی برای داده‌های تست
    test_texts = [' '.join(tokens) for tokens in test_df['tokens']]
    
    print("\nپیش‌بینی با Naive Bayes...")
    nb_predictions = nb_model.predict_hashtags_nb(test_texts, top_k=3, threshold=0.1)
    
    print("\nپیش‌بینی با CAR...")
    car_predictions = car_model.predict_hashtags(test_texts, top_k=3, min_score_threshold=0.01)
    
    # ارزیابی نتایج
    nb_stats = evaluate_predictions(None, nb_predictions, "Naive Bayes")
    car_stats = evaluate_predictions(None, car_predictions, "CAR")
    
    # ذخیره نتایج
    report_file = save_results(nb_predictions, car_predictions, test_texts, nb_stats, car_stats)
    print(f"\nگزارش نهایی در فایل {report_file} ذخیره شد.")

if __name__ == "__main__":
    main()
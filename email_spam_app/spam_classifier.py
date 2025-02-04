import pandas as pd
import numpy as np
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# 1️⃣ # Đọc dữ liệu từ file
df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

# Chuyển đổi nhãn thành số (spam = 1, ham = 0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# 2️⃣ Tách tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 3️⃣ Xây dựng Pipeline xử lý văn bản và huấn luyện mô hình
model = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),  # Chuyển đổi văn bản thành vector
    ('tfidf', TfidfTransformer()),  # Áp dụng TF-IDF
    ('classifier', MultinomialNB())  # Mô hình Naive Bayes
])

# 4️⃣ Huấn luyện mô hình
model.fit(X_train, y_train)

# 5️⃣ Đánh giá mô hình
y_pred = model.predict(X_test)
print("Độ chính xác:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Lưu mô hình đã huấn luyện
joblib.dump(model, 'model/spam_classifier_model.pkl')

# 6️⃣ Thử nghiệm dự đoán
sample = ["Congratulations! You have won a free vacation."]
print("Kết quả:", model.predict(sample))  # 1 = spam, 0 = ham

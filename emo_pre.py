import streamlit as st
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np

st.set_page_config(page_title="SVM Cảm Xúc", page_icon="💬")
st.title("💬 Ứng dụng phân loại cảm xúc bình luận bằng SVM")

# ===================== BƯỚC 1: TIỀN XỬ LÝ ===================== #
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean_text(text) for text in X]

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ===================== BƯỚC 2: ĐỌC DỮ LIỆU TỪ FILE TXT ===================== #
manual_comments = []
manual_labels = []

# Đọc file txt (mỗi dòng: comment<TAB>label)
with open("comments_merged_1300.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        parts = line.strip().split("\t")
        if len(parts) == 2:
            comment, label = parts
            manual_comments.append(comment)
            manual_labels.append(int(label))
        else:
            st.warning(f"Dòng {i} bị lỗi định dạng và đã bị bỏ qua.")


all_comments = manual_comments
all_labels = manual_labels


# ===================== BƯỚC 3: HUẤN LUYỆN MÔ HÌNH ===================== #
X_train, X_test, y_train, y_test = train_test_split(all_comments, all_labels, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2)
classifier = SVC(kernel='linear', probability=True)

# Lưu riêng vectorizer để phân tích trọng số
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('cleaner', TextCleaner()),
    ('tfidf', vectorizer),
    ('svc', classifier)
])

model.fit(X_train, y_train)

# ===================== BƯỚC 4: BÁO CÁO KẾT QUẢ ===================== #
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose()

# ===================== BƯỚC 5: STREAMLIT UI ===================== #

with st.expander("📋 Mô tả quy trình tiền xử lý"):
    st.markdown("""
    **Dữ liệu được xử lý qua các bước sau:**
    - Chuyển văn bản thành chữ thường
    - Loại bỏ số, ký tự đặc biệt
    - Chuẩn hóa khoảng trắng
    - Vector hóa bằng TF-IDF
    """)

st.subheader("📊 Đánh giá mô hình trên tập kiểm tra:")
st.dataframe(df_report.style.highlight_max(axis=0))

st.subheader("📝 Nhập bình luận để phân tích cảm xúc:")
user_input = st.text_area("Ví dụ: Dịch vụ tệ quá, tôi không hài lòng chút nào...")

if st.button("Dự đoán"):
    prediction = model.predict([user_input])[0]
    prob = model.predict_proba([user_input])[0]
    label = "Tích cực 😊" if prediction == 1 else "Tiêu cực 😞"
    st.success(f"✅ Dự đoán: {label}")
    st.write(f"🔢 Xác suất: {round(prob[prediction]*100, 2)}%")

    # ============ PHÂN TÍCH TỪ ẢNH HƯỞNG ============ #
    st.subheader("🔍 Giải thích lý do dự đoán")

    st.write("🔠 **Top từ ảnh hưởng đến quyết định phân loại:**")
    vectorizer_fitted = model.named_steps['tfidf']
    classifier_fitted = model.named_steps['svc']
    feature_names = np.array(vectorizer_fitted.get_feature_names_out())

    coefs = classifier_fitted.coef_.toarray()[0]

    top_positive_idx = np.argsort(coefs)[-10:][::-1]
    top_negative_idx = np.argsort(coefs)[:10]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🔴 **Top từ tiêu cực:**")
        st.table(pd.DataFrame({
            "Từ": feature_names[top_negative_idx],
            "Trọng số": coefs[top_negative_idx]
        }))

    with col2:
        st.markdown("🟢 **Top từ tích cực:**")
        st.table(pd.DataFrame({
            "Từ": feature_names[top_positive_idx],
            "Trọng số": coefs[top_positive_idx]
        }))

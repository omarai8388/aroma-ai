import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="نظام توصية العطور والزيوت", layout="centered")
st.title("نظام توصية العطور والزيوت")

uploaded_file = st.file_uploader("ارفع ملف المنتجات (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("تم تحميل الملف بنجاح")

    if 'اسم' in df.columns and 'الوصف' in df.columns:
        choice = st.selectbox("اختر منتجاً لعرض المنتجات المشابهة", df["اسم"].values)
        if choice:
            idx = df[df["اسم"] == choice].index[0]
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(df["الوصف"])
            similar_indices = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten().argsort()[::-1][1:6]
            st.subheader("المنتجات المشابهة:")
            for i in similar_indices:
                st.write(df.iloc[i]["اسم"])

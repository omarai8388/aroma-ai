import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

st.set_page_config(page_title="نظام توصية العطور والزيوت", layout="centered")

st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#e74c3c;">نظام توصية العطور والزيوت</h1>
        <p style="font-size:18px;">اكتشف المنتجات المتشابهة بناءً على اسم المنتج الذي تختاره</p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("ارفع ملف CSV يحتوي على أسماء المنتجات", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("تم رفع الملف بنجاح!")
    if 'اسم المنتج' in df.columns:
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df['اسم المنتج'])
        similarity = cosine_similarity(tfidf_matrix)

        product = st.selectbox("اختر منتجاً:", df['اسم المنتج'].values)

        if st.button("عرض المنتجات المشابهة"):
            idx = df[df['اسم المنتج'] == product].index[0]
            sim_scores = list(enumerate(similarity[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
            similar_products = df.iloc[[i[0] for i in sim_scores]]

            st.subheader("المنتجات المشابهة:")
            st.table(similar_products)

            # زر تحميل التوصيات كـ CSV
            csv = similar_products.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="recommendations.csv">تحميل النتائج كملف CSV</a>', unsafe_allow_html=True)
    else:
        st.warning("الملف لا يحتوي على عمود 'اسم المنتج'. تأكد من التنسيق الصحيح.")

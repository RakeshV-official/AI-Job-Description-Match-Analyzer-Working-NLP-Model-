import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Job Match Analyzer")

st.title("💼 AI Resume vs Job Match Analyzer")

resume = st.text_area("Paste Resume Text")
job_desc = st.text_area("Paste Job Description")

if resume and job_desc:
    documents = [resume, job_desc]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    match_score = round(similarity[0][0] * 100, 2)

    st.subheader("📊 Match Score")
    st.progress(int(match_score))
    st.write(f"Similarity: {match_score}%")

    if match_score > 70:
        st.success("Excellent Match 🔥")
    elif match_score > 40:
        st.warning("Moderate Match ⚠️")
    else:
        st.error("Low Match ❌ Improve your resume")

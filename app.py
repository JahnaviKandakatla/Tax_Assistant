# app.py (Free, Local Version)

import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence-transformers import SentenceTransformer
import requests
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

# =============== SETTINGS ==================
st.set_page_config(page_title="💼 CA AI Assistant (Free)", layout="wide")

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("income_tax_act_2024_sections.csv")
    df.dropna(subset=["Text"], inplace=True)
    df["Text"] = df["Text"].astype(str)
    return df

df = load_data()

# ---- Create FAISS index ----
@st.cache_resource
def create_index(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["Text"].tolist(), show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))
    return model, index

model, index = create_index(df)

# ---- Semantic search ----
def semantic_search(query, top_k=5):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb, dtype="float32"), top_k)
    return df.iloc[I[0]]

# ---- Local AI using Ollama ----
def ask_llama(prompt, model_name="llama3"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt},
            stream=False,
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response from model.")
        else:
            return f"⚠️ Model error ({response.status_code}). Make sure Ollama is running."
    except Exception as e:
        return f"⚠️ Connection failed: {e}. Start Ollama with `ollama run llama3`."

# ---- Voice Input ----
def get_voice():
    r = sr.Recognizer()
    with sr.Microphone() as src:
        st.info("🎙️ Listening... Speak your question.")
        audio = r.listen(src)
        try:
            text = r.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except:
            st.error("Could not recognize your voice.")
            return ""

# ---- Text to Speech ----
def speak_text(text):
    tts = gTTS(text=text, lang="en")
    path = tempfile.mktemp(suffix=".mp3")
    tts.save(path)
    st.audio(path)

# ================= UI START ===================
st.title("💼 Chartered Accountant AI Assistant (Offline & Free)")
st.caption("Your AI tax guide powered by free local models (Llama 3 + FAISS + Streamlit)")

tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Chat Assistant",
    "📚 Income Tax Explorer",
    "🔍 Advanced Search",
    "🎙️ Voice Chat"
])

# ---- 🧠 Chat Assistant ----
with tab1:
    st.header("Ask About Income Tax")
    q = st.text_area("💬 Enter your question:")
    if st.button("Ask AI"):
        if q.strip():
            ctx_df = semantic_search(q, top_k=5)
            ctx = "\n\n".join(ctx_df["Text"].tolist())
            prompt = f"Answer clearly using the following sections of the Income Tax Act:\n\n{ctx}\n\nQuestion: {q}\n\nAnswer:"
            ans = ask_llama(prompt)
            st.success("✅ Answer:")
            st.write(ans)
            if st.checkbox("🔊 Read answer aloud"):
                speak_text(ans)
        else:
            st.warning("Please enter a question first.")

# ---- 📚 Income Tax Explorer ----
with tab2:
    st.header("📘 Explore Income Tax Act, 1961")
    c1, c2 = st.columns(2)
    with c1:
        keyword = st.text_input("🔍 Keyword:")
    with c2:
        section = st.text_input("📜 Section Number:")
    results = df.copy()
    if keyword:
        results = results[results["Text"].str.contains(keyword, case=False, na=False)]
    if section:
        results = results[results["Section"].astype(str).str.contains(section, case=False, na=False)]
    st.dataframe(results.head(30), use_container_width=True, height=400)

# ---- 🔍 Advanced Search ----
with tab3:
    st.header("Advanced Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        word = st.text_input("Keyword:")
    with col2:
        min_sec = st.text_input("Start Section:")
    with col3:
        max_sec = st.text_input("End Section:")
    res = df.copy()
    if word:
        res = res[res["Text"].str.contains(word, case=False, na=False)]
    if min_sec and max_sec:
        try:
            min_s, max_s = int(min_sec), int(max_sec)
            res = res[
                res["Section"].astype(str).apply(lambda x: x.isdigit() and min_s <= int(x) <= max_s)
            ]
        except:
            st.warning("⚠️ Invalid range entered.")
    st.dataframe(res.head(50), use_container_width=True, height=400)

# ---- 🎙️ Voice Chat ----
with tab4:
    st.header("🎤 Ask by Voice")
    if st.button("🎧 Start Listening"):
        spoken = get_voice()
        if spoken:
            ctx_df = semantic_search(spoken, top_k=5)
            ctx = "\n\n".join(ctx_df["Text"].tolist())
            prompt = f"Based on these sections:\n\n{ctx}\n\nQuestion: {spoken}\n\nAnswer briefly:"
            ans = ask_llama(prompt)
            st.success("💬 Response:")
            st.write(ans)
            speak_text(ans)

st.markdown("---")
st.caption("© 2025 CA AI Assistant | Free Edition | Powered by Llama 3 (Ollama) + FAISS + Streamlit")

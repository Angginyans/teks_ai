import streamlit as st
import nltk
import re
import pandas as pd
import string
import stanza
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import joblib

# Download resources
nltk.download('punkt')
nltk.download('stopwords')
stanza.download('id')

# Inisialisasi
stop_words = set(stopwords.words('indonesian'))
nlp = stanza.Pipeline(lang='id', processors='tokenize,pos')

# Load bobot POS dan stopword
bobot_pos = pd.read_csv('bobot_pos_tag.csv')
bobot_POS = dict(zip(bobot_pos['POS_Tag'], bobot_pos['Bobot']))

bobot_stopword = pd.read_csv('bobot_Stopword.csv')
bobot_SW = dict(zip(bobot_stopword['Stopword'], bobot_stopword['Bobot']))

# Load model
model = joblib.load('logistic_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Rata-rata panjang dari data latih
rata2_ai = 234.41333333333333
rata2_nonai = 106.9693094629156

# Fungsi preprocessing
def normalize_teks(teks):
    teks = teks.lower()
    teks = re.sub(r'[^a-zA-Z\s]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def remove_stopwords(teks):
    tokens = word_tokenize(teks)
    filtered = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(filtered)

def hitung_rasio_panjang(panjang_teks):
    jarak_ke_ai = abs(panjang_teks - rata2_ai)
    jarak_ke_nonai = abs(panjang_teks - rata2_nonai)
    return 1 if jarak_ke_ai < jarak_ke_nonai else 0

def hitung_skor_pos(teks, bobot_POS):
    doc = nlp(teks)
    tags = [word.upos for sent in doc.sentences for word in sent.words]
    tag_counter = Counter(tags)

    total_bobot = 0
    total_tag = 0
    for tag, jumlah in tag_counter.items():
        bobot = bobot_POS.get(tag, 0.5)
        total_bobot += bobot * jumlah
        total_tag += jumlah

    return total_bobot / total_tag if total_tag > 0 else 0

def hitung_skor_stopword(teks, bobot_SW):
    tokens = word_tokenize(teks.lower())
    stopword_tokens = [t for t in tokens if t in stop_words]
    token_counter = Counter(stopword_tokens)

    total_bobot = 0
    total_token = 0
    for token, jumlah in token_counter.items():
        bobot = bobot_SW.get(token, 0.5)
        total_bobot += bobot * jumlah
        total_token += jumlah

    return total_bobot / total_token if total_token > 0 else 0

def proses_teks(teks):
    teks_normal = normalize_teks(teks)
    teks_stopword = remove_stopwords(teks_normal)
    tokens = word_tokenize(teks_stopword)

    panjang_teks = len(teks_stopword)
    rasio_panjang = hitung_rasio_panjang(panjang_teks)
    rasio_pos = hitung_skor_pos(teks, bobot_POS)
    rasio_stopword = hitung_skor_stopword(teks, bobot_SW)

    teks_bow = vectorizer.transform([" ".join(tokens)])
    df_bow = pd.DataFrame(teks_bow.toarray(), columns=vectorizer.get_feature_names_out())

    df_fitur = pd.DataFrame([{
        'rasio_panjang_teks': rasio_panjang,
        'rasio_stopword': rasio_stopword,
        'rasio_pos_tagging': rasio_pos
    }])

    return pd.concat([df_bow, df_fitur], axis=1), rasio_panjang, rasio_pos, rasio_stopword, df_bow, df_fitur

# Fungsi untuk menghitung probabilitas dari masing-masing fitur
def hitung_prob_fitur_terpisah(df_bow, df_fitur):
    prob_fitur = {}

    # Probabilitas Bag of Word (fitur lain = 0)
    X_bow_only = pd.concat([df_bow, pd.DataFrame(0, index=[0], columns=df_fitur.columns)], axis=1)
    prob_fitur['Bag of Word'] = model.predict_proba(X_bow_only)[0][1]

    # Probabilitas rasio_panjang_teks saja
    df_fitur_panjang = pd.DataFrame({col: [0] for col in df_fitur.columns})
    df_fitur_panjang['rasio_panjang_teks'] = df_fitur['rasio_panjang_teks'].values
    X_panjang_only = pd.concat([pd.DataFrame(0, index=[0], columns=df_bow.columns), df_fitur_panjang], axis=1)
    prob_fitur['Rasio Panjang Teks'] = model.predict_proba(X_panjang_only)[0][1]

    # Probabilitas rasio_stopword saja
    df_fitur_stopword = pd.DataFrame({col: [0] for col in df_fitur.columns})
    df_fitur_stopword['rasio_stopword'] = df_fitur['rasio_stopword'].values
    X_stopword_only = pd.concat([pd.DataFrame(0, index=[0], columns=df_bow.columns), df_fitur_stopword], axis=1)
    prob_fitur['Rasio Stopword'] = model.predict_proba(X_stopword_only)[0][1]

    # Probabilitas rasio_pos_tagging saja
    df_fitur_pos = pd.DataFrame({col: [0] for col in df_fitur.columns})
    df_fitur_pos['rasio_pos_tagging'] = df_fitur['rasio_pos_tagging'].values
    X_pos_only = pd.concat([pd.DataFrame(0, index=[0], columns=df_bow.columns), df_fitur_pos], axis=1)
    prob_fitur['Rasio POS Tag'] = model.predict_proba(X_pos_only)[0][1]

    return prob_fitur

# Streamlit UI
st.title("Klasifikasi Teks: AI vs Non-AI")

tab1, tab2 = st.tabs(["Input Manual", "Upload File"])

with tab1:
    teks_input = st.text_area("Masukkan teks di sini:")
    if st.button("Prediksi (Teks Manual)"):
        if teks_input.strip() == "":
            st.warning("Silakan masukkan teks terlebih dahulu.")
        else:
            X_pred, rasio_panjang, rasio_pos, rasio_stopword, df_bow, df_fitur = proses_teks(teks_input)
            prob = model.predict_proba(X_pred)[0]
            label_pred = model.predict(X_pred)[0]
            prob_fitur = hitung_prob_fitur_terpisah(df_bow, df_fitur)

            st.subheader("Hasil Prediksi:")
            st.write(f"**Label Prediksi**: {'AI (1)' if label_pred == 1 else 'Non-AI (0)'}")
            st.write(f"**Peluang AI (label 1)**: `{prob[1]:.4f}`")
            st.write(f"**Peluang Non-AI (label 0)**: `{prob[0]:.4f}`")

            st.subheader("Probabilitas Berdasarkan Masing-masing Fitur:")
            for fitur, prob_ft in prob_fitur.items():
                st.write(f"**{fitur}** → Probabilitas AI: `{prob_ft:.4f}`")

with tab2:
    uploaded_file = st.file_uploader("Upload file CSV (dengan kolom 'teks')", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'teks' not in df.columns:
            st.error("File harus memiliki kolom 'teks'.")
        else:
            st.success(f"{len(df)} baris teks berhasil dimuat.")
            hasil_list = []
            for i, row in df.iterrows():
                teks = str(row['teks'])
                X_pred, rasio_panjang, rasio_pos, rasio_stopword, df_bow, df_fitur = proses_teks(teks)
                prob = model.predict_proba(X_pred)[0]
                label_pred = model.predict(X_pred)[0]
                prob_fitur = hitung_prob_fitur_terpisah(df_bow, df_fitur)

                hasil_list.append({
                    'teks': teks,
                    'label_asli': row.get('label', 'Tidak Tersedia'),
                    'label_prediksi': label_pred,
                    'prob_ai_total': prob[1],
                    'prob_nonai_total': prob[0],
                    'prob_bow': prob_fitur['Bag of Word'],
                    'prob_rasio_panjang': prob_fitur['Rasio Panjang Teks'],
                    'prob_rasio_stopword': prob_fitur['Rasio Stopword'],
                    'prob_rasio_pos': prob_fitur['Rasio POS Tag'],
                    'rasio_panjang': rasio_panjang,
                    'rasio_pos': rasio_pos,
                    'rasio_stopword': rasio_stopword
                })

            df_hasil = pd.DataFrame(hasil_list)
            st.subheader("Hasil Prediksi dari File:")
            st.dataframe(df_hasil)

            csv = df_hasil.to_csv(index=False).encode('utf-8')
            st.download_button("Unduh Hasil Prediksi (CSV)", csv, "hasil_prediksi.csv", "text/csv")
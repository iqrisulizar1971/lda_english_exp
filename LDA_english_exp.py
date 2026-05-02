import pandas as pd
import re
import os
import warnings
import pyLDAvis
import pyLDAvis.gensim_models as gensim_models_vis
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# 1. PENGATURAN FOLDER & PERINGATAN
warnings.filterwarnings('ignore')
# Sesuaikan path folder untuk menyimpan hasil eksperimen Kaggle Anda
folder_path = 'D:/UNPAM S2/Semester 3/ADVANCED NLP/LDA_english/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 2. PEMUATAN DATASET KAGGLE
# Pastikan file 'I3_Industrial_Safety_And_Health_Data.csv' berada di direktori kerja Anda
try:
    # Mengambil dataset IHM Stefanini dari Kaggle
    raw_df = pd.read_csv('IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv')
    # Mengambil kolom 'Description' sesuai permintaan
    df = pd.DataFrame({'text': raw_df['Description'].astype(str)})
    print(f"Dataset berhasil dimuat. Total baris: {len(df)}")
except FileNotFoundError:
    print("Error: File CSV tidak ditemukan. Pastikan file dari Kaggle ada di folder yang sama.")
    # Fallback ke data kosong jika file tidak ada agar script tidak crash
    df = pd.DataFrame({'text': []})

# 3. PREPROCESSING (DIUBAH KE ENGLISH)
try:
    # Menggunakan English Stopwords sesuai instruksi
    stop_words = set(stopwords.words('english'))
    # Menambahkan custom stopwords medis/industri jika diperlukan
    stop_words.update(['also', 'using', 'found', 'reported', 'due', 'resulted'])
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Case folding
    text = text.lower()
    # Menghapus karakter non-alfabet
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenisasi
    tokens = word_tokenize(text)
    # Lemmatisasi & Filter Stopwords Inggris
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

if not df.empty:
    df['processed_text'] = df['text'].apply(preprocess_text)

    # 4. MEMBUAT DICTIONARY & CORPUS
    dictionary = corpora.Dictionary(df['processed_text'])
    
    # Filter kata ekstrem: kata harus muncul di minimal 5 dokumen
    # dan tidak boleh muncul di lebih dari 50% total dokumen agar topik lebih spesifik.
    dictionary.filter_extremes(no_below=5, no_above=0.5) 
    
    corpus = [dictionary.doc2bow(doc) for doc in df['processed_text']]

    # 5. PELATIHAN MODEL LDA
    # Kita gunakan 5 topik untuk mendapatkan variasi risiko HSE yang lebih detail
    num_topics = 5 
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=2000, # Ditingkatkan karena data Kaggle lebih besar
        passes=20,      # Iterasi lebih banyak agar model lebih stabil
        alpha='auto',
        per_word_topics=True
    )

    # 6. PENYIMPANAN HASIL
    path_model = folder_path + 'lda_model_kaggle.model'
    path_dictionary = folder_path + 'lda_dict_kaggle.gensim'
    path_viz = folder_path + 'hse_visualization_kaggle.html'

    lda_model.save(path_model)
    dictionary.save(path_dictionary)

    # Simpan Visualisasi Interaktif
    try:
        vis_data = gensim_models_vis.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis_data, path_viz)
        print(f"Visualisasi berhasil dibuat: {path_viz}")
    except Exception as e:
        print(f"Gagal visualisasi: {e}")

    # 7. OUTPUT STATUS
    print("\nRingkasan Topik Industri (Kaggle HSE):")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topik {idx}: {topic}")
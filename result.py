import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import warnings
from transformers import EncoderDecoderModel, BertTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

warnings.filterwarnings('ignore')

# # Jika menggunakan GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Jika menggunakan DirectML
import torch_directml
device = torch_directml.device()
sections = ['latarbelakang', 'rumusanmasalah', 'tujuanpenelitian', 'rangkumanpenelitianterkait', 'metodologipenelitian']

def perform_lsa(df, n_components=2):
    tfidf = TfidfVectorizer().fit_transform(df['summary'])
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf)
    lsa_matrix_normalized = normalize(lsa_matrix)
    return lsa_matrix_normalized

# Function to load data
def load_data(section):
    return pd.read_csv(f'data/final-data/{section}.csv')

# Function to calculate cosine similarity
def calculate_cosine_similarity(df):
    tfidf = TfidfVectorizer().fit_transform(df['summary'])
    return cosine_similarity(tfidf, tfidf)

def load_bert_model(model_path):
    model = EncoderDecoderModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = model.to(device)
    return model, tokenizer

def generate_bert_summary(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=256,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load BERT models
bert_models = {section: load_bert_model(f"model/saved_model_{section}") for section in sections}

# Load all data
sections = ['latarbelakang', 'rumusanmasalah', 'tujuanpenelitian', 'rangkumanpenelitianterkait', 'metodologipenelitian']
data = {section: load_data(section) for section in sections}

# Load judul data
judul_data = pd.read_csv('data/clean-data-csv/judul.csv')

# Mapping for display names
section_display_names = {
    'latarbelakang': 'Latar Belakang',
    'rumusanmasalah': 'Rumusan Masalah',
    'tujuanpenelitian': 'Tujuan Penelitian',
    'rangkumanpenelitianterkait': 'Rangkuman Penelitian Terkait',
    'metodologipenelitian': 'Metodologi Penelitian'
}

# Streamlit app
st.title('Ringkasan Untuk Penilaian Proposal Kualifikasi')

# Document selection
documents = list(data['latarbelakang']['nama_dokumen'].unique())
selected_document = st.selectbox('Pilih Dokumen:', documents)

# Display judul
judul = judul_data[judul_data['nama_dokumen'] == selected_document]['kalimat'].values[0]
st.header(f"{judul}")

# Method selection
method = st.radio("Pilih Metode Ringkasan:", ('TF-IDF', 'BERT'))

# Display summaries
for section in sections:
    st.subheader(section_display_names[section])
    if method == 'TF-IDF':
        summary = data[section][data[section]['nama_dokumen'] == selected_document]['summary'].values[0]
    else:  # BERT
        text = data[section][data[section]['nama_dokumen'] == selected_document]['kalimat'].values[0]
        model, tokenizer = bert_models[section]
        summary = generate_bert_summary(text, model, tokenizer)
    st.write(summary)

# Analysis method selection
analysis_method = st.radio("Pilih Metode Analisis:", ('Cosine Similarity', 'Latent Semantic Analysis'))

# Calculate and display similarity matrix
st.subheader('Matriks Similaritas:')
summaries = [data[section][data[section]['nama_dokumen'] == selected_document]['summary'].values[0] for section in sections]
df_summaries = pd.DataFrame({'summary': summaries})

if analysis_method == 'Cosine Similarity':
    similarity_matrix = calculate_cosine_similarity(df_summaries)
else:  # LSA
    similarity_matrix = perform_lsa(df_summaries)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(similarity_matrix, annot=True, cmap='Blues', xticklabels=[section_display_names[s] for s in sections], yticklabels=[section_display_names[s] for s in sections], ax=ax)

# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

st.pyplot(fig)

# Explanations
st.subheader('Analisis Similaritas:')
pairs = [
    ('Latar Belakang', 'Rumusan Masalah'),
    ('Latar Belakang', 'Tujuan Penelitian'),
    ('Rumusan Masalah', 'Tujuan Penelitian'),
    ('Rangkuman Penelitian Terkait', 'Metodologi Penelitian'),
    ('Tujuan Penelitian', 'Metodologi Penelitian')
]

# Reverse mapping for finding indices
reverse_section_mapping = {v: k for k, v in section_display_names.items()}

for pair in pairs:
    idx1 = sections.index(reverse_section_mapping[pair[0]])
    idx2 = sections.index(reverse_section_mapping[pair[1]])
    similarity = similarity_matrix[idx1][idx2]
    st.write(f"{pair[0]} Dan {pair[1]}")
    st.write(f"Kemiripan dokumen antara {pair[0]} dan {pair[1]} sebesar {similarity:.2f} yang artinya " +
             ("similaritas antara sub bagian tersebut mendekati 1" if similarity > 0.5 else "similaritas antara sub bagian tersebut menjauhi 1"))

    if pair == ('Latar Belakang', 'Rumusan Masalah'):
        st.write("Keterangan tambahan: Similaritas antara Latar Belakang dan Rumusan Masalah menunjukkan seberapa baik rumusan masalah mencerminkan isu-isu yang diangkat dalam latar belakang.")

st.write("Note: Nilai similaritas berkisar antara 0 (tidak mirip sama sekali) hingga 1 (identik).")

if analysis_method == 'Latent Semantic Analysis':
    st.write("Analisis LSA: LSA mengungkap struktur semantik tersembunyi dalam teks. Nilai yang lebih tinggi menunjukkan kesamaan tema atau konsep yang lebih kuat antara bagian-bagian dokumen.")
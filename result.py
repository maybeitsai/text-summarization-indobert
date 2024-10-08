import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to load data
def load_data(section):
    return pd.read_csv(f'data/summary-csv/{section}.csv')

# Function to calculate cosine similarity
def calculate_cosine_similarity(df):
    tfidf = TfidfVectorizer().fit_transform(df['summary'])
    return cosine_similarity(tfidf, tfidf)

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

# Display summaries
for section in sections:
    st.subheader(section_display_names[section])
    summary = data[section][data[section]['nama_dokumen'] == selected_document]['summary'].values[0]
    st.write(summary)

# Calculate and display cosine similarity
st.subheader('Matriks Similaritas:')
summaries = [data[section][data[section]['nama_dokumen'] == selected_document]['summary'].values[0] for section in sections]
cosine_sim = calculate_cosine_similarity(pd.DataFrame({'summary': summaries}))

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cosine_sim, annot=True, cmap='Blues', xticklabels=[section_display_names[s] for s in sections], yticklabels=[section_display_names[s] for s in sections], ax=ax)

# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

st.pyplot(fig)

# Explanations
st.subheader('Analisis Similaritas:')
pairs = [
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
    similarity = cosine_sim[idx1][idx2]
    st.write(f"{pair[0]} Dan {pair[1]}")
    st.write(f"Kemiripan dokumen antara {pair[0]} dan {pair[1]} sebesar {similarity:.2f} yang artinya " +
             ("similaritas antara sub bagian tersebut mendekati 1" if similarity > 0.5 else "similaritas antara sub bagian tersebut menjauhi 1"))

st.write("Note: Nilai similaritas berkisar antara 0 (tidak mirip sama sekali) hingga 1 (identik).")
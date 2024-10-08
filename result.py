import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Function to load data
def load_data(section):
    return pd.read_csv(f'data/final-data/{section}.csv')

# Function to load BERT summaries
def load_bert_summaries(section):
    return pd.read_csv(f'data/output-bert/{section}.csv')

# Function to calculate cosine similarity
def calculate_cosine_similarity(df):
    tfidf = TfidfVectorizer().fit_transform(df)
    return cosine_similarity(tfidf, tfidf)

# Function to perform LSA
def perform_lsa(df, n_components=2):
    tfidf = TfidfVectorizer(ngram_range=(1, 3)).fit_transform(df)
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf)
    lsa_matrix_normalized = normalize(lsa_matrix)
    return cosine_similarity(lsa_matrix_normalized)

# Initialize session state
if 'submit' not in st.session_state:
    st.session_state.submit = False
if 'selected_document' not in st.session_state:
    st.session_state.selected_document = None
if 'method' not in st.session_state:
    st.session_state.method = 'Fitur Kalimat'
if 'analysis_method' not in st.session_state:
    st.session_state.analysis_method = 'Cosine Similarity'

# Load all data
sections = ['latarbelakang', 'rumusanmasalah', 'tujuanpenelitian', 'rangkumanpenelitianterkait', 'metodologipenelitian']
data = {section: load_data(section) for section in sections}
bert_summaries = {section: load_bert_summaries(section) for section in sections}

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

# Sidebar for selections
st.sidebar.header("Pilihan")

# Document selection with session state
documents = list(data['latarbelakang']['nama_dokumen'].unique())

# Menyimpan pilihan dokumen sementara, hanya akan diperbarui setelah tombol submit ditekan
selected_document = st.sidebar.selectbox(
    'Pilih Dokumen:', documents, index=documents.index(st.session_state.selected_document) if st.session_state.selected_document else 0
)

# Method selection (disimpan di session state hanya setelah submit ditekan)
method = st.sidebar.radio("Pilih Metode Ringkasan:", ('Fitur Kalimat', 'BERT'), key='temp_method')

# Analysis method selection (disimpan di session state hanya setelah submit ditekan)
analysis_method = st.sidebar.radio("Pilih Metode Analisis:", ('Cosine Similarity', 'Latent Semantic Analysis'), key='temp_analysis')

# Create two columns for Reset and Submit buttons
col1, col2 = st.sidebar.columns(2)

# Reset button
if col1.button("Reset"):
    # Clear the session state
    st.session_state.clear()
    st.session_state.submit = False

# Submit button
if col2.button("Submit"):
    # Simpan dokumen terpilih ke session state
    st.session_state.selected_document = selected_document
    st.session_state.method = method
    st.session_state.analysis_method = analysis_method
    st.session_state.submit = True

# Main content
if not st.session_state.submit or not all([st.session_state.selected_document, st.session_state.method, st.session_state.analysis_method]):
    st.write("Silakan pilih dokumen, metode ringkasan, dan metode analisis di sidebar, lalu tekan tombol Submit untuk melihat hasil.")
else:
    # Display judul
    judul = judul_data[judul_data['nama_dokumen'] == st.session_state.selected_document]['kalimat'].values[0]
    st.header(f"{judul}")

    # Display summaries
    for section in sections:
        st.subheader(section_display_names[section])
        if st.session_state.method == 'Fitur Kalimat':
            summary = data[section][data[section]['nama_dokumen'] == st.session_state.selected_document]['summary'].values[0]
        else:  # BERT
            summary = bert_summaries[section][bert_summaries[section]['nama_dokumen'] == st.session_state.selected_document]['summary_bert'].values[0]
        st.write(summary)

    # Update titles based on analysis method
    if st.session_state.analysis_method == 'Cosine Similarity':
        st.subheader('Matriks Similaritas:')
    else:  # LSA
        st.subheader('Matriks LSA:')
    
    # Calculate and display similarity matrix
    summaries = [data[section][data[section]['nama_dokumen'] == st.session_state.selected_document]['summary'].values[0] for section in sections]
    
    if st.session_state.analysis_method == 'Cosine Similarity':
        similarity_matrix = calculate_cosine_similarity(summaries)
    else:  # LSA
        similarity_matrix = perform_lsa(summaries)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap='Blues', xticklabels=[section_display_names[s] for s in sections], yticklabels=[section_display_names[s] for s in sections], ax=ax)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)

    # Analisis dengan subjudul yang lebih jelas
    if st.session_state.analysis_method == 'Cosine Similarity':
        st.subheader('Analisis Similaritas:')
    else:  # LSA
        st.subheader('Analisis Hubungan:')
    pairs = [
        ('Latar Belakang', 'Rumusan Masalah'),
        ('Latar Belakang', 'Tujuan Penelitian'),
        ('Rumusan Masalah', 'Tujuan Penelitian'),
        ('Rangkuman Penelitian Terkait', 'Metodologi Penelitian'),
        ('Tujuan Penelitian', 'Metodologi Penelitian')
    ]

    reverse_section_mapping = {v: k for k, v in section_display_names.items()}
    
    for pair in pairs:
        idx1 = sections.index(reverse_section_mapping[pair[0]])
        idx2 = sections.index(reverse_section_mapping[pair[1]])
        similarity = similarity_matrix[idx1][idx2]
        
        st.write(f"**{pair[0]} Dan {pair[1]}**")
        
        if st.session_state.analysis_method == 'Cosine Similarity':
            st.write(f"Kemiripan dokumen antara {pair[0]} dan {pair[1]} sebesar {similarity:.2f} yang artinya " +
                     ("similaritas mendekati 1" if similarity > 0.5 else "similaritas menjauhi 1"))
        else:  # LSA
            st.write(f"Keterhubungan dokumen antara {pair[0]} dan {pair[1]} sebesar {similarity:.2f} yang artinya " +
                     ("keterhubungan kuat" if similarity > 0.5 else "keterhubungan lemah"))
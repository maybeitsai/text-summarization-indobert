import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Sistem Penilaian Proposal Kualifikasi", page_icon="📚")

# Custom CSS to improve the look and feel
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        /* Base Styles */
body {
    font-family: 'Poppins', sans-serif;
    background-color: #f4f4f9;
    margin: 0;
    padding: 20px;
}

/* Heading 1 Styles */
h1 {
    color: #A020F0;
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 3px;
    background: linear-gradient(45deg, #A020F0, #FF69B4);
    -webkit-background-clip: text;
    color: transparent;
    margin-bottom: 20px;
    text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    animation: fadeIn 2s ease-in-out;
}

/* Heading 2 Styles */
h2 {
    color: #2563EB;
    font-size: 2.5rem;
    font-weight: 600;
    text-align: left;
    padding: 10px;
    background-color: #E0E7FF;
    border-left: 15px solid #2563EB;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    border-radius: 15px;
    margin: 20px 0;
    animation: slideIn 1.5s ease-in-out;
}

/* Animations */
@keyframes fadeIn {
    0% {
        opacity: 0;
        transform: translateY(-30px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideIn {
    0% {
        opacity: 0;
        transform: translateX(-50px);
    }
    100% {
        opacity: 1;
        transform: translateX(0);
    }
}

        
        .stButton>button {
            background-color: #3B82F6;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #1E40AF;
            color: 	#D3D3D3;
        }
        
        .css-1q8dd3e {
            background-color: #2563EB;
            border-radius: 10px;
        }

        .block-container {
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

sections = ['latarbelakang', 'rumusanmasalah', 'tujuanpenelitian', 'rangkumanpenelitianterkait', 'metodologipenelitian']

# Load data functions
def load_data(section):
    return pd.read_csv(f'data/final-data/{section}.csv')

def load_bert_summaries(section):
    return pd.read_csv(f'data/output-bert/{section}.csv')

# Similarity calculations
def calculate_cosine_similarity(df):
    tfidf = TfidfVectorizer().fit_transform(df)
    return cosine_similarity(tfidf, tfidf)

def perform_lsa(df, n_components=2):
    tfidf = TfidfVectorizer(ngram_range=(1, 3)).fit_transform(df)
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf)
    lsa_matrix_normalized = normalize(lsa_matrix)
    return cosine_similarity(lsa_matrix_normalized)

# Initialize session state
def initialize_session_state():
    if 'submit' not in st.session_state:
        st.session_state.submit = False
    if 'selected_document' not in st.session_state:
        st.session_state.selected_document = None
    if 'method' not in st.session_state:
        st.session_state.method = 'Fitur Kalimat'
    if 'analysis_method' not in st.session_state:
        st.session_state.analysis_method = 'Cosine Similarity'

# Load all datasets
def load_all_data(sections):
    data = {section: load_data(section) for section in sections}
    bert_summaries = {section: load_bert_summaries(section) for section in sections}
    judul_data = pd.read_csv('data/clean-data-csv/judul.csv')
    return data, bert_summaries, judul_data

# Display summary based on method selection
def display_summaries(data, bert_summaries, sections, section_display_names):
    for section in sections:
        st.subheader(section_display_names[section])
        if st.session_state.method == 'Fitur Kalimat':
            summary = data[section][data[section]['nama_dokumen'] == st.session_state.selected_document]['summary'].values[0]
        else:
            summary = bert_summaries[section][bert_summaries[section]['nama_dokumen'] == st.session_state.selected_document]['summary_bert'].values[0]
        st.write(summary)

# Display similarity matrix
def display_similarity_matrix(sections, section_display_names, summaries):
    if st.session_state.analysis_method == 'Cosine Similarity':
        similarity_matrix = calculate_cosine_similarity(summaries)
        st.subheader('Matriks Similaritas:')
    else:
        similarity_matrix = perform_lsa(summaries)
        st.subheader('Matriks LSA:')
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap='Blues', 
                xticklabels=[section_display_names[s] for s in sections], 
                yticklabels=[section_display_names[s] for s in sections], 
                ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

    return similarity_matrix

# Analyze similarities or relationships between sections
def analyze_relationships(similarity_matrix, section_display_names):
    st.subheader('Analisis Similaritas:' if st.session_state.analysis_method == 'Cosine Similarity' else 'Analisis Hubungan:')
    
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
        st.write(f"- **{pair[0]} Dan {pair[1]}**")
        if st.session_state.analysis_method == 'Cosine Similarity':
            st.write(f"Kemiripan dokumen antara {pair[0]} dan {pair[1]} sebesar {similarity:.2f}.")
        else:
            st.write(f"Keterhubungan dokumen antara {pair[0]} dan {pair[1]} sebesar {similarity:.2f}.")

def display_qualification_assessment():
    st.subheader("Keterangan Penilaian:")
    st.write("- A(>= 85)")
    st.write("- B(75 - 84)")
    st.write("- C(< 75)")
    
    assessment_criteria = [
        "Nilai isi disertasi",
        "Nilai Penguasaan Materi dan Metode Penelitian",
        "Nilai Kontribusi hasil Penelitian bagi ilmu pengetahuan",
        "Nilai Kontribusi hasil Penelitian bagi masyarakat",
        "Nilai wawasan pengetahuan konsep ilmu komputer",
        "Nilai Kemampuan untuk menangkap, menganalisis, dan menjawab pertanyaan/sanggahan"
    ]
    
    grades = {}
    for criterion in assessment_criteria:
        grades[criterion] = st.selectbox(f"{criterion}:", ["A", "B", "C"], key=criterion)
    
    if st.button("Simpan Penilaian"):
        total_score = sum([4 if grade == "A" else 3 if grade == "B" else 2 for grade in grades.values()])
        average_score = total_score / len(assessment_criteria)
        recommendation = "Layak" if average_score >= 3 else "Tidak Layak"
        
        st.subheader("Hasil Penilaian:")
        for criterion, grade in grades.items():
            st.write(f"{criterion}: {grade}")
        if recommendation == "Layak":
            st.success(f"Hasil Penilaian: Rekomendasi **{recommendation}**")
        else:
            st.error(f"Hasil Penilaian: Rekomendasi **{recommendation}**")

def main():
    st.title('Sistem Ringkasan dan Penilaian Proposal Kualifikasi')

    initialize_session_state()

    # Sections and display names mapping
    section_display_names = {
        'latarbelakang': 'Latar Belakang',
        'rumusanmasalah': 'Rumusan Masalah',
        'tujuanpenelitian': 'Tujuan Penelitian',
        'rangkumanpenelitianterkait': 'Rangkuman Penelitian Terkait',
        'metodologipenelitian': 'Metodologi Penelitian'
    }

    data, bert_summaries, judul_data = load_all_data(sections)

    # Sidebar for selections with improved styling
    with st.sidebar:
        st.sidebar.image("images\logo_gunadarma.png", width=150)
        st.sidebar.header("🛠️ Pilihan")
        
        documents = list(data['latarbelakang']['nama_dokumen'].unique())
        selected_document = st.selectbox('📄 Pilih Dokumen:', documents, 
                                         index=documents.index(st.session_state.selected_document) if st.session_state.selected_document else 0)

        method = st.radio("📊 Pilih Metode Ringkasan:", ('Fitur Kalimat', 'BERT'), key='temp_method')
        
        analysis_method = st.radio("🔍 Pilih Metode Analisis:", ('Cosine Similarity', 'Latent Semantic Analysis'), key='temp_analysis')

        col1, col2 = st.columns(2)
        
        if col1.button("🔄 Reset"):
            st.session_state.clear()
            st.session_state.submit = False

        if col2.button("✅ Submit"):
            st.session_state.selected_document = selected_document
            st.session_state.method = method
            st.session_state.analysis_method = analysis_method
            st.session_state.submit = True

    # Main content
    if not st.session_state.submit or not all([st.session_state.selected_document, st.session_state.method, st.session_state.analysis_method]):
        st.info("👈 Silakan pilih dokumen, metode ringkasan, dan metode analisis di sidebar, lalu tekan tombol Submit untuk melihat hasil.")
    else:
        tab1, tab2 = st.tabs(["📝 Ringkasan Proposal", "🏆 Penilaian Proposal"])
        
        with tab1:
            st.header("📝 Ringkasan Proposal Kualifikasi")
            
            judul = judul_data[judul_data['nama_dokumen'] == st.session_state.selected_document]['kalimat'].values[0]
            st.subheader(f"{judul}")
            
            for section in sections:
                with st.expander(f"{section_display_names[section]} 👇"):
                    display_summaries(data, bert_summaries, [section], section_display_names)

            st.subheader("📊 Analisis Similaritas")
            summaries = [data[section][data[section]['nama_dokumen'] == st.session_state.selected_document]['summary'].values[0] for section in sections]
            similarity_matrix = display_similarity_matrix(sections, section_display_names, summaries)

            st.subheader("🔍 Analisis Hubungan Antar Bagian")
            analyze_relationships(similarity_matrix, section_display_names)

        with tab2:
            st.header("🏆 Penilaian Proposal Kualifikasi")
            display_qualification_assessment()

if __name__ == "__main__":
    main()
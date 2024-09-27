import os
import re
import csv
import pandas as pd
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer dan stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Unduh stopwords bahasa Indonesia dari NLTK
nltk.download('stopwords')
indonesian_stopwords = set(stopwords.words('indonesian'))

def process_and_save_title_csv(input_folder, output_csv_path, clean=False):
    if not os.path.exists(input_folder):
        print(f"Folder {input_folder} tidak ditemukan.")
        return

    # Membuka file CSV untuk dituliskan
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['nama_dokumen', 'kalimat'])  # Header CSV

        # Iterasi melalui setiap file di folder judul
        for filename in os.listdir(input_folder):
            file_path = os.path.join(input_folder, filename)

            if os.path.isfile(file_path) and filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().strip()

                if clean:
                    # Bersihkan stopwords dan lakukan stemming
                    cleaned_text = clean_stopwords_and_stemming(text)
                else:
                    cleaned_text = text

                # Menyimpan nama dokumen dan kalimat dalam format CSV
                writer.writerow([filename, cleaned_text])

    print(f"Proses selesai! File judul disimpan sebagai {output_csv_path}.")

def load_title_words_from_csv(csv_path):
    """
    Memuat judul dari file CSV dan mengembalikan dictionary dengan nama dokumen sebagai kunci
    dan daftar kata-kata judul sebagai nilai.
    """
    df = pd.read_csv(csv_path)
    title_dict = {}
    
    for _, row in df.iterrows():
        doc_name = row['nama_dokumen']
        title_words = row['kalimat'].split()  # Memecah judul menjadi kata-kata
        title_dict[doc_name] = title_words
    
    return title_dict

def save_to_csv_with_features(input_folder, output_folder, section_name, title_dict, cue_words, clean=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_file_path = os.path.join(output_folder, f"{section_name}.csv")
    
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['nama_dokumen', 'kalimat', 'index', 'important', 'length', 'words_in_title', 'position', 'cue_words'])

        for filename in os.listdir(input_folder):
            file_path = os.path.join(input_folder, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                # Memecah teks menjadi kalimat
                sentences = split_sentences(text)

                if clean:
                    sentences = [clean_stopwords_and_stemming(sentence) for sentence in sentences]

                # Ambil title_words berdasarkan nama dokumen
                title_words = title_dict.get(filename, [])  # Jika tidak ditemukan, kembalikan list kosong
                # Ekstraksi fitur untuk setiap kalimat
                features = extract_features(sentences, title_words, cue_words)
                for sentence, feature in zip(sentences, features):
                    writer.writerow([filename, sentence, feature['index'], feature['important'], feature['length'], 
                                     feature['words_in_title'], feature['position'], feature['cue_words']])

    print(f"Data untuk bagian '{section_name}' disimpan di {csv_file_path}.")

def split_sentences(text):
    """
    Memecah teks menjadi kalimat-kalimat menggunakan tanda titik sebagai pemisah.
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [sentence.strip() for sentence in sentences if sentence]

def clean_stopwords_and_stemming(sentence):
    """
    Bersihkan teks dengan melakukan stemming dan penghapusan stopwords.
    """
    # Split sentence menjadi kata-kata
    words = sentence.split()
    
    # Lakukan stemming pada setiap kata dan hapus stopwords
    cleaned_words = [stemmer.stem(word) for word in words if word.lower() not in indonesian_stopwords]
    
    # Gabungkan kembali kata-kata yang telah dibersihkan menjadi kalimat
    return ' '.join(cleaned_words)

def extract_features(sentences, title_words, cue_words):
    """
    Fungsi untuk mengekstrak fitur dari setiap kalimat.
    """
    features = []

    # Define keywords for unimportant information (tanpa stemming)
    unimportant_words = ["contohnya", "sebagai contoh", "contoh", "misalnya", "misal", "misalkan"]

    # Iterate over each sentence to extract features
    for i, sentence in enumerate(sentences):
        sentence_features = {}
        Mp = len(sentences)  # Total number of sentences

        # Sentence Index (B2i formula)
        if Mp > 1:
            sentence_features['index'] = i
        else:
            sentence_features['index'] = 1  # If there's only one sentence

        # Feature 1: Unimportant Information (menggunakan kalimat asli)
        if any(word in sentence.lower() for word in unimportant_words):
            sentence_features['important'] = 0
        else:
            sentence_features['important'] = 1

        # Bersihkan kalimat dengan stemming dan hapus stopword hanya untuk fitur
        cleaned_sentence = clean_stopwords_and_stemming(sentence)
        cleaned_words = cleaned_sentence.split()

        # Feature 2: Sentence Length
        sentence_features['length'] = 0 if len(cleaned_words) < 6 else len(cleaned_words)

        # Feature 3: Words in Title (Bi formula: intersection over union)
        title_count = len(set(cleaned_words).intersection(set(title_words)))  # Perbandingan menggunakan cleaned_words
        union_count = len(set(cleaned_words).union(set(title_words)))
        sentence_features['words_in_title'] = title_count / (union_count if union_count > 0 else 1)

        # Feature 4: Sentence Position (B2i formula - parabola terbalik)
        ji = i + 1  # Sentence index, starting from 1
        if Mp > 1:
            middle_index = (Mp + 1) / 2  # Middle sentence index
            sentence_features['position'] = abs(ji - middle_index) / (Mp / 2)  # Parabola formula
        else:
            sentence_features['position'] = 1  # If there's only one sentence

        # Feature 5: Cue Words (B3i formula)
        cue_count = sum(1 for word in cue_words if word in cleaned_sentence.lower())
        Tfi = sum(1 for sentence in sentences for word in cue_words if word in cleaned_sentence.lower())
        sentence_features['cue_words'] = cue_count / (Tfi if Tfi > 0 else 1)

        features.append(sentence_features)
    
    return features
import os
import re
import csv
import pandas as pd
import nltk
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
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
        writer.writerow(['nama_dokumen', 'kalimat', 'index', 'important', 'conjunction', 'length', 'words_in_title', 'position', 'cue_words'])

        for filename in os.listdir(input_folder):
            file_path = os.path.join(input_folder, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                sentences = split_sentences(text)

                if clean:
                    sentences = [clean_stopwords_and_stemming(sentence) for sentence in sentences]

                title_words = title_dict.get(filename, [])
                features = extract_features(sentences, title_words, cue_words, section_name)
                for sentence, feature in zip(sentences, features):
                    writer.writerow([filename, sentence, feature['index'], feature['important'], feature['conjunction'], feature['length'], 
                                     feature['words_in_title'], feature['position'], feature['cue_words']])

    print(f"Data untuk bagian '{section_name}' disimpan di {csv_file_path}.")

def split_sentences(text):
    """
    Memecah teks menjadi kalimat-kalimat menggunakan NLTK sent_tokenize.
    """
    sentences = sent_tokenize(text)
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

def extract_features(sentences, title_words, cue_words, section_name):
    features = []
    unimportant_words = ["contohnya", "sebagai contoh", "contoh", "misalnya", "misal", "misalkan"]
    conjunction_words = ["dan", "tetapi", "atau", "melainkan", "serta", "karena", "jika", "agar",
                         "meskipun", "walaupun", "sehingga", "supaya", "setelah", "sebelum", "sejak",
                         "ketika", "sebelum", "sesudah", "sejak", "sampai", "sementara", "tatkala", "sewaktu",
                         "oleh karena itu", "dengan demikian", "namun", "akan tetapi", "selain itu", "bahkan",
                         "maupun", "semakin"]

    Mp = len(sentences)

    for i, sentence in enumerate(sentences):
        sentence_features = {}
        original_words = word_tokenize(sentence.lower())
        cleaned_sentence = clean_stopwords_and_stemming(sentence)
        cleaned_words = cleaned_sentence.split()

        sentence_features['index'] = i if Mp > 1 else 1

        sentence_features['important'] = 1 if not any(re.search(r'\b' + re.escape(word) + r'\b', sentence.lower()) for word in unimportant_words) else 0
        
        sentence_features['conjunction'] = 0 if any(re.search(r'\b' + re.escape(word) + r'\b', sentence.lower()) for word in conjunction_words) else 1
        
        sentence_features['length'] = len(original_words)
        
        title_count = len(set(cleaned_words).intersection(set(title_words)))
        union_count = len(set(cleaned_words).union(set(title_words)))
        sentence_features['words_in_title'] = title_count / (union_count if union_count > 0 else 1)

        sentence_features['position'] = calculate_sentence_position(i, Mp)

        cue_count = sum(1 for word in cue_words if word in cleaned_sentence.lower())
        Tfi = sum(1 for sentence in sentences for word in cue_words if word in clean_stopwords_and_stemming(sentence).lower())
        sentence_features['cue_words'] = cue_count / (Tfi if Tfi > 0 else 1)

        if section_name in ['latarbelakang', 'metodologipenelitian']:
            if sentence_features['important'] == 1 and sentence_features['conjunction'] == 1 and sentence_features['length'] > 6:
                features.append(sentence_features)
        else:
            if sentence_features['important'] == 1 and sentence_features['length'] > 6:
                features.append(sentence_features)
    
    return features

def calculate_sentence_position(i, Mp):
    """
    Menghitung posisi kalimat dalam paragraf.
    
    :param i: Indeks kalimat (dimulai dari 0)
    :param Mp: Jumlah total kalimat dalam paragraf
    :return: Nilai posisi kalimat (1 untuk awal dan akhir, 0.5 untuk tengah)
    """
    if Mp <= 1:
        return 1  # Jika hanya ada satu kalimat, nilainya 1

    # Normalisasi posisi ke range [0, 1]
    normalized_position = i / (Mp - 1)
    
    # Rumus parabola terbalik: 4x(1-x)
    # Ini akan menghasilkan 1 untuk x=0 dan x=1, dan 0.5 untuk x=0.5
    position_value = 4 * normalized_position * (1 - normalized_position)
    
    return position_value
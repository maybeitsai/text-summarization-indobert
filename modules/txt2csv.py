import os
import re
import csv
import pandas as pd
import nltk
nltk.download('punkt')

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
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

        # Definisikan fungsi untuk memproses setiap file
        def process_file(filename):
            file_path = os.path.join(input_folder, filename)
            if os.path.isfile(file_path) and filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().strip()

                if clean:
                    # Bersihkan stopwords dan lakukan stemming dengan multiprocessing
                    with ProcessPoolExecutor() as process_executor:
                        cleaned_text = process_executor.submit(clean_stopwords_and_stemming, text).result()
                else:
                    cleaned_text = text

                return filename, cleaned_text
            return None

        # Menggunakan ThreadPoolExecutor untuk memproses file secara paralel (I/O-bound)
        with ThreadPoolExecutor() as thread_executor:
            futures = {thread_executor.submit(process_file, filename): filename for filename in os.listdir(input_folder)}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    filename, cleaned_text = result
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

        # Fungsi untuk memproses setiap file
        def process_file(filename):
            file_path = os.path.join(input_folder, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                sentences = split_sentences(text)

                if clean:
                    # Bersihkan stopwords dan lakukan stemming dengan multiprocessing
                    with ProcessPoolExecutor() as process_executor:
                        sentences = list(process_executor.map(clean_stopwords_and_stemming, sentences))

                title_words = title_dict.get(filename, [])
                features = extract_features(sentences, title_words, cue_words, section_name)

                return [(filename, sentence, feature) for sentence, feature in zip(sentences, features)]
            return []

        # Menggunakan ThreadPoolExecutor untuk memproses file secara paralel (I/O-bound)
        with ThreadPoolExecutor() as thread_executor:
            futures = {thread_executor.submit(process_file, filename): filename for filename in os.listdir(input_folder)}

            for future in as_completed(futures):
                results = future.result()
                for result in results:
                    filename, sentence, feature = result
                    writer.writerow([filename, sentence, feature['index'], feature['important'], feature['conjunction'],
                                     feature['length'], feature['words_in_title'], feature['position'], feature['cue_words']])

    print(f"Data untuk bagian '{section_name}' disimpan di {csv_file_path}.")


def split_sentences(text):
    """
    Memecah teks menjadi kalimat-kalimat menggunakan NLTK sent_tokenize.
    """
    sentences = sent_tokenize(text)
    return [sentence.strip() for sentence in sentences if sentence]

@lru_cache(maxsize=10000)  # Simpan hingga 10.000 kata
def cached_stem(word):
    return stemmer.stem(word)

def clean_stopwords_and_stemming(sentence):
    """
    Membersihkan kalimat dari stopwords dan melakukan stemming.
    """
    words = sentence.split()
    cleaned_words = [cached_stem(word) for word in words if word.lower() not in indonesian_stopwords]
    return ' '.join(cleaned_words)


def extract_features(sentences, title_words, cue_words, section_name):
    features = []
    unimportant_words = {"contohnya", "sebagai contoh", "contoh", "misalnya", "misal", "misalkan"}
    conjunction_words = {"dan", "tetapi", "atau", "melainkan", "serta", "karena", "jika", "agar",
                         "meskipun", "walaupun", "sehingga", "supaya", "setelah", "sebelum", "sejak",
                         "ketika", "sebelum", "sesudah", "sejak", "sampai", "sementara", "tatkala", "sewaktu",
                         "oleh karena itu", "dengan demikian", "namun", "akan tetapi", "selain itu", "bahkan",
                         "maupun", "semakin"}
    
    title_word_set = set(title_words)
    for i, sentence in enumerate(sentences):
        original_words = set(word_tokenize(sentence.lower()))
        cleaned_words = set(clean_stopwords_and_stemming(sentence).split())
        
        sentence_features = {
            'index': i,
            'important': 1 if not unimportant_words.intersection(original_words) else 0,
            'conjunction': 0 if conjunction_words.intersection(original_words) else 1,
            'length': len(original_words),
            'words_in_title': len(cleaned_words.intersection(title_word_set)) / (len(cleaned_words.union(title_word_set)) or 1),
            'position': calculate_sentence_position(i, len(sentences)),
            'cue_words': sum(1 for word in cue_words if word in cleaned_words)
        }
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
    
    if i == 0 or i == Mp - 1:
        return 1  # Kalimat awal dan akhir memiliki nilai 1
    elif i == (Mp - 1) // 2:
        return 0.5  # Kalimat tengah memiliki nilai 0.5
    else:
        # Posisi lainnya bisa dihitung sebagai proporsi dari panjang kalimat
        return 0.5 + 0.5 * (abs((2 * i / (Mp - 1)) - 1))
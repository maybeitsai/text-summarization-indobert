import os
import csv
import PyPDF2
from modules.txt2csv import clean_stopwords_and_stemming

def extract_text_from_pdfs(pdf_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        text = ''
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        
        # Save extracted text to a .txt file
        txt_filename = pdf_file.replace('.pdf', '.txt')
        txt_path = os.path.join(output_folder, txt_filename)
        with open(txt_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)
        print(f"Extracted text from {pdf_file} and saved to {txt_filename}")

def categorize_sentences(segmented_texts):
    """
    Fungsi untuk mengkategorikan kalimat dalam dokumen ke dalam 'title', 'latar_belakang', 'tujuan', dan 'metode'.
    """
    categorized_texts = {}

    for file_name, sentences in segmented_texts.items():
        categorized_texts[file_name] = {
            'title': [],
            'latar_belakang': [],
            'tujuan': [],
            'metode': []
        }

        # State flags
        in_latar_belakang = False
        in_tujuan = False
        in_metode = False

        for i, sentence in enumerate(sentences):
            lower_sentence = sentence.lower()

            # 1. Title: always the first sentence
            if i == 0:
                categorized_texts[file_name]['title'].append(sentence)
                continue

            # 2. Latar Belakang
            if "latar belakang" in lower_sentence:
                in_latar_belakang = True

            if in_latar_belakang:
                if any(kw in lower_sentence for kw in ["tujuan penelitian", "tujuan masalah", "batasan dan tujuan"]):
                    in_latar_belakang = False
                    in_tujuan = True
                    categorized_texts[file_name]['tujuan'].append(sentence)
                    continue  # Skip this sentence (boundary marker)

                categorized_texts[file_name]['latar_belakang'].append(sentence)
                continue

            # 3. Tujuan
            if in_tujuan:
                if any(kw in lower_sentence for kw in ["metode penelitian", "metodologi"]):
                    in_tujuan = False
                    in_metode = True
                    categorized_texts[file_name]['metode'].append(sentence)
                    continue  # Skip this sentence (boundary marker)

                categorized_texts[file_name]['tujuan'].append(sentence)
                continue

            # 4. Metode
            if in_metode:
                categorized_texts[file_name]['metode'].append(sentence)
                continue

    return categorized_texts

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

        # Feature 4: Sentence Position (B2i formula)
        Mp = len(sentences)  # Total number of sentences
        ji = i + 1  # Sentence index, starting from 1
        if Mp > 1:
            sentence_features['position'] = (Mp * (ji  - 1)) / Mp
        else:
            sentence_features['position'] = 1  # If there's only one sentence

        # Feature 5: Cue Words (B3i formula)
        cue_count = sum(1 for word in cue_words if word in cleaned_sentence.lower())
        Tfi = sum(1 for sentence in sentences for word in cue_words if word in cleaned_sentence.lower())
        sentence_features['cue_words'] = cue_count / (Tfi if Tfi > 0 else 1)

        features.append(sentence_features)
    
    return features

def categorize_and_extract_features(segmented_texts, title_words_mapping, cue_words):
    """
    Fungsi utama untuk mengkategorikan dan mengekstrak fitur dari kalimat dalam dokumen.
    Menghapus kalimat jika unimportant == 0 atau length == 0.
    """
    # Dictionary to store categorized texts with features
    categorized_texts_with_features = {}

    # Iterate over each document and its sentences
    for file_name, sentences in segmented_texts.items():
        categorized_texts_with_features[file_name] = {
            'title': [],
            'latar_belakang': [],
            'tujuan': [],
            'metode': []
        }

        # Get title words for the current document
        title_words = title_words_mapping.get(file_name, [])

        # Categorize sentences first
        categorized_sentences = categorize_sentences({file_name: sentences})

        # Extract features and associate them with categorized sentences
        for category in ['title', 'latar_belakang', 'tujuan', 'metode']:
            category_sentences = categorized_sentences[file_name][category]
            category_features = extract_features(category_sentences, title_words, cue_words)
            
            # Store sentences along with their features
            categorized_texts_with_features[file_name][category] = [
                {'sentence': sent, 'features': feat}
                for sent, feat in zip(category_sentences, category_features)
                if feat['unimportant'] != 0 and feat['length'] != 0  # Additional check to remove sentences
            ]

    return categorized_texts_with_features

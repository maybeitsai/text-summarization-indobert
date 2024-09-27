import os
import pandas as pd
import nltk
nltk.download('punkt')

def load_and_segment_texts(input_clean_folder):
    segmented_texts = {}
    
    for file_name in os.listdir(input_clean_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_clean_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                # Tokenize text into sentences
                sentences = nltk.sent_tokenize(text)
                segmented_texts[file_name] = sentences
    
    return segmented_texts

# Membuat direktori jika belum ada
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Fungsi untuk memecah teks berdasarkan bagian-bagian
def split_text_into_sections(text, sections):
    sections_content = {}
    current_section = None

    # Memecah teks baris demi baris
    for line in text.splitlines():
        stripped_line = line.strip().lower()  # Menghapus spasi berlebih dan menurunkan huruf besar

        # Menentukan bagian berdasarkan keyword
        if stripped_line in sections:
            current_section = stripped_line
            continue  # Lanjutkan ke baris berikutnya agar nama bagian tidak disimpan dalam konten
        
        # Menambahkan konten ke bagian yang sesuai
        if current_section:
            if current_section not in sections_content:
                sections_content[current_section] = []
            sections_content[current_section].append(line)
    
    # Menggabungkan daftar baris menjadi teks lengkap untuk setiap bagian
    for section in sections_content:
        sections_content[section] = '\n'.join(sections_content[section])
    
    return sections_content

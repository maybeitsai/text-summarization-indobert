import re
import torch
import torch_directml
from transformers import AutoTokenizer, AutoModel
import pdfplumber
import nltk

# Inisialisasi device untuk DirectML
device = torch_directml.device()

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Hapus karakter non-alfabet dan simbol khusus
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Hapus spasi ganda atau lebih
    text = re.sub(r'\s+', ' ', text)

    # Lowercase
    text = text.lower()

    return text.strip()

# Fungsi untuk mengekstrak teks dari PDF
def read_pdf(pdf_path):
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                # Bersihkan teks dari halaman
                cleaned_text = clean_text(page_text)
                extracted_text += cleaned_text + "\n"
    return extracted_text

# Fungsi untuk token embedding, segment embedding, dan position embedding
def encode_text(text, tokenizer, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
    
    input_ids = inputs['input_ids'].to(device)            # Token Embedding
    token_type_ids = inputs['token_type_ids'].to(device)  # Segment Embedding
    attention_mask = inputs['attention_mask'].to(device)  # Attention Mask
    
    return input_ids, token_type_ids, attention_mask

# Fungsi untuk memuat tokenizer dan menambahkan token baru jika perlu
def load_tokenizer(new_tokens=None, model=None):
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
    
    # Jika ada token baru yang ingin ditambahkan
    if new_tokens is not None:
        tokens_to_add = []
        
        # Cek token mana yang belum ada di tokenizer
        for token in new_tokens:
            if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
                tokens_to_add.append(token)
        
        # Jika ada token yang perlu ditambahkan
        if tokens_to_add:
            tokenizer.add_tokens(tokens_to_add)
            if model is not None:
                model.resize_token_embeddings(len(tokenizer))
            print(f"Added tokens: {tokens_to_add}")
        else:
            print("No new tokens were added.")
    
    return tokenizer

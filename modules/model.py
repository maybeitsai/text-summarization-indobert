from modules.variabel import *
from transformers import *

# Fungsi untuk memuat model BERT untuk summarization
def load_model():
    # Memuat model yang sudah dilatih sebelumnya
    model = EncoderDecoderModel.from_pretrained(MODEL).to(DEVICE)
    return model

# Fungsi untuk menghasilkan ringkasan dari teks
def generate_summary(model, tokenizer, text, min_length=MIN_LENGTH, max_length=MAX_LENGTH):
    # Tokenisasi input
    input_ids = tokenizer.encode(text, return_tensors='pt').to(DEVICE)

    # Generasi ringkasan
    summary_ids = model.generate(
        input_ids,
        min_length=min_length,
        max_length=max_length,
        num_beams=10,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=2,
        use_cache=True,
        do_sample=False,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )

    # Dekode ringkasan
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True).to(DEVICE)
    return summary_text

def split_into_chunks(text, tokenizer, max_length=512, overlap=50):
    # Tokenisasi teks
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    chunks = []
    start = 0
    
    # Potong teks menjadi beberapa bagian dengan overlap
    while start < tokens.size(1):
        end = min(start + max_length, tokens.size(1))
        chunks.append(tokens[:, start:end])
        if end == tokens.size(1):
            break
        start += max_length - overlap
    
    return chunks

def generate_full_summary(chunks, model, tokenizer) :
    # Inisialisasi ringkasan
    full_summary = ""

    # Meringkas setiap chunk
    for chunk in chunks:
        summary_ids = model.generate(chunk,
                    min_length=20,
                    max_length=80, 
                    num_beams=10,
                    repetition_penalty=2.5, 
                    length_penalty=1.0, 
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    use_cache=True,
                    do_sample=False,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95)
    
        # Gabungkan setiap ringkasan
        summary_text = tokenizer.decode(summary_ids[0].cpu(), skip_special_tokens=True)
        full_summary += summary_text + " "

    return full_summary


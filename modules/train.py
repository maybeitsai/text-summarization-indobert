from preprocessing import read_pdf, encode_text, load_tokenizer
from model import load_indobert_model, summarize_text
from rouge_score import rouge_scorer

# Fungsi evaluasi menggunakan ROUGE
def evaluate_summary(summary, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

# Fungsi utama pipeline
def summarization_pipeline(file_path):
    # Langkah 1: Membaca dokumen PDF
    text = read_pdf(file_path)
    
    # Langkah 2: Memuat tokenizer dan model
    tokenizer = load_tokenizer()
    model = load_indobert_model()

    # Langkah 3: Melakukan embedding
    input_ids, token_type_ids, attention_mask = encode_text(text, tokenizer)
    inputs = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }
    
    # Langkah 4: Menghasilkan ringkasan
    summarization_output = summarize_text(inputs, model)

    # Konversi tensor ke list token IDs
    summarization_output = summarization_output.tolist()

    # Konversi tensor output model menjadi string
    summary_text = tokenizer.decode(summarization_output[0], skip_special_tokens=True)
    
    # Langkah 5: Evaluasi hasil ringkasan (misal: dengan ringkasan referensi)
    reference_summary = "..."  # Ringkasan yang benar
    rouge_scores = evaluate_summary(summary_text, text)

    return summarization_output, rouge_scores

file_path = 'clean-data/Ike Putri Kusumawijaya (99216004).pdf'  # Sesuaikan dengan file yang diunggah

summary, rouge = summarization_pipeline(file_path)

print("Ringkasan:", summary)
print("ROUGE Scores:", rouge)
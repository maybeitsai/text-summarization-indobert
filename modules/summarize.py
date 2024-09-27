import pandas as pd

def normalize(series):
    """Fungsi untuk normalisasi (Min-Max Scaling) pada kolom pandas."""
    return (series - series.min()) / (series.max() - series.min())

def summarize_section(csv_file_path, output_summary_csv, percentage=0.3, min_sentences=2):
    """
    Fungsi untuk membuat summary berdasarkan CSV yang sudah dibuat.
    Mengambil kalimat dengan bobot (words_in_title + cue_words + position) yang telah distandarisasi,
    dan memilih 30% kalimat teratas lalu mengurutkannya berdasarkan fitur position terkecil.
    
    Parameters:
    - csv_file_path: Path file CSV untuk bagian tertentu, misalnya 'latarbelakang.csv'.
    - output_summary_csv: Path file CSV untuk menyimpan hasil ringkasan.
    - percentage: Persentase kalimat yang diambil berdasarkan bobot terbesar (default 30%).
    
    Returns:
    - Menyimpan hasil ringkasan ke file CSV.
    """
    
    # Baca file CSV
    df = pd.read_csv(csv_file_path)
    
    # List untuk menyimpan hasil summary
    summaries = []
    
    # Group berdasarkan nama dokumen
    grouped = df.groupby('nama_dokumen')
    # Loop setiap group (dokumen)
    for doc_name, group in grouped:
        # Standarisasi kolom words_in_title, cue_words, dan position
        group['std_words_in_title'] = normalize(group['words_in_title'])
        group['std_cue_words'] = normalize(group['cue_words'])
        group['std_position'] = normalize(group['position'])
        
        # Hitung total bobot sebagai penjumlahan dari nilai yang sudah distandarisasi
        group['total_weight'] = group['std_words_in_title'] + group['std_cue_words'] + group['std_position']
        
        # Urutkan berdasarkan total_weight dari yang terbesar
        group_sorted = group.sort_values(by='total_weight', ascending=False)
        
        if len(group) < 7:
            num_sentences = min_sentences
        
        elif len(group) == 1:
            num_sentences = 1

        else:
            # Tentukan jumlah kalimat yang akan diambil (30% dari total kalimat)
            num_sentences = int(len(group) * percentage)
        
        # Ambil 30% kalimat teratas berdasarkan bobot
        top_sentences = group_sorted.head(num_sentences)
        
        # Urutkan kembali top_sentences berdasarkan fitur position dari yang terkecil
        top_sentences_sorted = top_sentences.sort_values(by='index', ascending=True)
        
        # Gabungkan kalimat-kalimat menjadi satu string untuk ringkasan
        summary_text = ' '.join(top_sentences_sorted['kalimat'].tolist())
        
        # Simpan ringkasan dalam list
        summaries.append({'nama_dokumen': doc_name, 'summary': summary_text})
    
    # Simpan hasil summary ke dalam CSV
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_summary_csv, index=False, encoding='utf-8')
    print(f"Ringkasan per dokumen telah disimpan ke {output_summary_csv}")

# Fungsi untuk menggabungkan kalimat menjadi paragraf berdasarkan nama_dokumen
def create_paragraph(df):
    return df.groupby('nama_dokumen')['kalimat'].apply(' '.join).reset_index()
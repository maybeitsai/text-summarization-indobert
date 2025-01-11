# Pengembangan Sistem Peringkasan Teks Secara Ekstraktif Menggunakan Metode BERT Pada Penilaian Proposal SBK Seminar Bidang Kajian dan Ujian Kualifikasi

## Deskripsi Proyek

Proyek ini bertujuan untuk mengembangkan sistem peringkasan teks secara ekstraktif untuk proposal seminar bidang kajian (SBK) dan ujian kualifikasi menggunakan metode BERT. Sistem ini dirancang untuk meningkatkan efisiensi evaluasi proposal dengan menghasilkan ringkasan yang jelas, akurat, dan relevan.

## Fitur Utama

- **Peringkasan Ekstraktif**: Menggunakan metode BERT untuk menganalisis bobot kalimat dan posisi dalam teks.
- **Analisis Kesamaan**: Memanfaatkan Cosine Similarity dan Latent Semantic Analysis (LSA) untuk menganalisis hubungan antar bagian proposal.
- **Evaluasi Otomatis**: Sistem mampu memberikan skor dan rekomendasi kelayakan berdasarkan kriteria yang telah ditentukan.

## Alur Kerja Sistem

1. **Latar Belakang**:
   - Tokenisasi
   - Segmentasi Kalimat
   - Analisis Posisi
   - Peringkasan hingga 30%
2. **Tujuan**:
   - Tokenisasi
   - Segmentasi Kalimat
   - Analisis Posisi
   - Peringkasan hingga 30%
   - Peringkasan ke satu kalimat
3. **Metodologi**:
   - Tokenisasi
   - Segmentasi Kalimat
   - Analisis Posisi
   - Peringkasan hingga 30%
   - Visualisasi melalui gambar
4. **Kontribusi**:
   - Tokenisasi
   - Segmentasi Kalimat
   - Analisis Posisi
   - Peringkasan hingga 30%

## Pendekatan Teknologi

- **Metode Peringkasan**:
  - Kalimat diberi bobot berdasarkan fitur (kata hubung, posisi, panjang, dan lainnya).
  - Kalimat dengan contoh langsung dihilangkan dari ringkasan.
- **Pengolahan Teks**:
  - Segmentasi kalimat dilakukan untuk memecah teks menjadi unit yang lebih kecil.
  - Stemming dan penghilangan stopword diterapkan sebelum mengembalikan teks ke bentuk aslinya.
- **Rekomendasi Peringkasan**:
  - Menggunakan algoritma if-else yang mengintegrasikan LDA atau Cosine Similarity.

## Pencapaian

- **Kinerja Model**:
  - ROUGE Scores:
    - ROUGE-1: 56.63%
    - ROUGE-2: 47.38%
    - ROUGE-L: 51.14%
- **Inovasi Algoritmik**:
  - Pengembangan algoritma untuk pemberian bobot kalimat berdasarkan relevansi dan posisi.
  - Peningkatan efisiensi evaluasi proposal dengan ringkasan yang lebih jelas.

## Cara Menjalankan

1. **Persiapan Lingkungan**:
   - Pastikan Python 3.8 atau lebih tinggi telah terinstal.
   - Instal dependensi dengan perintah berikut:
     ```bash
     pip install -r requirements.txt
     ```

2. **Menjalankan Streamlit**:
   - Jalankan antarmuka pengguna dengan perintah:
     ```bash
     streamlit run result.py
     ```

3. **Akses Sistem**:
   - Buka browser dan akses sistem melalui URL yang ditampilkan.

## Catatan Tambahan

- Sistem ini dirancang untuk menyederhanakan proses penilaian proposal akademik dan memberikan rekomendasi berbasis data.
- Pengembangan lebih lanjut mencakup peningkatan akurasi ringkasan menggunakan model GPT-2 atau metode hybrid.
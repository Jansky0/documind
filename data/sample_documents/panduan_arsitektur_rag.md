# Panduan Lengkap Arsitektur Retrieval-Augmented Generation (RAG)

## 1. Pengenalan RAG
Retrieval-Augmented Generation (RAG) adalah teknik AI modern yang menggabungkan kemampuan model bahasa besar (Large Language Model / LLM) dengan sistem pencarian informasi eksternal (Information Retrieval). Dengan RAG, LLM tidak hanya mengandalkan data latihannya, tetapi dapat mengakses basis pengetahuan spesifik secara real-time.

### Keunggulan Utama RAG:
1. **Mengurangi Halusinasi**: LLM menjawab berbasis fakta spesifik dari dokumen referensi yang disediakan.
2. **Kerahasiaan Data**: Dokumen privat perusahaan atau pengguna tidak perlu dikirimkan untuk melatih ulang (fine-tuning) model.
3. **Efisiensi Biaya**: Jauh lebih murah dan fleksibel dibanding fine-tuning LLM secara berkala.

## 2. Komponen Inti Pipeline RAG
Arsitektur RAG pada umumnya terdiri atas tahapan berikut:

1. **Document Ingestion & Chunking**: Dokumen dipecah menjadi potongan-potongan kecil (chunks) dengan ukuran tertentu (misal 500-1000 karakter) disertai overlap untuk menjaga keutuhan konteks antar kalimat.
2. **Embedding Generation**: Teks pada setiap chunk dikonversi menjadi representasi vektor numerik berdimensi tinggi menggunakan embedding model (misal `text-embedding-004`).
3. **Vector Database**: Vektor disimpan dalam database seperti ChromaDB dengan indeks pencarian kedekatan semantik (cosine similarity).
4. **Semantic Retrieval**: Saat pengguna bertanya, pertanyaan dikonversi menjadi vektor dan dicarikan potongan dokumen (chunks) yang paling relevan (Top-K).
5. **Prompt Augmentation & Generation**: Potongan dokumen yang relevan disisipkan ke dalam prompt instruksi, kemudian dikirimkan ke LLM (seperti Google Gemini) untuk menghasilkan jawaban yang disertai kutipan sumber.

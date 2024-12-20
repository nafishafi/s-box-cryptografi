![S-Box Analyzer GUI](assets/gui.png)

# Alat Analisis S-Box

Proyek ini menyediakan serangkaian alat analisis kriptografi untuk S-Box (Substitution Box). Alat-alat ini menghitung berbagai properti kriptografi dari S-Box, termasuk namun tidak terbatas pada:

- Strict Avalanche Criterion (SAC)
- Linear Approximation Probability (LAP)
- Nonlinearity
- Differential Uniformity
- Differential Approximation Probability (DAP)
- Entropy
- Bit Independence Criterion (BIC)

## Fitur

1. **Analisis S-Box**: Alat untuk menganalisis kekuatan dan properti kriptografi dari sebuah S-Box.
2. **Antarmuka Web Streamlit**: Antarmuka berbasis web untuk mengunggah, memproses, dan memvisualisasikan S-Box.

## Persyaratan

Pastikan Anda sudah menginstal hal-hal berikut:

- Python 3.x
- pip

Instal dependensi Python yang diperlukan:
```bash
pip install streamlit
pip install numpy
pip install pandas
pip install xlsxwriter
```

Untuk menjalankan aplikasi Streamlit
```bash
streamlit run main.py
```

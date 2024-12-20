# Sebelum run install dahulu:
# pip install streamlit
# pip install xlsxwriter

# Panduan run
# Open lokasi file di direktory kemudian klik kanan dimana saja (bukan di file nya) lalu open in terminal
# lalu jalan kan prompt 
# streamlit run gui.py


import streamlit as st
import pandas as pd
import numpy as np
import math
from io import BytesIO

# Fungsi membaca S-Box dari file Excel
def read_sbox(file):
    df = pd.read_excel(file, header=None)
    return df.values.flatten().astype(int)

# Fungsi untuk Bit Independence Criterion (BIC) - Nonlinearity
def calculate_bic(sbox):
    frequency = [0] * 256
    total_elements = len(sbox)

    for value in sbox:
        frequency[value] += 1

    probability = [freq / total_elements for freq in frequency]
    bic_score = -sum(p * (0 if p == 0 else math.log2(p)) for p in probability)
    return bic_score

# Fungsi untuk Bit Independence Criterion (BIC) - Nonlinearity
def walsh_transform(t):
    """Melakukan transformasi Walsh pada urutan biner t."""
    n = log2n(len(t))  # n tidak digunakan, tetapi memastikan jika n bukan pangkat 2
    wt = len(t) * [0]
    for w in range(len(t)):
        for x in range(len(t)):
            wt[w] = wt[w] + (-1) ** (t[x] ^ binary_inner_product(w, x))
    return wt

def binary_inner_product(a, b):
    """Produk dalam biner antara a dan b."""
    ip = 0
    ab = a & b
    while ab > 0:
        ip = ip ^ (ab & 1)  # baik ^ atau + dapat digunakan untuk transformasi Walsh
        ab = ab >> 1
    return ip

def non_linearity(t):
    """Non-linearity dari urutan biner t."""
    wt = walsh_transform(t)
    nl = len(t) / 2 - 0.5 * max([abs(i) for i in wt])
    return nl

def log2n(l):
    """Log2 dari sebuah integer hanya untuk angka yang merupakan pangkat dari 2."""
    x = l
    n = 0
    while x > 0:
        x = x >> 1
        n = n + 1
    n = n - 1
    assert 2**n == l, "log2n(l) hanya valid untuk l = 2**n"
    return n

# Fungsi untuk menghitung ukuran Non-Linearity dari S-Box
def calculate_nonlinearity(sbox):
    """Menghitung ukuran nonlinearity dari s-box menggunakan transformasi Walsh."""
    n = log2n(len(sbox))  # Panjang s-box harus merupakan pangkat dari 2
    nlv = (2**n - 1) * [0]  # Vektor untuk menyimpan hasil perhitungan nonlinearity untuk setiap hasil 255

    for c in range(len(nlv)):  # Untuk setiap cara menggabungkan 8 bit
        t = [binary_inner_product(c + 1, sbox[i]) for i in range(len(sbox))]
        nlv[c] = non_linearity(t)

    # Menghitung nilai nonlinearity
    nonlinearity_score = sum(abs(i) for i in nlv) / len(nlv)
    
    # Kembalikan nilai nonlinearity_score
    return nonlinearity_score

def calculate_bic_nl(sbox):
    bic_score = calculate_bic(sbox)
    nonlinearity_score = calculate_nonlinearity(sbox)
    bic_nl_score = abs(bic_score + nonlinearity_score) - 8
    return bic_nl_score

# Strict Avalanche Criterion (SAC)
def strict_avalanche_criterion(sbox):
    bit_changes = 0
    total_bits = 0

    for input1 in range(256):
        output1 = sbox[input1]
        for i in range(8):
            input2 = input1 ^ (1 << i)
            output2 = sbox[input2]
            bit_changes += bin(output1 ^ output2).count('1')
            total_bits += 8

    return bit_changes / total_bits

# Linear Approximation Probability (LAP)
def calculate_lap(sbox):
    n = len(sbox)
    max_lap = 0

    for a in range(n):
        for b in range(n):
            if a == 0 and b == 0:
                continue
            count = 0
            for x in range(n):
                if bin((x & a)).count('1') % 2 == bin((sbox[x] & b)).count('1') % 2:
                    count += 1
            probability = count / n
            lap = abs(probability - 0.5)
            max_lap = max(max_lap, lap)

    return max_lap


# Differential Approximation Probability (DAP)
def calculate_dap(sbox):
    n = len(sbox)
    dap = 0
    for a in range(1, n):
        for b in range(1, n):
            count = 0
            for x in range(n):
                if (sbox[x] ^ sbox[x ^ a]) == b:
                    count += 1
            dap = max(dap, count / n)
    return dap

# Bit Independence Criterion-Strict Avalanche Criterion (BIC-SAC)
def bit_independence_criterion_adjusted(sbox):
    num_bits = 8
    bic_total = 0
    num_elements = len(sbox)

    for i in range(num_elements):
        for bit1 in range(num_bits):
            for bit2 in range(num_bits):
                if bit1 == bit2:
                    continue
                mask1 = 1 << bit1
                mask2 = 1 << bit2
                bic_total += (
                    (((sbox[i] & mask1) >> bit1) ^ ((sbox[i] & mask2) >> bit2)) * 1.0037
                )

    return bic_total / (num_elements * num_bits * (num_bits - 1.03))

def calculate_bic_sac(sbox):
    sac_value = strict_avalanche_criterion(sbox)
    bic_value = bit_independence_criterion_adjusted(sbox)
    bic_sac_score = (sac_value + bic_value) / 2
    return bic_sac_score

def save_to_excel(data):
    output = BytesIO()
    df = pd.DataFrame(
        [(method, result) for method, result in data.items()],
        columns=["Method", "Result"]
    )
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    output.seek(0)
    return output

st.title("S-Box Analysis Tool")

# Upload file S-Box
uploaded_file = st.file_uploader("Upload S-Box File (Excel Format)", type="xlsx")

if uploaded_file:
    sbox = read_sbox(uploaded_file)
    st.write("### Imported S-Box")
    st.dataframe(sbox.reshape(16, 16))

    # Pilihan operasi
    operations = st.multiselect(
        "Choose operations to perform:",
        ["Non-Linearity (NL)", "Strict Avalanche Criterion (SAC)", 
         "Linear Approximation Probability (LAP)", 
         "Differential Approximation Probability (DAP)", 
         "Bit Independence Criterion-Nonlinearity (BIC-NL)", 
         "Bit Independence Criterion-Strict Avalanche Criterion (BIC-SAC)"]
    )

    # Tombol proses
    if st.button("Run Analysis"):
        if not operations:
            st.warning("No operations selected. Please choose at least one operation to perform.")
        else:
            results = {}

            if "Non-Linearity (NL)" in operations:
                nl_result = calculate_nonlinearity(sbox)
                results["Non-Linearity"] = nl_result
                st.write(f"**Non-Linearity (NL):** {nl_result}")

            if "Strict Avalanche Criterion (SAC)" in operations:
                sac_result = strict_avalanche_criterion(sbox)
                results["Strict Avalanche Criterion"] = sac_result
                st.write(f"**Strict Avalanche Criterion (SAC):** {sac_result:.5f}")

            if "Linear Approximation Probability (LAP)" in operations:
                lap_result = calculate_lap(sbox)
                results["Linear Approximation Probability"] = lap_result
                st.write(f"**Linear Approximation Probability (LAP):** {lap_result:.5f}")

            if "Differential Approximation Probability (DAP)" in operations:
                dap_result = calculate_dap(sbox)
                results["Differential Approximation Probability"] = dap_result
                st.write(f"**Differential Approximation Probability (DAP):** {dap_result:.6f}")

            if "Bit Independence Criterion-Nonlinearity (BIC-NL)" in operations:
                bic_nl_result = calculate_bic_nl(sbox)
                results["BIC-NL"] = bic_nl_result
                st.write(f"**Bit Independence Criterion-Nonlinearity (BIC-NL):** {bic_nl_result}")

            if "Bit Independence Criterion-Strict Avalanche Criterion (BIC-SAC)" in operations:
                bic_sac_result = calculate_bic_sac(sbox)
                results["BIC-SAC"] = bic_sac_result
                st.write(f"**Bit Independence Criterion-Strict Avalanche Criterion (BIC-SAC):** {bic_sac_result:.5f}")

            # Unduh hasil
            st.download_button(
                label="Download Results",
                data=save_to_excel(results),
                file_name="sbox_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

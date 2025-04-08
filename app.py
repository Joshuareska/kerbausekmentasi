import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Aplikasi Streamlit Sederhana")

# Sidebar
st.sidebar.title("Navigasi")
menu = st.sidebar.selectbox("Pilih Menu", ["Home", "Data Input", "Visualisasi"])

# Home
if menu == "Home":
    st.header("Selamat Datang di Aplikasi Streamlit")
    st.write("""
    Aplikasi ini dibuat sebagai contoh sederhana untuk memahami dasar-dasar penggunaan Streamlit.
    Anda dapat memasukkan data, melihat tabel, atau visualisasi sederhana.
    """)

# Data Input
elif menu == "Data Input":
    st.header("Input Data")

    # Input Form
    with st.form("input_form"):
        nama = st.text_input("Nama")
        usia = st.number_input("Usia", min_value=1, max_value=100, step=1)
        pekerjaan = st.selectbox("Pekerjaan", ["Pelajar", "Mahasiswa", "Pekerja", "Pengusaha", "Lainnya"])
        submit = st.form_submit_button("Submit")

    if submit:
        st.success(f"Data berhasil dimasukkan: {nama}, {usia} tahun, {pekerjaan}")

# Visualisasi
elif menu == "Visualisasi":
    st.header("Visualisasi Data")

    # Dummy Data
    data = {
        "Bulan": ["Jan", "Feb", "Mar", "Apr", "Mei"],
        "Penjualan": [100, 200, 300, 400, 500]
    }
    df = pd.DataFrame(data)

    # Display Dataframe
    st.subheader("Tabel Data")
    st.dataframe(df)

    # Plot
    st.subheader("Grafik Penjualan")
    fig, ax = plt.subplots()
    ax.plot(df["Bulan"], df["Penjualan"], marker="o")
    ax.set_title("Penjualan per Bulan")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Penjualan")
    st.pyplot(fig)
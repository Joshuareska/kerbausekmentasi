import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_and_classify_resized_grayscale(image_path, output_excel_path):
    # Load YOLOv8 model untuk segmentasi
    model = YOLO('../yolov8x-seg.pt')  # Gunakan model YOLO yang sudah dilatih untuk mendeteksi sapi

    # Baca gambar
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Deteksi dan segmentasi objek sapi
    results = model.predict(image, conf=0.5)
    result = results[0]  # Mengambil hasil pertama jika ada lebih dari satu objek yang terdeteksi

    # Pastikan objek sapi ada yang terdeteksi
    if result.masks is not None:
        # Dapatkan mask dari objek yang terdeteksi
        mask = result.masks.data[0].cpu().numpy()  # Ambil mask untuk objek sapi pertama
        mask = (mask * 255).astype(np.uint8)  # Konversi mask ke format uint8

        # Ubah ukuran mask jika perlu agar sama dengan ukuran gambar
        if mask.shape != image_rgb.shape[:2]:  # Memastikan ukuran mask sama dengan gambar
            mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))

        # Terapkan mask pada gambar untuk area sapi
        masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

        # Konversi gambar hasil masking menjadi grayscale
        grayscale_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)

        # Resize gambar ke 5x5
        resized_grayscale = cv2.resize(grayscale_image, (20, 20), interpolation=cv2.INTER_NEAREST)

        # Eksklusi nilai 0 (background)
        valid_pixels = resized_grayscale[resized_grayscale > 0]

        # Klasifikasi piksel hitam dan putih dari piksel valid
        black_pixels = (valid_pixels <= 95).sum()
        white_pixels = (valid_pixels > 95).sum()
        total_pixels = valid_pixels.size
        black_percentage = (black_pixels / total_pixels) * 100
        white_percentage = (white_pixels / total_pixels) * 100

        # Simpan matriks grayscale ke DataFrame
        df_resized = pd.DataFrame(resized_grayscale)

        # Simpan DataFrame ke file Excel
        df_resized.to_excel(output_excel_path, index=False, header=False)

        # Visualisasi gambar asli, grayscale, dan hasil klasifikasi
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Gambar asli
        ax[0].imshow(image_rgb)
        ax[0].set_title("Gambar Asli")
        ax[0].axis("off")

        # Gambar hasil grayscale dan resize
        ax[1].imshow(resized_grayscale, cmap='gray')
        # ax[1].set_title(f"Grayscale\nHitam: {black_percentage:.2f}%, Putih: {white_percentage:.2f}%")
        ax[1].axis("off")

        # Tampilkan plot
        plt.suptitle("Visualisasi dan Klasifikasi Grayscale", fontsize=16)
        plt.tight_layout()
        plt.show()

        # Informasi data yang disimpan
        print(f"Data matriks grayscale telah disimpan ke {output_excel_path}")
        # print(f"Persentase Hitam: {black_percentage:.2f}%, Putih: {white_percentage:.2f}%")
        print(f"Matriks Grayscale:\n{resized_grayscale}")
    else:
        print("Tidak ada objek sapi yang terdeteksi.")

# Ganti 'path/to/image.jpg' dengan path gambar Anda dan 'output_excel_path' dengan lokasi file Excel yang akan disimpan
visualize_and_classify_resized_grayscale('dataset/data (1).jpg', 'output/output_grayscale.xlsx')

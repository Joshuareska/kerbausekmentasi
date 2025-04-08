import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fungsi untuk membaca gambar dan meresize ke 5x5 piksel
def resize_image_to_5x5(image_path):
    # Baca gambar asli
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Membaca gambar dalam skala abu-abu
    if image is None:
        print(f"Error: Tidak dapat membaca gambar {image_path}")
        return None, None
    # Resize gambar ke 5x5 piksel
    resized_image = cv2.resize(image, (5, 5), interpolation=cv2.INTER_AREA)
    return image, resized_image


# Path ke gambar
path_gambar = "dataset/salekoa.jpg"  # Ganti dengan path ke gambar Anda

# Resize gambar ke 5x5
original_image, resized_matrix_5x5 = resize_image_to_5x5(path_gambar)

# Tampilkan gambar asli dan gambar yang diresize menjadi 5x5
if original_image is not None and resized_matrix_5x5 is not None:
    # Buat plot untuk menampilkan gambar asli dan hasil resize
    plt.figure(figsize=(10, 5))

    # Tampilkan matriks 5x5 di konsol
    print("Matriks 5x5 dari gambar yang diresize:")
    print(resized_matrix_5x5)

    # Gambar asli
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Gambar Asli")
    plt.axis('off')

    # Gambar yang diresize ke 5x5
    plt.subplot(1, 2, 2)
    plt.imshow(resized_matrix_5x5, cmap='gray')
    plt.title("Gambar yang diresize (5x5)")
    plt.axis('off')

    plt.show()

    # Simpan matriks 5x5 ke CSV
    df = pd.DataFrame(resized_matrix_5x5)
    df.to_csv("resized_matrix_5x5.csv", index=False, header=False)
    print("Matriks 5x5 disimpan ke resized_matrix_5x5.csv")
else:
    print("Gagal memuat atau meresize gambar.")
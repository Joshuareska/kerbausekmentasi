import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Path ke gambar yang diupload
uploaded_image_path = "dataset/lotongbokoo.png"

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

# Resize gambar ke 5x5
original_image, resized_matrix_5x5 = resize_image_to_5x5(uploaded_image_path)

# Tampilkan matriks 5x5 di konsol
print("Matriks 5x5 dari gambar yang diresize:")
print(resized_matrix_5x5)

# Proses jika gambar berhasil dibaca
if original_image is not None and resized_matrix_5x5 is not None:
    # Hitung histogram warna hitam dan putih
    total_pixels = resized_matrix_5x5.size  # Total piksel dalam matriks
    white_pixels = np.sum(resized_matrix_5x5 > 127)  # Piksel putih > 127
    black_pixels = total_pixels - white_pixels  # Sisanya adalah piksel hitam

    # Persentase
    white_percentage = (white_pixels / total_pixels) * 100
    black_percentage = (black_pixels / total_pixels) * 100

    # Buat histogram persentase warna
    categories = ['Hitam', 'Putih']
    percentages = [black_percentage, white_percentage]

    # Plot hasil
    plt.figure(figsize=(12, 6))

    # Plot gambar asli
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Gambar Asli")
    plt.axis('off')

    # Plot gambar yang diresize
    plt.subplot(1, 3, 2)
    plt.imshow(resized_matrix_5x5, cmap='gray')
    plt.title("Gambar Resize (5x5)")
    plt.axis('off')

    # Plot histogram
    plt.subplot(1, 3, 3)
    plt.bar(categories, percentages, color=['black', 'white'], edgecolor='gray')
    plt.title("Intensitas Hitam & Putih")
    plt.ylabel("Persentase (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Gambar yang diresize ke 5x5
    plt.subplot(1, 2, 2)
    plt.imshow(resized_matrix_5x5, cmap='gray')
    plt.title("Gambar yang diresize (5x5)")
    plt.axis('off')


    # Simpan matriks 5x5 ke CSV
    df = pd.DataFrame(resized_matrix_5x5)
    # csv_path = "/mnt/data/resized_matrix_5x5.csv"
    csv_path = r"C:\Users\Joshua\PycharmProjects\kerbaruestimas\data\output.csv"
    df.to_csv(csv_path, index=False, header=False)
    print(f"Matriks 5x5 disimpan ke {csv_path}")

else:
    print("Gagal memuat atau meresize gambar.")
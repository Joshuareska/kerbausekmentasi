from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_segmented_pixels(image_path):
    # Load YOLOv8 model untuk segmentasi
    model = YOLO('yolov8x-seg.pt')  # Gunakan model YOLO yang sudah dilatih untuk mendeteksi sapi

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

        # Ubah gambar yang tersegmentasi menjadi grayscale dan resize
        grayscale = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(grayscale, (20, 20), interpolation=cv2.INTER_NEAREST)

        # Hilangkan piksel dengan nilai 0
        valid_pixels = resized[resized > 0]

        # Hitung persentase hitam dan putih (berdasarkan nilai grayscale)
        black_percentage = (valid_pixels <= 80).sum() / valid_pixels.size * 100
        white_percentage = 100 - black_percentage

        # Visualisasi
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Tampilkan gambar asli
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Gambar Asli")
        ax[0].axis("off")

        # Tampilkan gambar resize hasil segmentasi
        ax[1].imshow(resized, cmap='gray')
        ax[1].set_title("Gambar Resize (5x5)")
        ax[1].axis("off")

        # Tampilkan grafik persentase hitam dan putih
        ax[2].bar(["Hitam", "Putih"], [black_percentage, white_percentage], color=['black', 'white'], edgecolor='black')
        ax[2].set_ylim(0, 100)
        ax[2].set_title("Intensitas Hitam & Putih")
        ax[2].set_ylabel("Persentase (%)")

        # Tampilkan judul utama
        plt.suptitle("Segmentasi dan Estimasi Piksel", fontsize=16)

        # Tampilkan plot
        plt.tight_layout()
        plt.show()

        # Cetak matriks piksel (termasuk nilai nol untuk referensi)
        print("Matriks 5x5 dari gambar yang diresize:")
        print(resized)

        # Cetak matriks piksel yang valid (tanpa nilai nol)
        print("\nPiksel valid (tanpa nilai nol):")
        print(valid_pixels)

    else:
        print("Tidak ada objek sapi yang terdeteksi.")

# Ganti 'path/to/image.jpg' dengan path gambar Anda
visualize_segmented_pixels('kerbautes/data (12).jpg')

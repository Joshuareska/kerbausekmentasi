
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin

# Step 1: Load model YOLO untuk deteksi jenis sapi
model_detection = YOLO('best.pt')  # Model untuk deteksi jenis sapi

# Step 2: Load model YOLO untuk segmentasi
model_segmentation = YOLO('yolov8x-seg.pt')  # Model untuk segmentasi

# Step 3: Mapping nama kelas untuk deteksi jenis sapi
class_names = {
    0: "Bonga (Price: 150 million)",
    1: "Lotongboko (Price: 350 million)",
    2: "Saleko",
    3: "Other Types"
}

# Step 4: Centroids Tetap
centroids = np.array([
    [80, 80, 80],  # Centroid warna putih
    [65, 65, 65]   # Centroid warna hitam
])
print("Centroids yang digunakan:\n", centroids)

# Step 5: Baca gambar dan ubah ukurannya menjadi 640x640
image_path = 'data/gambar (9).jpeg'  # Ganti dengan path gambar Anda
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (640, 640))  # Resize ke 640x640

# Step 6: Lakukan prediksi dengan YOLO untuk deteksi jenis sapi
results_detection = model_detection.predict(source=resized_image, conf=0.80)  # Deteksi jenis sapi

# Step 7: Visualisasi hasil dan proses khusus untuk Saleko
output_image = resized_image.copy()
detected = False  # Flag untuk mengecek apakah ada kelas yang terdeteksi

for result in results_detection:
    if hasattr(result, 'boxes') and result.boxes is not None:
        for box in result.boxes:
            detected = True  # Set flag jika ada kotak yang terdeteksi
            class_id = int(box.cls)  # ID kelas
            original_class_name = class_names[class_id]  # Nama kelas asli
            confidence = float(box.conf)  # Konversi confidence menjadi float

            # Koordinat kotak pembatas
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Konversi ke integer
            roi = output_image[y1:y2, x1:x2]  # Region of interest

            # Proses khusus untuk Saleko
            if class_id == 2:  # Saleko
                # Lakukan segmentasi dengan YOLO untuk ROI
                print("Deteksi kelas Saleko. Menjalankan segmentasi...")
                results_segmentation = model_segmentation.predict(source=roi, conf=0.50, task='segment')

                for seg_result in results_segmentation:
                    if seg_result.masks is not None:
                        # Gabungkan semua mask
                        masks = seg_result.masks.data.cpu().numpy()
                        combined_mask = np.sum(masks, axis=0) > 0

                        # Resize mask agar sesuai dengan ROI asli
                        combined_mask_resized = cv2.resize(
                            combined_mask.astype(np.uint8),
                            (roi.shape[1], roi.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )

                        # Ambil piksel dari ROI berdasarkan mask
                        masked_roi = roi[combined_mask_resized.astype(bool)]

                        # Hitung persentase warna putih menggunakan centroids tetap
                        print("Menggunakan centroids tetap untuk klasifikasi warna...")
                        roi_reshaped = masked_roi.reshape((-1, 3)).astype(float)  # Flatten ke (n_pixels, 3)
                        labels = pairwise_distances_argmin(roi_reshaped, centroids)  # Tentukan cluster
                        white_cluster = np.argmax(np.mean(centroids, axis=1))  # Cluster dengan mean RGB tertinggi
                        white_percentage = np.sum(labels == white_cluster) / len(labels) * 100

                        # Tentukan harga berdasarkan persentase warna putih
                        if white_percentage > 70:
                            price_estimation = "500 Million"
                        else:
                            price_estimation = "400 Million"
                        original_class_name += f" (Price: {price_estimation})"

                        # Informasi tambahan di konsol
                        print(f"Persentase warna putih untuk Saleko: {white_percentage:.2f}%")
                        print(f"Harga ditentukan: {price_estimation}")

            else:
                # Harga default untuk kelas selain Saleko
                price_estimation = original_class_name.split("(")[-1].replace(")", "")

            # Gambar kotak pembatas dengan warna biru
            color = (255, 0, 0)  # Biru
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

            # Tambahkan label kelas dengan font yang lebih besar
            label = f"{original_class_name} ({confidence:.2f})"
            cv2.putText(output_image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # Informasi di konsol
            print(f"Kelas: {original_class_name}, Confidence: {confidence:.2f}")

# Jika tidak ada objek terdeteksi, tambahkan teks "Other Types"
if not detected:
    text = "Other Types"
    font_scale = 1.5
    font_color = (255, 0, 0)  # Warna biru
    thickness = 3
    x, y = 50, 50  # Lokasi teks pada gambar
    cv2.putText(output_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

# Tampilkan hasil
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
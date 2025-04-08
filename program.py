import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from PIL import Image

# Judul aplikasi dan deskripsi
st.title("Prediksi Jenis dan Harga Kerbau")
st.write("Unggah gambar kerbau untuk mendeteksi jenis dan menentukan harga berdasarkan warna dan corak")

# Memuat model YOLO
model_detection = YOLO('best.pt')  # Model YOLO untuk deteksi jenis kerbau
model_segmentation = YOLO('yolov8x-seg.pt')  # Model YOLO untuk segmentasi

# Centroids tetap untuk klasifikasi warna (putih dan hitam)
centroids = np.array([
    [80, 80, 80],  # Centroid untuk warna putih
    [65, 65, 65]   # Centroid untuk warna hitam
])

# Mapping nama kelas kerbau
class_names = {
    0: "Bonga (Price: 150 juta)",
    1: "Lotongboko (Price: 350 juta)",
    2: "Saleko",
    3: "Jenis Lain"
}

# Fungsi untuk prediksi dan visualisasi hasil
def predict_and_visualize(image_path):
    # Membaca gambar dan meresize ke ukuran 640x640
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (640, 640))

    # Melakukan deteksi jenis kerbau dengan YOLO
    results_detection = model_detection.predict(source=resized_image, conf=0.70)
    output_image = resized_image.copy()
    detected = False  # Flag untuk mengecek apakah ada objek yang terdeteksi

    # Loop melalui hasil deteksi
    for result in results_detection:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                detected = True
                class_id = int(box.cls)  # ID kelas dari hasil deteksi
                original_class_name = class_names[class_id]  # Nama kelas berdasarkan ID
                confidence = float(box.conf)  # Confidence score

                # Mendapatkan koordinat kotak pembatas (bounding box)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = output_image[y1:y2, x1:x2]  # Region of Interest (ROI)

                # Jika kelas adalah "Saleko", proses segmentasi dan klasifikasi warna
                if class_id == 2:  # Saleko
                    results_segmentation = model_segmentation.predict(source=roi, conf=0.50, task='segment')
                    for seg_result in results_segmentation:
                        if seg_result.masks is not None:
                            # Menggabungkan semua mask
                            masks = seg_result.masks.data.cpu().numpy()
                            combined_mask = np.sum(masks, axis=0) > 0

                            # Meresize mask agar sesuai dengan ROI asli
                            combined_mask_resized = cv2.resize(
                                combined_mask.astype(np.uint8),
                                (roi.shape[1], roi.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                            # Ambil piksel dari ROI berdasarkan mask
                            masked_roi = roi[combined_mask_resized.astype(bool)]

                            # Klasifikasi warna menggunakan centroids tetap
                            roi_reshaped = masked_roi.reshape((-1, 3)).astype(float)
                            labels = pairwise_distances_argmin(roi_reshaped, centroids)
                            white_cluster = np.argmax(np.mean(centroids, axis=1))  # Cluster dengan mean RGB tertinggi
                            white_percentage = np.sum(labels == white_cluster) / len(labels) * 100

                            # Menentukan harga berdasarkan persentase warna putih
                            if white_percentage > 90:
                                price_estimation = "500 Million"
                            else:
                                price_estimation = "400 Million"
                            original_class_name += f" (Price: {price_estimation})"


                            # Menampilkan informasi tambahan di Streamlit
                            st.write(f"White color percentage: {white_percentage:.2f}%")
                            st.write(f"Price specified: {price_estimation}")

                # Menambahkan kotak pembatas (bounding box) ke gambar
                color = (255, 0, 0)  # Warna biru untuk kotak pembatas
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

                # Menambahkan label kelas di atas bounding box
                label = f"{original_class_name} ({confidence:.2f})"
                cv2.putText(output_image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                # Menampilkan informasi kelas dan confidence di Streamlit
                st.write(f"Class: {original_class_name}, Confidence: {confidence:.2f}")

    # Jika tidak ada objek yang terdeteksi
    if not detected:
        st.write("Other types. (Price not available).")
    return output_image

# File uploader untuk mengunggah gambar
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Tombol untuk memulai proses prediksi
if uploaded_file is not None:
    # Simpan gambar yang diunggah ke folder sementara
    image_path = f"temp_{uploaded_file.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Tampilkan gambar yang diunggah
    st.image(Image.open(image_path), caption="View Image", use_column_width=True)

    # Tombol untuk memulai proses
    if st.button("Process"):
        st.write("Processing images...")
        output_image = predict_and_visualize(image_path)

        # Tampilkan hasil prediksi
        st.image(output_image[:, :, ::-1], caption="Prediction Results", use_column_width=True)
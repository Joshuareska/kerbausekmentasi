import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Judul aplikasi
st.title("Prediksi Jenis dan Harga Kerbau")
st.write("Unggah gambar kerbau untuk mendeteksi Harga")

# Memuat model YOLO
model_detection = YOLO('model/best.pt')  # Model YOLO untuk deteksi jenis kerbau
model_segmentation = YOLO('model/yolov8x-seg.pt')  # Model YOLO untuk segmentasi

# Mapping nama kelas kerbau
class_names = {
    0: "Bonga",
    1: "Lotongboko",
    2: "Saleko",
    3: "Other Types"
}

# Mapping harga berdasarkan kelas
price_mapping = {
    0: "150 Million - 200 Million",
    1: "350 Million - 400 Million",
    2: None,  # Harga Saleko dihitung melalui proses
    3: "Price Not Available"
}

# Fungsi untuk memproses kerbau Saleko
def process_saleko(masked_image):
    # Konversi hasil masking menjadi grayscale
    grayscale_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Resize gambar ke 20x20
    resized_grayscale = cv2.resize(grayscale_image, (20, 20), interpolation=cv2.INTER_NEAREST)

    # Hitung piksel valid (tanpa 0)
    valid_pixels = resized_grayscale[resized_grayscale > 0]

    # Klasifikasi piksel hitam dan putih dari valid pixels
    black_pixels = (valid_pixels <= 80).sum()
    white_pixels = (valid_pixels > 80).sum()
    total_pixels = valid_pixels.size
    black_percentage = (black_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    white_percentage = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    # Menentukan harga berdasarkan persentase warna putih
    if white_percentage > 90:
        price_estimation = "(Harga Rp. 500 Juta atau diatasnya)" #lebih dari sama dengan
    else:
        price_estimation = "(Harga: Rp. 350 Juta - Rp. 500 Juta)"

    # Tampilkan hasil klasifikasi dan visualisasi
    st.write("### Persentase Berdasarkan Matriks Klasifikasi")
    st.write(f"White Percentage: {white_percentage:.2f}%")
    st.write(f"Black Percentage: {black_percentage:.2f}%")
    st.write(f"Price: {price_estimation}")

    # Visualisasi gambar asli, grayscale, dan hasil klasifikasi
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Gambar hasil grayscale dan resize
    ax.imshow(resized_grayscale, cmap='gray')
    ax.set_title(
        f"Grayscale\nHitam: {black_percentage:.2f}%, Putih: {white_percentage:.2f}%"
    )
    ax.axis("off")

    plt.suptitle("Visualisasi Grayscale dan Klasifikasi", fontsize=16)
    st.pyplot(fig)

    # Tampilkan matriks grayscale
    st.write("### Matriks Grayscale")
    st.dataframe(pd.DataFrame(resized_grayscale))

    return price_estimation

# Fungsi utama untuk prediksi dan visualisasi
def predict_and_visualize(image_path):
    # Membaca gambar
    image = cv2.imread(image_path)
    if image is None:
        st.error("Gambar tidak ditemukan.")
        return

    resized_image = cv2.resize(image, (640, 640))

    # Deteksi jenis kerbau
    results_detection = model_detection.predict(source=resized_image, conf=0.70)
    output_image = resized_image.copy()
    detected = False

    # Loop melalui hasil deteksi
    for result in results_detection:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                detected = True
                class_id = int(box.cls)
                class_name = class_names[class_id]
                price_estimation = price_mapping[class_id]
                confidence = float(box.conf)

                # Mendapatkan ROI
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = output_image[y1:y2, x1:x2]

                # Proses segmentasi untuk Saleko
                if class_id == 2:  # Saleko
                    results_segmentation = model_segmentation.predict(source=roi, conf=0.50, task='segment')
                    for seg_result in results_segmentation:
                        if seg_result.masks is not None:
                            masks = seg_result.masks.data.cpu().numpy()
                            combined_mask = np.sum(masks, axis=0) > 0
                            combined_mask_resized = cv2.resize(
                                combined_mask.astype(np.uint8),
                                (roi.shape[1], roi.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                            masked_roi = cv2.bitwise_and(roi, roi, mask=combined_mask_resized)

                            # Jalankan logika Saleko untuk mendapatkan harga
                            price_estimation = process_saleko(masked_roi)

                # Tambahkan bounding box
                color = (255, 0, 0)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} ({confidence:.2f}) | {price_estimation}"
                cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                st.write(f"Class: {class_name}, Confidence: {confidence:.2f}, Price: {price_estimation}")

    if not detected:
        st.write("Other Types - Price Not Available")
    else:
        return output_image

# Streamlit untuk upload gambar
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Tombol untuk memulai proses
if uploaded_file is not None:
    image_path = f"upload/temp_{uploaded_file.name}"  # Menyimpan gambar di folder upload
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(Image.open(image_path), caption="Uploaded Image", use_column_width=True)

    if st.button("Process"):
        output_image = predict_and_visualize(image_path)
        if output_image is not None:
            st.image(output_image[:, :, ::-1], caption="Prediction Results", use_column_width=True)

import matplotlib.pyplot as plt
import numpy as np

# Definisi centroid
centroids = np.array([
    [80, 80, 80],  # Centroid untuk warna putih
    [65, 65, 65]   # Centroid untuk warna hitam
])

# Membuat gambar warna dari centroid
colors = centroids / 255.0  # Normalisasi ke skala 0-1 untuk matplotlib
labels = ["Putih (80, 80, 80)", "Hitam (65, 65, 65)"]

# Visualisasi
plt.figure(figsize=(5, 2))
for i, color in enumerate(colors):
    plt.subplot(1, 2, i + 1)
    plt.imshow([[color]], aspect='auto')
    plt.axis('off')
    plt.title(labels[i])

plt.tight_layout()
plt.show()

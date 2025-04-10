# Matriks Grayscale
grayscale_matrix_new = np.array([
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  54, 247, 138, 245,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0, 154, 131, 207, 140, 110, 250,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0, 166, 251, 147, 160, 127,  97, 232, 207, 203, 240, 253,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0, 116, 254, 166, 154, 104, 106, 140, 125, 189, 235, 113,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0, 109, 135, 131, 123,  78, 126, 177, 102,  17,  43,  15,   0,   0,   0,   0,   0],
    [  0,   0,   0,  76,  70,  92,  99, 142,  77,  71, 137,  91,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0, 115,  77,  53,  63, 128,  35,  73, 192,  54,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0, 134,  62,  92,  38, 176,  65, 172, 185,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0, 105,  84, 154, 152, 174, 162, 148, 144,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0, 134,   0, 131, 153, 119, 164, 162,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0, 173, 128,   0,   0,   0,   0, 173, 185,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0, 108, 140,   0,   0,   0,   0, 170,  87,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0, 106, 141,   0,   0,   0,   0, 171, 170,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0, 224, 147,   0,   0,   0,   0,   0,  86,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0, 221, 161,   0,   0,   0,   0,   0, 223,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0, 210,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
])

# Menghitung total piksel
total_pixels_new = grayscale_matrix_new.size

# Menghitung black dan white pixels berdasarkan ambang batas 95
black_pixels_new = (grayscale_matrix_new <= 95).sum()
white_pixels_new = (grayscale_matrix_new > 95).sum()

# Menghitung persentase warna putih
white_percentage_new = (white_pixels_new / total_pixels_new) * 100

# Menentukan estimasi harga
if white_percentage_new > 90:
    price_estimation_new = "500 Million"
else:
    price_estimation_new = "400 Million"

# Hasil
total_pixels_new, black_pixels_new, white_pixels_new, white_percentage_new, price_estimation_new

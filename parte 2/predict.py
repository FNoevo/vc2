import os
import cv2
import numpy as np
import scipy.io
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

# Caminhos
IMG_DIR = 'part_B/test_data/images'
GT_DIR = 'part_B/test_data/ground-truth'
MODEL_PATH = 'contador_pessoas.keras'

# Carregar modelo
model = load_model(MODEL_PATH)

# Selecionar 5 imagens para teste
test_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])[:5]

for filename in test_imgs:
    img_path = os.path.join(IMG_DIR, filename)
    mat_path = os.path.join(GT_DIR, 'GT_' + filename.replace('.jpg', '.mat'))

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (128, 128))
    img_norm = np.expand_dims(img_resized / 255.0, axis=0)

    # Previsão
    pred_log = model.predict(img_norm, verbose=0)[0][0]
    pred = np.expm1(pred_log)  # desfaz o log1p

    # Ground truth real
    mat = scipy.io.loadmat(mat_path)
    gt_points = mat["image_info"][0][0][0][0][0]
    real_count = len(gt_points)

    print(f"{filename}: Previsão = {pred:.2f}, Real = {real_count}")

    # Mostrar imagem
    cv2.putText(img, f"Previsao: {int(pred)} | Real: {real_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

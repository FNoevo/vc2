import os
import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Caminhos
IMG_DIR = 'part_B/test_data/images'
GT_DIR = 'part_B/test_data/ground-truth'
MODEL_PATH = 'modelo_densitymap.keras'

IMG_SIZE = 224  # entrada do modelo
OUT_SIZE = IMG_SIZE // 8  # saída do modelo: 28x28

model = load_model(MODEL_PATH)
test_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])[:5]

for filename in test_imgs:
    img_path = os.path.join(IMG_DIR, filename)
    mat_path = os.path.join(GT_DIR, 'GT_' + filename.replace('.jpg', '.mat'))

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Erro ao ler imagem: {filename}")
        continue

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_input = np.expand_dims(img_resized.astype('float32') / 255.0, axis=0)

    pred_map = model.predict(img_input, verbose=0)[0, :, :, 0]
    pred_count = np.sum(pred_map)

    mat = scipy.io.loadmat(mat_path)
    gt_points = mat["image_info"][0][0][0][0][0]
    real_count = len(gt_points)

    print(f"{filename}: Previsão = {pred_count:.2f}, Real = {real_count}")

    # Gerar heatmap colorido
    heatmap = cv2.normalize(pred_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    overlay = cv2.addWeighted(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), 0.6, heatmap, 0.4, 0)

    # Visualização
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title("Imagem")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_map, cmap='jet')
    plt.title("Mapa de Densidade")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay\nPrevisto: {pred_count:.1f} | Real: {real_count}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

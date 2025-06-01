import os
import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === Caminhos ===
IMG_DIR = 'part_B/test_data/images'
GT_DIR = 'part_B/test_data/ground-truth'
MODEL_PATH = 'modelo_densitymap.keras'
IMG_SIZE = 224

# === Carregar modelo ===
model = load_model(MODEL_PATH)

# === Selecionar 5 imagens para teste ===
test_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])[:5]

for filename in test_imgs:
    img_path = os.path.join(IMG_DIR, filename)
    mat_path = os.path.join(GT_DIR, 'GT_' + filename.replace('.jpg', '.mat'))

    # Ler imagem
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Erro ao ler imagem: {filename}")
        continue

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_input = np.expand_dims(img_resized.astype('float32') / 255.0, axis=0)

    # Previsão
    pred_map = model.predict(img_input, verbose=0)[0, :, :, 0]
    pred_count = np.sum(pred_map)

    # Ground truth
    mat = scipy.io.loadmat(mat_path)
    gt_points = mat["image_info"][0][0][0][0][0]
    real_count = len(gt_points)

    print(f"{filename}: Previsão = {pred_count:.2f}, Real = {real_count}")

    # Gerar heatmap para visualização (mapa 28x28 → 224x224)
    heatmap = cv2.normalize(pred_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

    # Preparar imagem original para overlay
    img_rgb_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb_resized, 0.6, heatmap, 0.4, 0)

    # Mostrar resultados
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb_resized)
    plt.title("Imagem")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_map, cmap='jet')
    plt.title("Mapa de Densidade (28x28)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay\nPrevisto: {pred_count:.1f} | Real: {real_count}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

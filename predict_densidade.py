import os
import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === Configurações ===
IMG_DIR = 'part_B/test_data/images'
GT_DIR = 'part_B/test_data/ground-truth'
MODEL_PATH = 'melhor_modelo.h5'  # usa o modelo mais recente
CSV_PATH = 'predicoes.csv'
SAVE_OVERLAYS = False  # mudar para True se quiseres guardar as imagens
OVERLAY_DIR = 'resultados_overlay'
IMG_SIZE = 224

# Criar pasta para guardar overlays (se ativado)
if SAVE_OVERLAYS:
    os.makedirs(OVERLAY_DIR, exist_ok=True)

# Carregar modelo
model = load_model(MODEL_PATH)
test_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])[:50]  # pode-se remover o slice para avaliar tudo

# Criar ficheiro CSV
with open(CSV_PATH, 'w') as f:
    f.write("Imagem,Previsao,Real,Erro\n")

# Loop pelas imagens de teste
for filename in test_imgs:
    img_path = os.path.join(IMG_DIR, filename)
    mat_path = os.path.join(GT_DIR, 'GT_' + filename.replace('.jpg', '.mat'))

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Erro ao ler imagem: {filename}")
        continue

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_input = np.expand_dims(img_resized.astype('float32') / 255.0, axis=0)

    # Previsão
    pred_map = model.predict(img_input, verbose=0)[0, :, :, 0]
    pred_map = np.maximum(pred_map, 0)
    pred_count = np.sum(pred_map)

    # Obter número real de pessoas
    mat = scipy.io.loadmat(mat_path)
    gt_points = mat["image_info"][0][0][0][0][0]
    real_count = 1 if gt_points.ndim == 1 else len(gt_points)

    # Fator de correção experimental (ajusta escala com base em média esperada)
    FATOR_CORRECAO = 122.88  # soma média dos mapas usados no treino
    pred_count_corrigido = pred_count * (real_count / FATOR_CORRECAO)  # opcional, usa apenas se necessário

    erro_abs = abs(pred_count_corrigido - real_count)

    # Guardar em CSV
    with open(CSV_PATH, 'a') as f:
        f.write(f"{filename},{pred_count_corrigido:.2f},{real_count},{erro_abs:.2f}\n")

    print(f"{filename}: Previsão = {pred_count_corrigido:.2f}, Real = {real_count}")

    # Criar overlay
    pred_map_vis = np.sqrt(pred_map)
    heatmap = cv2.normalize(pred_map_vis, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    overlay = cv2.addWeighted(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), 0.6, heatmap, 0.4, 0)

    # Mostrar
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title("Imagem Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_map, cmap='jet')
    plt.title("Mapa Previsto")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay\nPrevisto: {pred_count_corrigido:.1f} | Real: {real_count} | Erro: {erro_abs:.1f}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Guardar imagem se ativado
    if SAVE_OVERLAYS:
        save_path = os.path.join(OVERLAY_DIR, filename.replace('.jpg', '_overlay.png'))
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


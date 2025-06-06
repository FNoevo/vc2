import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from tensorflow.keras.models import load_model

# Caminhos
CSV_PATH = 'avaliacao_modelo.csv'
IMG_DIR = 'part_B/test_data/images'
GT_DIR = 'part_B/test_data/ground-truth'
MODEL_PATH = 'melhor_modelo.h5'

IMG_SIZE = 224

# Carregar modelo
model = load_model(MODEL_PATH)

# Carregar CSV
df = pd.read_csv(CSV_PATH)
# Calcular erro absoluto
df["Erro"] = abs(df["Previsao"] - df["Real"])

# Ordenar e filtrar os 5 piores
df = df.sort_values(by="Erro", ascending=False).head(5)


# Loop nas 5 piores
for _, row in df.iterrows():
    filename = row['Imagem']
    real = row['Real']
    pred = row['Previsao']
    erro = row['Erro']

    img_path = os.path.join(IMG_DIR, filename)
    mat_path = os.path.join(GT_DIR, 'GT_' + filename.replace('.jpg', '.mat'))

    # Ler imagem
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_input = np.expand_dims(img_resized.astype('float32') / 255.0, axis=0)

    # Previsão
    pred_map = model.predict(img_input, verbose=0)[0, :, :, 0]
    pred_map = np.maximum(pred_map, 0)

    # Overlay
    heatmap = cv2.normalize(np.sqrt(pred_map), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    overlay = cv2.addWeighted(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), 0.6, heatmap, 0.4, 0)

    # Mostrar
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
    plt.title(f"{filename}\nReal: {real} | Prev: {pred:.1f} | Erro: {erro:.1f}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

import os
import cv2
import numpy as np
import scipy.io
import math
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Caminhos
IMG_DIR = 'part_B/test_data/images'
GT_DIR = 'part_B/test_data/ground-truth'
MODEL_PATH = 'melhor_modelo.h5'
CSV_PATH = 'avaliacao_modelo.csv'

IMG_SIZE = 224

# Carregar modelo treinado
model = load_model(MODEL_PATH)

# Lista de imagens de teste
test_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])

# Guardar CSV com cabeçalho
with open(CSV_PATH, 'w') as f:
    f.write("Imagem,Previsao,Real,Erro_Absoluto\n")

# Inicializar listas para métricas
erros_absolutos = []
erros_quadrados = []

# Avaliar cada imagem
for filename in tqdm(test_imgs):
    img_path = os.path.join(IMG_DIR, filename)
    mat_path = os.path.join(GT_DIR, 'GT_' + filename.replace('.jpg', '.mat'))

    # Ler imagem
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Erro ao ler imagem: {filename}")
        continue

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # Previsão do modelo
    pred_map = model.predict(img_input, verbose=0)[0, :, :, 0]
    pred_map = np.maximum(pred_map, 0)
    pred_count = np.sum(pred_map)

    # Contagem real a partir do .mat
    mat = scipy.io.loadmat(mat_path)
    points = mat["image_info"][0][0][0][0][0]
    real_count = 1 if points.ndim == 1 else len(points)

    # Erros
    erro_abs = abs(pred_count - real_count)
    erro_quad = (pred_count - real_count) ** 2

    erros_absolutos.append(erro_abs)
    erros_quadrados.append(erro_quad)

    # Guardar no CSV
    with open(CSV_PATH, 'a') as f:
        f.write(f"{filename},{pred_count:.2f},{real_count},{erro_abs:.2f}\n")

# Cálculo de métricas finais
mae = np.mean(erros_absolutos)
rmse = math.sqrt(np.mean(erros_quadrados))

print(f"\n📊 Avaliação Final:")
print(f"➡️  MAE  (Erro absoluto médio)     : {mae:.2f}")
print(f"➡️  RMSE (Raiz do erro quadrático) : {rmse:.2f}")
print(f"✅ Resultados guardados em {CSV_PATH}")


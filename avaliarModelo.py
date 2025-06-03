import os
import cv2
import numpy as np
import scipy.io
from tqdm import tqdm
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
IMG_DIRS = ['part_B/train_data/images', 'part_B/test_data/images']
GT_DIRS  = ['part_B/train_data/ground-truth', 'part_B/test_data/ground-truth']
model = load_model('contador_pessoas.h5')


# Carregar dados
X = []
y_true = []

for IMG_DIR, GT_DIR in zip(IMG_DIRS, GT_DIRS):
    for filename in tqdm(os.listdir(IMG_DIR)):
        if not filename.endswith('.jpg'):
            continue

        img_path = os.path.join(IMG_DIR, filename)
        mat_path = os.path.join(GT_DIR, 'GT_' + filename.replace('.jpg', '.mat'))

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        img = img / 255.0  # Normalização
        X.append(img)

        mat = scipy.io.loadmat(mat_path)
        points = mat["image_info"][0][0][0][0][0]
        y_true.append(len(points))

X = np.array(X)
y_true = np.array(y_true)

# Fazer previsões
y_pred = model.predict(X).flatten()

# Avaliar desempenho
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

# Divisão em grupos
faixas = {
    'Baixa (≤30)': (y_true <= 30),
    'Média (31–70)': ((y_true > 30) & (y_true <= 70)),
    'Alta (>70)': (y_true > 70)
}

for nome, mask in faixas.items():
    if not np.any(mask):
        continue  # evitar grupos vazios
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    mae = mean_absolute_error(y_true_f, y_pred_f)
    rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
    erro_rel = np.mean(np.abs(y_true_f - y_pred_f) / y_true_f) * 100

    print(f"\n📁 Faixa: {nome}")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  Erro Médio Relativo: {erro_rel:.2f}%")

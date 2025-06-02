import os
import cv2
import numpy as np
import scipy.io
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# Diretórios
IMG_DIR = 'part_B/train_data/images'
GT_DIR = 'part_B/train_data/ground-truth'
OUT_DIR = 'part_B/train_data/density_maps'

# Parâmetros do mapa
MAP_WIDTH = 224
MAP_HEIGHT = 224
SIGMA = 16

# Criar pasta de saída se não existir
os.makedirs(OUT_DIR, exist_ok=True)


def gerar_mapa_densidade(points, shape, original_shape):
    mapa = np.zeros(shape, dtype=np.float32)
    for point in points:
        x = min(int(point[0] * shape[1] / original_shape[1]), shape[1] - 1)
        y = min(int(point[1] * shape[0] / original_shape[0]), shape[0] - 1)
        mapa[y, x] += 1

    mapa = gaussian_filter(mapa, sigma=SIGMA)

    # Normalizar para manter soma igual ao nº de pessoas
    total = len(points)
    if mapa.sum() > 0:
        mapa *= (total / mapa.sum())

    return mapa


# Processar todas as imagens
for filename in tqdm(os.listdir(IMG_DIR)):
    if not filename.endswith('.jpg'):
        continue

    img_path = os.path.join(IMG_DIR, filename)
    mat_path = os.path.join(GT_DIR, 'GT_' + filename.replace('.jpg', '.mat'))
    out_path = os.path.join(OUT_DIR, filename.replace('.jpg', '.npy'))

    # Ler imagem original
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"❌ Erro ao ler {filename}")
        continue
    original_shape = original_img.shape[:2]

    # Ler anotações
    mat = scipy.io.loadmat(mat_path)
    points = mat["image_info"][0][0][0][0][0]

    # Gerar e guardar mapa de densidade
    mapa_densidade = gerar_mapa_densidade(points, (MAP_HEIGHT, MAP_WIDTH), original_shape)
    np.save(out_path, mapa_densidade)

    # Verificar soma do mapa
    print(f"{filename}: pessoas = {len(points)}, soma_mapa = {np.sum(mapa_densidade):.2f}")

print(f"✅ Mapas de densidade guardados em: {OUT_DIR}")
